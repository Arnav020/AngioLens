"""Build the 3D anatomy asset (web/public/anatomy/heart.glb + anatomy.json) from BodyParts3D.

    python scripts/build_anatomy.py

Source: "BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International" (licence updated by DBCLS on 2025-02-25).
Meshes are selected by FMA concept id, merged into named groups, decimated for
CPU-only WebGL, and two per-vertex attributes are baked into COLOR_0:

* vessels   (LAD / LCX / RCA / LM): R = normalised distance downstream of the
  coronary ostium -> drives the animated blood-flow shader.
* ventricles: (R, G, B) = soft perfusion-territory weights for (LAD, LCX, RCA)
  computed from distance to each artery, times a proximity falloff -> lets the
  viewer tint the myocardium supplied by a high-risk vessel ("territory at risk").

The result is ~200k triangles, small enough for integrated laptop graphics.
"""

from __future__ import annotations

import collections
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "bodyparts3d"
OUT = ROOT / "web" / "public" / "anatomy"
BASE_URL = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/"
ZIP_NAME = "partof_BP3D_4.0_obj_99.zip"
PARTS_NAME = "partof_element_parts.txt"

# group -> (FMA concepts, target triangle count or None to keep full detail)
VESSEL_GROUPS = {
    "LM": (["FMA3855"], None),
    "LAD": (["FMA3862", "FMA3860", "FMA3868", "FMA3872", "FMA3874", "FMA3876"], 30000),
    "LCX": (["FMA3895"], 12000),
    "RCA": (["FMA3802", "FMA3815", "FMA3818", "FMA3837", "FMA3840"], 30000),
}
EXTRA_RCA_ELEMENTS = ["FJ2670", "FJ2676"]  # right conus artery
EXTRA_VEIN_ELEMENTS = ["FJ2724", "FJ2731", "FJ2727", "FJ2728", "FJ2729"]  # small + anterior cardiac veins
CONTEXT_GROUPS = {
    "veins": (["FMA4706", "FMA4707", "FMA4712", "FMA4713"], 8000),
    "aorta": (["FMA3736", "FMA3768", "FMA87217"], 6000),
    "pulmonary_trunk": (["FMA8612"], 3000),
    "vena_cava": (["FMA4720"], 1500),
    "lungs": (["FMA7309", "FMA7310"], 22000),
    "ribcage": (["FMA7480"], 30000),
    "spine": (["FMA9140"], 12000),
    "clavicles": (["FMA13322", "FMA13323"], 3000),
    "skin": (["FMA7163"], 24000),
}
# Chamber shells. BodyParts3D models the *endocardial* (blood-pool) surface of the
# ventricles, while coronary arteries run on the *epicardium*; the measured gap is
# ~9 mm for the LV (myocardium thickness) and ~4 mm for the RV. We inflate each
# shell along smoothed normals so the arteries sit on the surface (element ids
# verified visually; concept names in BodyParts3D are ambiguous for these parts).
CHAMBERS = {
    "heart_lv": (["FJ2422"], 7.5),           # cavity of left ventricle
    "heart_rv": (["FJ2423"], 3.0),           # cavity of right ventricle
    "heart_la": (["FJ2425", "FJ2438"], 0.0), # cavity + wall of left atrium
    "heart_ra": (["FJ2424", "FJ2439"], 0.0), # cavity + wall of right atrium
}
SCALE = 0.01  # mm -> scene units (heart ~ 1.1 units wide)


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        print(f"[download] {BASE_URL + name}")
        with urllib.request.urlopen(BASE_URL + name, timeout=600) as r:  # noqa: S310 - fixed public URL
            p.write_bytes(r.read())
    return p


def concept_index():
    elems = collections.defaultdict(list)
    names = {}
    inv = collections.defaultdict(set)
    for line in fetch(PARTS_NAME).read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3:
            elems[p[0]].append(p[2])
            names[p[0]] = p[1]
            inv[p[2]].add(p[0])
    return elems, names, inv


class Loader:
    def __init__(self):
        self.z = zipfile.ZipFile(fetch(ZIP_NAME))

    def mesh(self, fj: str) -> trimesh.Trimesh:
        data = self.z.read(f"partof_BP3D_4.0_obj_99/{fj}.obj")
        return trimesh.load(io.BytesIO(data), file_type="obj", force="mesh", process=False)

    def group(self, fjs) -> trimesh.Trimesh:
        meshes = [self.mesh(f) for f in dict.fromkeys(fjs)]
        m = trimesh.util.concatenate(meshes)
        m.merge_vertices()
        return m


def decimate(m: trimesh.Trimesh, target: int | None) -> trimesh.Trimesh:
    if target is None or len(m.faces) <= target:
        return m
    v, f = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32),
                                        target_reduction=1 - target / len(m.faces))
    out = trimesh.Trimesh(v, f, process=True)
    return out


def to_scene(v: np.ndarray, center: np.ndarray) -> np.ndarray:
    """BodyParts3D (X = patient left, Y = posterior, Z = up, mm) -> three.js (y up, +z anterior)."""
    d = v - center
    return np.stack([d[:, 0], d[:, 2], -d[:, 1]], axis=1) * SCALE


def main():
    elems, names, inv = concept_index()
    L = Loader()
    groups: dict[str, trimesh.Trimesh] = {}

    # ---------------- coronary arteries ----------------
    for g, (concepts, target) in VESSEL_GROUPS.items():
        fjs = [e for c in concepts for e in elems.get(c, [])]
        if g == "RCA":
            fjs += EXTRA_RCA_ELEMENTS
        groups[g] = decimate(L.group(fjs), target)
        print(f"[vessel] {g:4s} {len(set(fjs)):3d} elements -> {len(groups[g].faces):6d} tris")

    # ---------------- heart chambers (epicardial shells) ----------------
    for g, (fjs, offset_mm) in CHAMBERS.items():
        m = L.group(fjs)
        if offset_mm > 0:
            m = m.subdivide().subdivide()
            trimesh.smoothing.filter_taubin(m, iterations=10)
            m.vertices = m.vertices + m.vertex_normals * offset_mm
            trimesh.smoothing.filter_taubin(m, iterations=10)
        groups[g] = decimate(m, 18000)
        print(f"[chamber] {g} {len(fjs)} elements, +{offset_mm} mm -> {len(groups[g].faces)} tris")

    # ---------------- context anatomy ----------------
    for g, (concepts, target) in CONTEXT_GROUPS.items():
        fjs = [e for c in concepts for e in elems.get(c, [])]
        if not fjs:
            print(f"[warn] no elements for {g}")
            continue
        if g == "veins":
            fjs += EXTRA_VEIN_ELEMENTS
        m = L.group(fjs)
        if g == "skin":  # keep only the torso shell (drop head, arms, legs)
            c = m.triangles_center
            keep = (c[:, 2] > 960) & (c[:, 2] < 1470) & (np.abs(c[:, 0]) < 182)
            m = trimesh.Trimesh(m.vertices, m.faces[keep], process=True)
            m.remove_unreferenced_vertices()
        groups[g] = decimate(m, target)
        print(f"[context] {g} {len(fjs)} elements -> {len(groups[g].faces)} tris")

    heart_parts = [groups[g] for g in groups if g.startswith("heart_")]
    center = trimesh.util.concatenate(heart_parts).bounds.mean(0)

    # ---------------- flow coordinate along each artery ----------------
    aorta_v = groups["aorta"].vertices
    aorta_tree = cKDTree(aorta_v)
    lm_d, _ = aorta_tree.query(groups["LM"].vertices)
    ostium_left = groups["LM"].vertices[np.argmin(lm_d)]
    rca_d, _ = aorta_tree.query(groups["RCA"].vertices)
    ostium_right = groups["RCA"].vertices[np.argmin(rca_d)]
    meta_vessels = {}
    flow_max = 0.0
    dists = {}
    for g in VESSEL_GROUPS:
        ost = ostium_right if g == "RCA" else ostium_left
        dists[g] = np.linalg.norm(groups[g].vertices - ost, axis=1)
        flow_max = max(flow_max, dists[g].max())
    for g in VESSEL_GROUPS:
        d = dists[g] / flow_max
        rgba = np.zeros((len(d), 4), dtype=np.uint8)
        rgba[:, 0] = np.round(d * 255)
        rgba[:, 3] = 255
        groups[g].visual = trimesh.visual.ColorVisuals(groups[g], vertex_colors=rgba)
        verts = groups[g].vertices
        # label anchor: vertex ~45 % along the vessel, nudged outward from the heart centre
        target = np.quantile(dists[g], 0.45 if g != "LM" else 0.5)
        anchor = verts[np.argmin(np.abs(dists[g] - target))]
        meta_vessels[g] = {
            "anchor": to_scene(anchor[None], center)[0].round(4).tolist(),
            "ostium": to_scene((ostium_right if g == "RCA" else ostium_left)[None], center)[0].round(4).tolist(),
            "length_mm": float(round(dists[g].max(), 1)),
            "triangles": int(len(groups[g].faces)),
        }

    # ---------------- perfusion territory weights on ventricles ----------------
    trees = {v: cKDTree(groups[v].vertices) for v in ("LAD", "LCX", "RCA")}
    for g in ("heart_lv", "heart_rv", "heart_la", "heart_ra"):
        verts = groups[g].vertices
        D = np.stack([trees[v].query(verts)[0] for v in ("LAD", "LCX", "RCA")], axis=1)
        tau = 7.0  # mm, softness of territory borders
        W = np.exp(-(D - D.min(1, keepdims=True)) / tau)
        W /= W.sum(1, keepdims=True)
        prox = np.exp(-D.min(1) / 22.0)  # fade with distance from any artery
        if g in ("heart_la", "heart_ra"):
            prox *= 0.25  # atria are not part of the AHA ventricular territory model
        rgba = np.zeros((len(verts), 4), dtype=np.uint8)
        rgba[:, :3] = np.round(W * prox[:, None] * 255)
        rgba[:, 3] = 255
        groups[g].visual = trimesh.visual.ColorVisuals(groups[g], vertex_colors=rgba)

    # ---------------- export ----------------
    scene = trimesh.Scene()
    total = 0
    for g, m in groups.items():
        m = m.copy()
        m.vertices = to_scene(np.asarray(m.vertices), center)
        _ = m.vertex_normals  # computed -> exported as NORMAL
        scene.add_geometry(m, node_name=g, geom_name=g)
        total += len(m.faces)
    OUT.mkdir(parents=True, exist_ok=True)
    glb = scene.export(file_type="glb")
    (OUT / "heart.glb").write_bytes(glb)
    bounds = {g: to_scene(m.bounds, center).round(4).tolist() for g, m in groups.items()}
    meta = {
        "source": "BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International",
        "units": "1 scene unit = 100 mm",
        "triangles": total,
        "vessels": meta_vessels,
        "bounds": bounds,
        "groups": list(groups),
    }
    (OUT / "anatomy.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {OUT / 'heart.glb'}  {len(glb) / 1e6:.1f} MB, {total} triangles")


if __name__ == "__main__":
    sys.exit(main())
