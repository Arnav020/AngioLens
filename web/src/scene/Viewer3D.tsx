import { Environment, Html, Lightformer, OrbitControls, useGLTF } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { VESSELS, type Vessel } from "../api";
import { cardiacPhase, contraction } from "../lib/beat";
import { pct, riskRGB } from "../lib/color";
import { useStore, type CameraPreset } from "../store";
import { makeFresnelMaterial, makeHaloMaterial, makeVesselMaterial } from "./shaders";

const GLB = "/anatomy/heart.glb";
useGLTF.preload(GLB);

type Anchors = Record<string, { anchor: [number, number, number] }>;

// camera presets: [position, target]  (scene units: 1 = 100 mm, heart centred at origin, +z anterior, +x patient-left)
const PRESETS: Record<CameraPreset, [THREE.Vector3, THREE.Vector3]> = {
  overview: [new THREE.Vector3(2.3, 0.9, 6.0), new THREE.Vector3(0, -0.1, 0)],
  anterior: [new THREE.Vector3(0.45, 0.35, 4.9), new THREE.Vector3(0.05, -0.05, 0)],
  left: [new THREE.Vector3(4.8, 0.5, 1.0), new THREE.Vector3(0.05, -0.05, 0)],
  posterior: [new THREE.Vector3(-0.5, 0.35, -4.9), new THREE.Vector3(0.05, -0.05, 0)],
  inferior: [new THREE.Vector3(0.6, -4.3, 2.1), new THREE.Vector3(0.05, -0.05, 0)],
};

const TISSUE: Record<string, string> = {
  heart_lv: "#6e2c2b",
  heart_rv: "#773632",
  heart_la: "#5f2e31",
  heart_ra: "#643133",
};

function useAnatomy() {
  const gltf = useGLTF(GLB);
  return useMemo(() => {
    const geo: Record<string, THREE.BufferGeometry> = {};
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) geo[m.name.replace(/_\d+$/, "")] = m.geometry;
    });
    // vessels: COLOR_0.r holds the baked flow coordinate -> dedicated float attribute
    for (const v of ["LM", ...VESSELS]) {
      const g = geo[v];
      if (!g || g.getAttribute("aFlow")) continue;
      const c = g.getAttribute("color");
      const f = new Float32Array(c.count);
      for (let i = 0; i < c.count; i++) f[i] = c.getX(i);
      g.setAttribute("aFlow", new THREE.BufferAttribute(f, 1));
    }
    // chambers: COLOR_0.rgb holds (LAD, LCX, RCA) territory weights -> keep them, repaint 'color' per patient
    for (const k of Object.keys(TISSUE)) {
      const g = geo[k];
      if (!g || g.userData.w) continue;
      const c = g.getAttribute("color");
      const w = new Float32Array(c.count * 3);
      for (let i = 0; i < c.count; i++) {
        w[3 * i] = c.getX(i);
        w[3 * i + 1] = c.getY(i);
        w[3 * i + 2] = c.getZ(i);
      }
      g.userData.w = w;
      g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(c.count * 3), 3));
    }
    return geo;
  }, [gltf]);
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

function paintChamber(g: THREE.BufferGeometry, base: string, risk: Record<Vessel, number>, focus: Vessel | null, on: boolean) {
  const w = g.userData.w as Float32Array;
  const col = g.getAttribute("color") as THREE.BufferAttribute;
  const b = new THREE.Color(base);
  const rc = VESSELS.map((v) => riskRGB(risk[v]));
  const s = VESSELS.map((v) => smooth(0.3, 0.9, risk[v]));
  const n = col.count;
  for (let i = 0; i < n; i++) {
    let r = b.r, gg = b.g, bb = b.b;
    if (on) {
      let amt = 0, cr = 0, cg = 0, cb = 0, ws = 0;
      for (let k = 0; k < 3; k++) {
        let wk = w[3 * i + k];
        if (focus) wk *= VESSELS[k] === focus ? 1.7 : 0.25;
        const a = focus && VESSELS[k] === focus ? Math.max(s[k], 0.45) : s[k];
        amt += wk * a;
        cr += wk * a * rc[k][0];
        cg += wk * a * rc[k][1];
        cb += wk * a * rc[k][2];
        ws += wk * a;
      }
      if (ws > 1e-4) {
        const m = Math.min(0.78, amt * 1.05);
        r = r + (cr / ws - r) * m;
        gg = gg + (cg / ws - gg) * m;
        bb = bb + (cb / ws - bb) * m;
      }
    }
    col.setXYZ(i, r, gg, bb);
  }
  col.needsUpdate = true;
}

function Heart() {
  const geo = useAnatomy();
  const result = useStore((s) => s.result);
  const selected = useStore((s) => s.selected);
  const hovered = useStore((s) => s.hovered);
  const layers = useStore((s) => s.layers);
  const select = useStore((s) => s.select);
  const hover = useStore((s) => s.hover);
  const pickRegion = useStore((s) => s.pickRegion);
  const heartRef = useRef<THREE.Group>(null);
  const anchors = useAnchors();

  const risk = useMemo<Record<Vessel, number>>(
    () => ({
      LAD: result?.targets.LAD.prob ?? 0.3,
      LCX: result?.targets.LCX.prob ?? 0.3,
      RCA: result?.targets.RCA.prob ?? 0.3,
    }),
    [result],
  );
  const focus = hovered ?? selected;

  const vesselMats = useMemo(() => {
    const m: Record<string, THREE.ShaderMaterial> = {};
    for (const v of ["LM", ...VESSELS]) m[v] = makeVesselMaterial();
    return m;
  }, []);
  const haloMats = useMemo(() => {
    const m: Record<string, THREE.ShaderMaterial> = {};
    for (const v of ["LM", ...VESSELS]) m[v] = makeHaloMaterial();
    return m;
  }, []);
  const chamberMats = useMemo(() => {
    const m: Record<string, THREE.MeshPhysicalMaterial> = {};
    for (const k of Object.keys(TISSUE))
      m[k] = new THREE.MeshPhysicalMaterial({
        vertexColors: true,
        roughness: 0.48,
        metalness: 0.0,
        clearcoat: 0.35,
        clearcoatRoughness: 0.35,
        sheen: 0.8,
        sheenRoughness: 0.45,
        sheenColor: new THREE.Color("#f0a8a0"),
      });
    return m;
  }, []);
  const ctx = useMemo(
    () => ({
      skin: makeFresnelMaterial("#4fb0c6", 0.36, 2.8, 0.0, 1),
      bone: makeFresnelMaterial("#cdbfa6", 0.3, 1.8, 0.03),
      lungs: makeFresnelMaterial("#7aa6d8", 0.22, 1.8, 0.02),
      aorta: new THREE.MeshPhysicalMaterial({ color: "#9c3a35", roughness: 0.4, clearcoat: 0.4, sheen: 0.5, sheenColor: new THREE.Color("#f3a59a") }),
      pulm: new THREE.MeshPhysicalMaterial({ color: "#4a5f9a", roughness: 0.45, clearcoat: 0.4 }),
      vena: new THREE.MeshPhysicalMaterial({ color: "#3d4f88", roughness: 0.45, clearcoat: 0.3 }),
      veins: new THREE.MeshPhysicalMaterial({ color: "#41579a", roughness: 0.4, clearcoat: 0.5, transparent: true, opacity: 0.85 }),
    }),
    [],
  );

  // repaint myocardium when the prediction / focus changes
  useEffect(() => {
    for (const k of Object.keys(TISSUE)) if (geo[k]) paintChamber(geo[k], TISSUE[k], risk, focus, layers.territory);
  }, [geo, risk, focus, layers.territory]);

  // vessel colours
  useEffect(() => {
    for (const v of VESSELS) {
      const [r, g, b] = riskRGB(risk[v]);
      vesselMats[v].uniforms.uColor.value.setRGB(r, g, b);
      vesselMats[v].uniforms.uRisk.value = risk[v];
    }
    const lm = Math.max(risk.LAD, risk.LCX);
    const [r, g, b] = riskRGB(lm);
    vesselMats.LM.uniforms.uColor.value.setRGB(r, g, b);
    vesselMats.LM.uniforms.uRisk.value = lm;
    for (const v of ["LM", ...VESSELS]) haloMats[v].uniforms.uColor.value.copy(vesselMats[v].uniforms.uColor.value);
  }, [risk, vesselMats, haloMats]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const hr = result?.heart_rate ?? 72;
    const c = contraction(cardiacPhase(performance.now(), hr));
    if (heartRef.current) {
      const s = 1 - 0.03 * c;
      heartRef.current.scale.set(s, 1 - 0.018 * c, s);
    }
    for (const [k, m] of Object.entries(vesselMats)) {
      m.uniforms.uTime.value = t;
      m.uniforms.uFlowOn.value = layers.flow ? 1 : 0;
      const isFocus = focus && (k === focus || (k === "LM" && (focus === "LAD" || focus === "LCX")));
      m.uniforms.uEmph.value = focus ? (isFocus ? 1 : -1) : 0;
      m.uniforms.uBeat.value = 0.0045 + 0.002 * c; // inflate thin arteries ~0.5 mm so they read at torso scale
      const rk = m.uniforms.uRisk.value as number;
      haloMats[k].uniforms.uStrength.value =
        (focus ? (isFocus ? 0.9 : 0.06) : 0.22 + 0.5 * rk) * (0.85 + 0.15 * Math.sin(t * 3 + rk * 4));
    }
    ctx.skin.uniforms.uTime.value = t;
  });

  const vesselProps = (v: string) => ({
    onPointerOver: (e: { stopPropagation: () => void }) => {
      e.stopPropagation();
      if (v !== "LM") hover(v as Vessel);
      document.body.style.cursor = "pointer";
    },
    onPointerOut: () => {
      hover(null);
      document.body.style.cursor = "";
    },
    onClick: (e: { stopPropagation: () => void }) => {
      e.stopPropagation();
      if (v !== "LM") select(selected === v ? null : (v as Vessel));
    },
  });

  return (
    <group>
      <group ref={heartRef}>
        {Object.keys(TISSUE).map((k) =>
          geo[k] ? (
            <mesh key={k} geometry={geo[k]} material={chamberMats[k]} renderOrder={1}
              onClick={(e) => {
                e.stopPropagation();
                const face = e.face;
                if (!face) return;
                const w = geo[k].userData.w as Float32Array;
                const sum = [0, 1, 2].map((j) => w[3 * face.a + j] + w[3 * face.b + j] + w[3 * face.c + j]);
                const tot = sum.reduce((a, b) => a + b, 0) || 1;
                const weights = { LAD: sum[0] / tot, LCX: sum[1] / tot, RCA: sum[2] / tot };
                const vessel = Object.entries(weights).sort((a, b) => b[1] - a[1])[0][0] as Vessel;
                pickRegion({ chamber: k, weights, vessel });
              }} />
          ) : null,
        )}
        {["LM", ...VESSELS].map((v) =>
          geo[v] ? <mesh key={v} geometry={geo[v]} material={vesselMats[v]} {...vesselProps(v)} renderOrder={2} /> : null,
        )}
        {["LM", ...VESSELS].map((v) =>
          geo[v] ? <mesh key={`h${v}`} geometry={geo[v]} material={haloMats[v]} renderOrder={8} raycast={() => null} /> : null,
        )}
        {layers.context && (
          <>
            {geo.aorta && <mesh geometry={geo.aorta} material={ctx.aorta} />}
            {geo.pulmonary_trunk && <mesh geometry={geo.pulmonary_trunk} material={ctx.pulm} />}
            {geo.vena_cava && <mesh geometry={geo.vena_cava} material={ctx.vena} />}
            {geo.veins && <mesh geometry={geo.veins} material={ctx.veins} />}
          </>
        )}
      </group>
      {layers.lungs && geo.lungs && <mesh geometry={geo.lungs} material={ctx.lungs} renderOrder={5} />}
      {layers.skeleton &&
        ["ribcage", "spine", "clavicles"].map((k) =>
          geo[k] ? <mesh key={k} geometry={geo[k]} material={ctx.bone} renderOrder={6} /> : null,
        )}
      {layers.torso && geo.skin && <mesh geometry={geo.skin} material={ctx.skin} renderOrder={7} />}
      {layers.labels && anchors && result && (
        <>
          {VESSELS.map((v) => (
            <Html key={v} position={anchors[v].anchor} center zIndexRange={[20, 0]}>
              <button
                className={`vlabel ${focus === v ? "on" : ""} ${focus && focus !== v ? "dim" : ""}`}
                style={{ ["--c" as string]: `rgb(${riskRGB(risk[v]).map((x) => Math.round(x * 255)).join(",")})` }}
                onMouseEnter={() => hover(v)}
                onMouseLeave={() => hover(null)}
                onClick={() => select(selected === v ? null : v)}
              >
                <i />
                <b>{v}</b>
                <span>{pct(risk[v])}</span>
              </button>
            </Html>
          ))}
        </>
      )}
    </group>
  );
}

function useAnchors(): Anchors | null {
  const [a, setA] = useState<Anchors | null>(null);
  useEffect(() => {
    fetch("/anatomy/anatomy.json")
      .then((r) => r.json())
      .then((j) => setA(j.vessels))
      .catch(() => setA(null));
  }, []);
  return a;
}

function CameraRig() {
  const camera = useStore((s) => s.camera);
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const cam = useThree((s) => s.camera);
  const goal = useRef<{ pos: THREE.Vector3; tgt: THREE.Vector3; t: number } | null>(null);

  useEffect(() => {
    const [p, t] = PRESETS[camera.preset];
    goal.current = { pos: p.clone(), tgt: t.clone(), t: 0 };
  }, [camera]);

  useFrame((_, dt) => {
    const g = goal.current;
    if (!g || !controls) return;
    g.t = Math.min(1, g.t + dt * 1.1);
    const k = 1 - Math.pow(1 - g.t, 3);
    cam.position.lerp(g.pos, k * 0.18 + 0.02);
    controls.target.lerp(g.tgt, k * 0.18 + 0.02);
    controls.update();
    if (g.t >= 1 && cam.position.distanceTo(g.pos) < 0.01) goal.current = null;
  });
  return null;
}

function Floor() {
  const ring = useRef<THREE.Mesh>(null);
  const grid = useMemo(() => {
    const h = new THREE.PolarGridHelper(3.2, 24, 8, 96, "#1f3b44", "#16262c");
    (h.material as THREE.Material).transparent = true;
    (h.material as THREE.Material).opacity = 0.55;
    return h;
  }, []);
  useFrame((_, dt) => {
    if (ring.current) ring.current.rotation.z += dt * 0.15;
  });
  return (
    <group position={[0, -2.8, 0]}>
      <primitive object={grid} />
      <mesh ref={ring} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[3.18, 3.22, 128, 1, 0, Math.PI * 1.4]} />
        <meshBasicMaterial color="#e9a23b" transparent opacity={0.55} />
      </mesh>
    </group>
  );
}

// Exposes the three.js state for debugging and for the scripted demo-video recorder.
function DebugHook() {
  const state = useThree();
  useEffect(() => {
    const w = window as unknown as { __angiolens: unknown; __angiolensProject: unknown };
    w.__angiolens = state;
    // scene point -> page pixel coordinates (used by the scripted demo recorder)
    w.__angiolensProject = (x: number, y: number, z: number) => {
      const v = new THREE.Vector3(x, y, z).project(state.camera);
      const r = state.gl.domElement.getBoundingClientRect();
      return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
    };
  }, [state]);
  return null;
}


export default function Viewer3D() {
  const ctrl = useRef<OrbitControlsImpl>(null);
  return (
    <Canvas
      className="viewer-canvas"
      dpr={[1, 1.6]}
      gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
      camera={{ position: [2.3, 0.9, 6.0], fov: 32, near: 0.05, far: 60 }}
    >
      <color attach="background" args={["#081216"]} />
      <fog attach="fog" args={["#081216", 9, 22]} />
      <ambientLight intensity={0.25} />
      <directionalLight position={[3, 5, 4]} intensity={1.4} color="#fff2e6" />
      <directionalLight position={[-4, -1, -3]} intensity={0.5} color="#7fb3ff" />
      <Environment resolution={128}>
        <Lightformer intensity={1.6} position={[0, 4, 3]} scale={[6, 2, 1]} color="#ffe6d6" />
        <Lightformer intensity={0.8} position={[-5, 0, 0]} rotation-y={Math.PI / 2} scale={[4, 4, 1]} color="#7fb8ff" />
        <Lightformer intensity={0.6} position={[5, -1, -2]} rotation-y={-Math.PI / 2} scale={[4, 3, 1]} color="#ffb18a" />
      </Environment>
      <Suspense fallback={<Html center><div className="loading3d">Loading anatomy…</div></Html>}>
        <Heart />
      </Suspense>
      <Floor />
      <OrbitControls
        ref={ctrl}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={1.4}
        maxDistance={11}
        autoRotate
        autoRotateSpeed={0.45}
        onStart={() => {
          if (ctrl.current) ctrl.current.autoRotate = false;
        }}
      />
      <CameraRig />
      <DebugHook />

    </Canvas>
  );
}
