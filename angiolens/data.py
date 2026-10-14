"""Dataset loading, encoding and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import BY_ID, DROPPED_COLUMNS, FEATURES, FEATURE_IDS, LEAKAGE_COLUMNS, TARGETS

ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
UCI_URL = "https://archive.ics.uci.edu/static/public/411/extention+of+z+alizadeh+sani+dataset.zip"

POSITIVE = {"CAD": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}


def download_raw(dest: Path = RAW_XLSX) -> Path:
    """Fetch the UCI spreadsheet if it is not already present."""
    if dest.exists():
        return dest
    import io
    import urllib.request
    import zipfile

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:  # noqa: S310 - fixed public URL
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest.parent)
    return dest


def load_raw(path: Path = RAW_XLSX) -> pd.DataFrame:
    df = pd.read_excel(download_raw(path))
    df.columns = [c.strip() for c in df.columns]
    return df


def encode_raw_value(feature_id: str, value) -> float:
    """Map one raw spreadsheet value to the model's numeric encoding."""
    f = BY_ID[feature_id]
    if f.levels:
        key = str(value).strip()
        if key not in f.levels:
            raise ValueError(f"{f.raw}: unexpected value {value!r}")
        return float(f.levels[key])
    return float(value)


def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw spreadsheet -> numeric model matrix (columns = FEATURE_IDS)."""
    out = {}
    for f in FEATURES:
        col = df[f.raw]
        out[f.id] = col.map(lambda v, fid=f.id: encode_raw_value(fid, v)) if f.levels else col.astype(float)
    X = pd.DataFrame(out, index=df.index)[FEATURE_IDS]
    assert not X.isna().any().any(), "unexpected missing values after encoding"
    return X


def encode_targets(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {t: (df[col].astype(str).str.strip() == POSITIVE[t]).astype(int) for t, col in TARGETS.items()},
        index=df.index,
    )


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (X, Y, raw). X never contains LAD/LCX/RCA/Cath (leakage guard)."""
    raw = load_raw()
    X = encode_frame(raw)
    Y = encode_targets(raw)
    leaked = set(X.columns) & {c.lower() for c in LEAKAGE_COLUMNS}
    assert not leaked, f"target leakage: {leaked}"
    return X, Y, raw


def dataset_audit(raw: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """Facts about the data reported in the documentation."""
    any_vessel = Y[["LAD", "LCX", "RCA"]].max(axis=1)
    inconsistent = raw.index[(any_vessel != Y["CAD"])].tolist()
    return {
        "n_patients": int(len(raw)),
        "n_raw_columns": int(raw.shape[1]),
        "n_model_features": len(FEATURE_IDS),
        "dropped_columns": DROPPED_COLUMNS,
        "missing_values": int(raw.isna().sum().sum()),
        "prevalence": {t: float(Y[t].mean()) for t in Y.columns},
        "n_diseased_vessels": {int(k): int(v) for k, v in Y[["LAD", "LCX", "RCA"]].sum(axis=1).value_counts().sort_index().items()},
        "label_inconsistencies": [
            {"row": int(i), "Cath": str(raw.loc[i, "Cath"]),
             "LAD": str(raw.loc[i, "LAD"]), "LCX": str(raw.loc[i, "LCX"]), "RCA": str(raw.loc[i, "RCA"])}
            for i in inconsistent
        ],
    }


def encode_record(rec: dict, defaults: dict | None = None) -> dict:
    """One uploaded row -> {feature id: value}. Accepts feature ids or raw UCI column names.

    Missing features are filled from `defaults` (cohort medians) and reported back so the UI can flag them.
    """
    norm = {str(k).strip(): v for k, v in rec.items()}
    out, filled = {}, []
    for f in FEATURES:
        if f.id in norm and norm[f.id] not in ("", None):
            out[f.id] = float(norm[f.id])
        elif f.raw in norm and norm[f.raw] not in ("", None):
            out[f.id] = encode_raw_value(f.id, norm[f.raw]) if f.levels else float(norm[f.raw])
        elif defaults is not None:
            out[f.id] = float(defaults[f.id])
            filled.append(f.id)
        else:
            raise KeyError(f"missing feature '{f.id}' (raw column '{f.raw}')")
    to_feature_vector(out)  # range validation
    return {"values": out, "filled_with_defaults": filled}


def to_feature_vector(values: dict) -> np.ndarray:
    """Validated API/UI dict (feature ids -> numbers) -> model row."""
    row = []
    for f in FEATURES:
        if f.id not in values:
            raise KeyError(f"missing feature '{f.id}'")
        v = float(values[f.id])
        if not np.isfinite(v):
            raise ValueError(f"{f.id}: not a finite number")
        if f.kind == "binary" and v not in (0.0, 1.0):
            raise ValueError(f"{f.id}: binary feature must be 0 or 1")
        if not (f.min <= v <= f.max):
            raise ValueError(f"{f.id}: {v} outside valid range [{f.min}, {f.max}]")
        row.append(v)
    return np.asarray(row, dtype=float)
