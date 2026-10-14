"""Single source of truth for clinical features.

Every feature the model consumes is declared here once, with its raw column
name, clinical group, unit, valid input range and adult reference range.
The training pipeline, the API validation layer, the physiological breakdown
and the web form are all generated from this registry, so adding a new
clinical feature means adding one entry here (plus data) - nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Kind = Literal["binary", "numeric", "ordinal"]

GROUPS = {
    "demographic": "Demographics & history",
    "symptom": "Symptoms & examination",
    "ecg": "ECG",
    "lab": "Laboratory",
    "echo": "Echocardiography",
}

# Prediction targets. These columns are NEVER used as model inputs (leakage guard).
TARGETS = {
    "CAD": "Cath",  # overall CAD by angiography (>=50% narrowing in any vessel)
    "LAD": "LAD",
    "LCX": "LCX",
    "RCA": "RCA",
}
LEAKAGE_COLUMNS = frozenset(TARGETS.values())

# Raw columns removed before modelling, with the reason (reported in the docs).
DROPPED_COLUMNS = {
    "Exertional CP": "constant (all 303 patients = 'N'), carries no information",
}


@dataclass(frozen=True)
class Feature:
    id: str  # snake_case id used by API / UI / model
    raw: str  # column name in the UCI spreadsheet
    label: str
    group: str
    kind: Kind
    unit: str = ""
    min: float = 0.0
    max: float = 1.0
    step: float = 1.0
    normal: tuple[float, float] | None = None  # adult reference range (numeric only)
    modifiable: bool = False  # can plausibly change with lifestyle/therapy (what-if UI)
    description: str = ""
    levels: dict[str, int] = field(default_factory=dict)  # raw label -> code (ordinal / mapped)
    level_labels: tuple[str, ...] = ()  # human labels for ordinal codes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["normal"] = list(self.normal) if self.normal else None
        return d


YN = {"N": 0, "Y": 1}

FEATURES: list[Feature] = [
    # ---------------- demographics & history ----------------
    Feature("age", "Age", "Age", "demographic", "numeric", "years", 18, 100, 1, None, False, "Age at presentation"),
    Feature("sex_male", "Sex", "Male sex", "demographic", "binary", description="1 = male, 0 = female",
            levels={"Male": 1, "Fmale": 0, "Female": 0}),
    Feature("weight", "Weight", "Weight", "demographic", "numeric", "kg", 30, 200, 1, None, True),
    Feature("height", "Length", "Height", "demographic", "numeric", "cm", 120, 220, 1),
    Feature("bmi", "BMI", "Body-mass index", "demographic", "numeric", "kg/m²", 12, 60, 0.1, (18.5, 25.0), True),
    Feature("dm", "DM", "Diabetes mellitus", "demographic", "binary"),
    Feature("htn", "HTN", "Hypertension", "demographic", "binary"),
    Feature("current_smoker", "Current Smoker", "Current smoker", "demographic", "binary", modifiable=True),
    Feature("ex_smoker", "EX-Smoker", "Ex-smoker", "demographic", "binary"),
    Feature("fh", "FH", "Family history of CAD", "demographic", "binary"),
    Feature("obesity", "Obesity", "Obesity (BMI > 25)", "demographic", "binary", levels=YN, modifiable=True),
    Feature("crf", "CRF", "Chronic renal failure", "demographic", "binary", levels=YN),
    Feature("cva", "CVA", "Prior stroke (CVA)", "demographic", "binary", levels=YN),
    Feature("airway_disease", "Airway disease", "Airway disease", "demographic", "binary", levels=YN),
    Feature("thyroid_disease", "Thyroid Disease", "Thyroid disease", "demographic", "binary", levels=YN),
    Feature("chf", "CHF", "Congestive heart failure", "demographic", "binary", levels=YN),
    Feature("dlp", "DLP", "Dyslipidaemia", "demographic", "binary", levels=YN),
    # ---------------- symptoms & examination ----------------
    Feature("bp", "BP", "Systolic blood pressure", "symptom", "numeric", "mmHg", 70, 240, 1, (90, 130), True),
    Feature("pr", "PR", "Pulse rate", "symptom", "numeric", "bpm", 30, 180, 1, (60, 100), True),
    Feature("edema", "Edema", "Peripheral oedema", "symptom", "binary"),
    Feature("weak_peripheral_pulse", "Weak Peripheral Pulse", "Weak peripheral pulse", "symptom", "binary", levels=YN),
    Feature("lung_rales", "Lung rales", "Lung rales", "symptom", "binary", levels=YN),
    Feature("systolic_murmur", "Systolic Murmur", "Systolic murmur", "symptom", "binary", levels=YN),
    Feature("diastolic_murmur", "Diastolic Murmur", "Diastolic murmur", "symptom", "binary", levels=YN),
    Feature("typical_chest_pain", "Typical Chest Pain", "Typical chest pain", "symptom", "binary"),
    Feature("dyspnea", "Dyspnea", "Dyspnoea", "symptom", "binary", levels=YN),
    Feature("function_class", "Function Class", "Functional class", "symptom", "ordinal", "", 0, 4, 1,
            description="Functional (angina/dyspnoea) class, 0 = none … 4 = symptoms at rest",
            level_labels=("0 – none", "I", "II", "III", "IV")),
    Feature("atypical", "Atypical", "Atypical chest pain", "symptom", "binary", levels=YN),
    Feature("nonanginal", "Nonanginal", "Non-anginal chest pain", "symptom", "binary", levels=YN),
    Feature("lowth_ang", "LowTH Ang", "Low-threshold angina", "symptom", "binary", levels=YN),
    # ---------------- ECG ----------------
    Feature("q_wave", "Q Wave", "Pathological Q wave", "ecg", "binary"),
    Feature("st_elevation", "St Elevation", "ST elevation", "ecg", "binary"),
    Feature("st_depression", "St Depression", "ST depression", "ecg", "binary"),
    Feature("t_inversion", "Tinversion", "T-wave inversion", "ecg", "binary"),
    Feature("lvh", "LVH", "Left-ventricular hypertrophy", "ecg", "binary", levels=YN),
    Feature("poor_r_progression", "Poor R Progression", "Poor R-wave progression", "ecg", "binary", levels=YN),
    Feature("bbb_lbbb", "BBB", "Left bundle-branch block", "ecg", "binary", levels={"LBBB": 1, "RBBB": 0, "N": 0}),
    Feature("bbb_rbbb", "BBB", "Right bundle-branch block", "ecg", "binary", levels={"RBBB": 1, "LBBB": 0, "N": 0}),
    # ---------------- laboratory ----------------
    Feature("fbs", "FBS", "Fasting blood sugar", "lab", "numeric", "mg/dL", 40, 600, 1, (70, 100), True),
    Feature("cr", "CR", "Creatinine", "lab", "numeric", "mg/dL", 0.2, 10, 0.1, (0.6, 1.3)),
    Feature("tg", "TG", "Triglycerides", "lab", "numeric", "mg/dL", 20, 1500, 1, (0, 150), True),
    Feature("ldl", "LDL", "LDL cholesterol", "lab", "numeric", "mg/dL", 10, 400, 1, (0, 130), True),
    Feature("hdl", "HDL", "HDL cholesterol", "lab", "numeric", "mg/dL", 5, 150, 0.1, (40, 150), True),
    Feature("bun", "BUN", "Blood urea nitrogen", "lab", "numeric", "mg/dL", 2, 150, 1, (7, 20)),
    Feature("esr", "ESR", "Erythrocyte sedimentation rate", "lab", "numeric", "mm/h", 0, 150, 1, (0, 20)),
    Feature("hb", "HB", "Haemoglobin", "lab", "numeric", "g/dL", 5, 22, 0.1, (12, 17.5)),
    Feature("k", "K", "Potassium", "lab", "numeric", "mEq/L", 2, 8, 0.1, (3.5, 5.0)),
    Feature("na", "Na", "Sodium", "lab", "numeric", "mEq/L", 110, 170, 1, (135, 145)),
    Feature("wbc", "WBC", "White blood cells", "lab", "numeric", "cells/µL", 1000, 40000, 100, (4000, 11000)),
    Feature("lymph", "Lymph", "Lymphocytes", "lab", "numeric", "%", 0, 100, 1, (20, 40)),
    Feature("neut", "Neut", "Neutrophils", "lab", "numeric", "%", 0, 100, 1, (40, 70)),
    Feature("plt", "PLT", "Platelets", "lab", "numeric", "×10³/µL", 10, 1000, 1, (150, 450)),
    # ---------------- echocardiography ----------------
    Feature("ef_tte", "EF-TTE", "Ejection fraction (TTE)", "echo", "numeric", "%", 5, 80, 1, (50, 75)),
    Feature("region_rwma", "Region RWMA", "Regions with wall-motion abnormality", "echo", "ordinal", "", 0, 4, 1,
            description="Number/extent of myocardial regions with regional wall-motion abnormality (0 = none)",
            level_labels=("0 – none", "1", "2", "3", "4")),
    Feature("vhd", "VHD", "Valvular heart disease", "echo", "ordinal", "", 0, 3, 1,
            levels={"N": 0, "mild": 1, "Moderate": 2, "Severe": 3},
            level_labels=("None", "Mild", "Moderate", "Severe")),
]

FEATURE_IDS: list[str] = [f.id for f in FEATURES]
BY_ID: dict[str, Feature] = {f.id: f for f in FEATURES}

assert not (set(f.raw for f in FEATURES) & LEAKAGE_COLUMNS), "target column used as a feature"
assert len(FEATURE_IDS) == len(set(FEATURE_IDS))


def schema_payload() -> dict:
    """JSON-serialisable schema consumed by the web UI."""
    return {
        "groups": GROUPS,
        "features": [f.to_dict() for f in FEATURES],
        "targets": list(TARGETS),
    }
