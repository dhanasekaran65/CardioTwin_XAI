"""Global configuration: paths, schema, feature metadata."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
APP_DIR = ROOT / "appdata"          # login accounts, patient records, uploaded reports
REPORT_DIR = APP_DIR / "reports"

SEED = 42
TARGET = "target"

# UCI Heart Disease (Cleveland) style schema
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [f for f in FEATURES if f not in NUMERIC]

LABELS = {
    "age": "Age", "sex": "Sex (1=M)", "cp": "Chest pain type",
    "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting glucose >120",
    "restecg": "Resting ECG", "thalach": "Max heart rate", "exang": "Exercise angina",
    "oldpeak": "ST depression", "slope": "ST slope", "ca": "Major vessels (ca)",
    "thal": "Thalassemia",
}

# Physiological plausibility bounds used to clip twin states
BOUNDS = {"age": (20, 95), "trestbps": (80, 220), "chol": (100, 600),
          "thalach": (60, 210), "oldpeak": (0.0, 7.0), "ca": (0, 3)}

# Variables the digital twin can evolve / that interventions can modify
MODIFIABLE = ["trestbps", "chol", "thalach", "oldpeak"]

# What a patient can plausibly self-report (home BP monitor, wearable, last lab result,
# how they feel). Everything else (cp, restecg, oldpeak, slope, ca, thal) needs an ECG,
# stress test or angiogram and is entered by the clinician after those results are in.
PATIENT_EDITABLE = ["trestbps", "chol", "thalach", "fbs", "exang"]
CLINICIAN_ONLY = [f for f in FEATURES if f not in PATIENT_EDITABLE]

# Risk banding used by the digital-twin heart visual and both dashboards.
# Thresholds are on predicted probability of cardiovascular disease.
RISK_BANDS = [("Low", 0.0, 0.33, "#16A34A"), ("Moderate", 0.33, 0.66, "#D97706"),
              ("High", 0.66, 1.01, "#DC2626")]


def risk_category(p: float) -> str:
    for name, lo, hi, _ in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "High"


def risk_color(p: float) -> str:
    for name, lo, hi, color in RISK_BANDS:
        if lo <= p < hi:
            return color
    return RISK_BANDS[-1][3]


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, FIG_DIR, ARTIFACT_DIR, APP_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
