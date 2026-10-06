"""Shared helpers for the Streamlit dashboards: model loading, vitals input widgets,
prediction + explanation, and twin reconstruction from a patient's logged history.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from . import data as D
from .config import ARTIFACT_DIR, FEATURES, LABELS, TARGET, risk_category, risk_color
from .explain import ShapExplainer
from .model import RiskModel
from .twin import PatientTwin

CP_OPTIONS = {0: "Asymptomatic", 1: "Non-anginal pain", 2: "Atypical angina", 3: "Typical angina"}
RESTECG_OPTIONS = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
SLOPE_OPTIONS = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_OPTIONS = {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}


def inject_custom_css() -> None:
    """Inject global modern clinical healthcare styling into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* -------------------------------------------------------------
           Healthcare Design Tokens & Variables
        ------------------------------------------------------------- */
        :root {
            --primary-blue: #2563EB;
            --primary-blue-hover: #1D4ED8;
            --secondary-teal: #0F766E;
            --page-bg: #F7F9FC;
            --card-bg: #FFFFFF;
            --main-text: #1E293B;
            --secondary-text: #64748B;
            --border-color: #E2E8F0;
            --hover-light-blue: #EFF6FF;
            --light-teal-bg: #F0FDFA;
            --risk-low: #16A34A;
            --risk-mod: #D97706;
            --risk-high: #DC2626;
        }

        /* -------------------------------------------------------------
           Page & Body Background & Typography
        ------------------------------------------------------------- */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: #F7F9FC !important;
            color: #1E293B !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* -------------------------------------------------------------
           Sidebar Styling
        ------------------------------------------------------------- */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"] * {
            color: #1E293B;
        }
        [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] caption {
            color: #64748B !important;
        }

        /* -------------------------------------------------------------
           Headings & Typography
        ------------------------------------------------------------- */
        h1, h2, h3, h4, h5, h6 {
            color: #1E293B !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
        }
        p, span, label, div {
            color: #1E293B;
        }
        .stCaption, small, [data-testid="stCaptionContainer"] {
            color: #64748B !important;
        }

        /* -------------------------------------------------------------
           Buttons
        ------------------------------------------------------------- */
        /* Primary Buttons */
        button[kind="primary"], .stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: 1px solid #2563EB !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15) !important;
            transition: all 0.15s ease-in-out !important;
        }
        button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
            background-color: #1D4ED8 !important;
            border-color: #1D4ED8 !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 4px rgba(29, 78, 216, 0.25) !important;
        }

        /* Secondary / Standard Buttons */
        button[kind="secondary"], .stButton > button, button[data-testid="stBaseButton-secondary"] {
            background-color: #FFFFFF !important;
            color: #2563EB !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.15s ease-in-out !important;
        }
        button[kind="secondary"]:hover, .stButton > button:hover, button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #EFF6FF !important;
            border-color: #2563EB !important;
            color: #2563EB !important;
        }

        /* -------------------------------------------------------------
           Tabs
        ------------------------------------------------------------- */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 1px solid #E2E8F0 !important;
            gap: 6px !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            color: #64748B !important;
            background-color: transparent !important;
            border: none !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            border-bottom: 2px solid transparent !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"]:hover {
            color: #1E293B !important;
            background-color: #EFF6FF !important;
            border-radius: 6px 6px 0 0 !important;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            color: #2563EB !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #2563EB !important;
            background-color: transparent !important;
        }

        /* -------------------------------------------------------------
           Metrics / KPI Cards
        ------------------------------------------------------------- */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
        }
        [data-testid="stMetricValue"] {
            color: #1E293B !important;
            font-weight: 700 !important;
            font-size: 1.6rem !important;
        }

        /* -------------------------------------------------------------
           Expanders & Form Containers
        ------------------------------------------------------------- */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
            margin-bottom: 12px !important;
        }
        [data-testid="stExpander"] details summary {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        [data-testid="stForm"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }

        /* -------------------------------------------------------------
           Inputs, Selects, Sliders
        ------------------------------------------------------------- */
        input[type="text"], input[type="password"], input[type="number"], .stTextInput > div > div > input {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }
        input:focus, .stTextInput > div > div > input:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
        }
        [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
        }

        /* -------------------------------------------------------------
           Data Tables
        ------------------------------------------------------------- */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            background-color: #FFFFFF !important;
            overflow: hidden !important;
        }

        /* -------------------------------------------------------------
           Dividers
        ------------------------------------------------------------- */
        hr {
            border-color: #E2E8F0 !important;
            margin: 20px 0 !important;
        }

        /* -------------------------------------------------------------
           Custom Clinical Cards & Badges
        ------------------------------------------------------------- */
        .clinical-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            margin-bottom: 16px;
        }
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.875rem;
            letter-spacing: 0.3px;
        }
        .risk-badge-low {
            background-color: #F0FDF4;
            color: #16A34A;
            border: 1px solid #BBF7D0;
        }
        .risk-badge-moderate {
            background-color: #FFFBEB;
            color: #D97706;
            border: 1px solid #FDE68A;
        }
        .risk-badge-high {
            background-color: #FEF2F2;
            color: #DC2626;
            border: 1px solid #FECACA;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(cat: str, extra_text: str = "") -> str:
    """Return clean HTML for a clinical risk badge."""
    c_lower = cat.lower()
    cls = f"risk-badge risk-badge-{c_lower}"
    text = f"{cat.upper()} RISK" if not extra_text else f"{cat.upper()} RISK — {extra_text}"
    return f'<span class="{cls}">{text}</span>'


@st.cache_resource(show_spinner="Loading risk model...")
def load_model_bundle():
    path = ARTIFACT_DIR / "champion_latest.joblib"
    if not path.exists():
        path = ARTIFACT_DIR / "champion.joblib"
    if path.exists():
        model = RiskModel.load(path)
    else:
        df = D.generate_synthetic(3000)
        model = RiskModel().fit(df[FEATURES], df[TARGET])
    return model, ShapExplainer(model)


@st.cache_resource
def reference_cohort():
    return D.generate_synthetic(600, seed=7)


def model_version() -> int:
    reg = ARTIFACT_DIR / "model_registry.json"
    if reg.exists():
        import json
        try:
            entries = json.loads(reg.read_text())
            return entries[-1]["version"]
        except Exception:
            pass
    return 1


def format_last_assessment(ts: str | None) -> str:
    if not ts:
        return "Today"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if dt.date() == now.date():
            return f"Today, {dt.strftime('%H:%M')}"
        return dt.strftime("%d %b %Y")
    except Exception:
        return str(ts)[:10]


def grouped_doctor_vitals_form(vitals: dict, key_prefix: str) -> dict:
    """Grouped 13-feature clinical update form for clinicians."""
    out = dict(vitals)

    st.markdown("##### 👤 Patient Information")
    c1, c2 = st.columns(2)
    out["age"] = c1.slider("Age", 20, 95, int(vitals.get("age", 50)), key=f"{key_prefix}_age")
    sex_val = vitals.get("sex", 1)
    out["sex"] = 1 if c2.selectbox("Sex", ["Male", "Female"], index=0 if sex_val else 1, key=f"{key_prefix}_sex") == "Male" else 0

    st.markdown("##### 🩺 Clinical Measurements")
    m1, m2, m3 = st.columns(3)
    out["trestbps"] = m1.slider("Resting BP (mmHg)", 80, 220, int(vitals.get("trestbps", 120)), key=f"{key_prefix}_trestbps")
    out["chol"] = m2.slider("Cholesterol (mg/dL)", 100, 600, int(vitals.get("chol", 200)), key=f"{key_prefix}_chol")
    out["thalach"] = m3.slider("Maximum Heart Rate (bpm)", 60, 220, int(vitals.get("thalach", 150)), key=f"{key_prefix}_thalach")

    m4, m5 = st.columns(2)
    fbs_bool = bool(vitals.get("fbs", 0))
    out["fbs"] = 1 if m4.selectbox("Fasting Blood Sugar", ["Normal (≤ 120 mg/dL)", "Elevated (> 120 mg/dL)"], index=1 if fbs_bool else 0, key=f"{key_prefix}_fbs") == "Elevated (> 120 mg/dL)" else 0
    out["oldpeak"] = m5.slider("ST Depression (oldpeak)", 0.0, 6.5, float(vitals.get("oldpeak", 0.0)), 0.1, key=f"{key_prefix}_oldpeak")

    st.markdown("##### 🔬 Clinical Findings")
    f1, f2 = st.columns(2)
    cp_val = int(vitals.get("cp", 0))
    out["cp"] = f1.selectbox("Chest Pain Type", list(CP_OPTIONS), format_func=lambda k: CP_OPTIONS[k],
                             index=list(CP_OPTIONS).index(cp_val) if cp_val in CP_OPTIONS else 0, key=f"{key_prefix}_cp")

    recg_val = int(vitals.get("restecg", 0))
    out["restecg"] = f2.selectbox("Resting ECG", list(RESTECG_OPTIONS), format_func=lambda k: RESTECG_OPTIONS[k],
                                 index=list(RESTECG_OPTIONS).index(recg_val) if recg_val in RESTECG_OPTIONS else 0, key=f"{key_prefix}_restecg")

    f3, f4 = st.columns(2)
    ex_val = bool(vitals.get("exang", 0))
    out["exang"] = 1 if f3.selectbox("Exercise Angina", ["No", "Yes"], index=1 if ex_val else 0, key=f"{key_prefix}_exang") == "Yes" else 0

    slope_val = int(vitals.get("slope", 1))
    out["slope"] = f4.selectbox("ST Slope", list(SLOPE_OPTIONS), format_func=lambda k: SLOPE_OPTIONS[k],
                               index=list(SLOPE_OPTIONS).index(slope_val) if slope_val in SLOPE_OPTIONS else 0, key=f"{key_prefix}_slope")

    f5, f6 = st.columns(2)
    ca_val = int(vitals.get("ca", 0))
    out["ca"] = f5.selectbox("Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3],
                             index=ca_val if ca_val in [0, 1, 2, 3] else 0, key=f"{key_prefix}_ca")

    thal_val = int(vitals.get("thal", 2))
    out["thal"] = f6.selectbox("Thalassemia", list(THAL_OPTIONS), format_func=lambda k: THAL_OPTIONS[k],
                               index=list(THAL_OPTIONS).index(thal_val) if thal_val in THAL_OPTIONS else 1, key=f"{key_prefix}_thal")

    return out


def patient_health_data_form(vitals: dict, key_prefix: str = "pat") -> dict:
    """Patient-friendly health input form without confusing medical encodings."""
    out = dict(vitals)
    c1, c2 = st.columns(2)
    out["trestbps"] = c1.number_input("Blood Pressure (mmHg)", 80, 220, int(vitals.get("trestbps", 120)), step=1, key=f"{key_prefix}_trestbps")
    out["chol"] = c2.number_input("Cholesterol (mg/dL)", 100, 500, int(vitals.get("chol", 200)), step=1, key=f"{key_prefix}_chol")

    c3, c4 = st.columns(2)
    out["thalach"] = c3.number_input("Maximum Heart Rate (bpm)", 60, 220, int(vitals.get("thalach", 150)), step=1, key=f"{key_prefix}_thalach")
    fbs_bool = bool(vitals.get("fbs", 0))
    out["fbs"] = 1 if c4.selectbox("Fasting Blood Sugar", ["Normal (≤ 120 mg/dL)", "High (> 120 mg/dL)"],
                                  index=1 if fbs_bool else 0, key=f"{key_prefix}_fbs") == "High (> 120 mg/dL)" else 0

    ex_bool = bool(vitals.get("exang", 0))
    out["exang"] = 1 if st.selectbox("Chest Discomfort / Angina During Exercise", ["No", "Yes"],
                                     index=1 if ex_bool else 0, key=f"{key_prefix}_exang") == "Yes" else 0
    return out


def vitals_form(vitals: dict, *, editable_fields: list[str], key_prefix: str,
                columns: int = 2) -> dict:
    """Render input widgets for `editable_fields` seeded with current `vitals`."""
    out = dict(vitals)
    cols = st.columns(columns)
    i = 0
    for f in FEATURES:
        if f not in editable_fields:
            continue
        c = cols[i % columns]; i += 1
        v = vitals.get(f)
        key = f"{key_prefix}_{f}"
        if f == "age":
            out[f] = c.slider(LABELS[f], 20, 90, int(v), key=key)
        elif f == "sex":
            out[f] = 1 if c.selectbox(LABELS[f], ["Male", "Female"], index=0 if v else 1, key=key) == "Male" else 0
        elif f == "cp":
            out[f] = c.selectbox(LABELS[f], list(CP_OPTIONS), format_func=lambda k: CP_OPTIONS[k],
                                 index=list(CP_OPTIONS).index(int(v)), key=key)
        elif f == "trestbps":
            out[f] = c.slider(f"{LABELS[f]} (mmHg)", 90, 200, int(v), key=key)
        elif f == "chol":
            out[f] = c.slider(f"{LABELS[f]} (mg/dl)", 120, 480, int(v), key=key)
        elif f == "fbs":
            out[f] = int(c.checkbox(f"{LABELS[f]}", value=bool(v), key=key))
        elif f == "restecg":
            out[f] = c.selectbox(LABELS[f], list(RESTECG_OPTIONS), format_func=lambda k: RESTECG_OPTIONS[k],
                                 index=list(RESTECG_OPTIONS).index(int(v)), key=key)
        elif f == "thalach":
            out[f] = c.slider(f"{LABELS[f]} (bpm)", 80, 205, int(v), key=key)
        elif f == "exang":
            out[f] = int(c.checkbox("Chest pain / angina during exercise", value=bool(v), key=key))
        elif f == "oldpeak":
            out[f] = c.slider(f"{LABELS[f]} (ECG)", 0.0, 6.0, float(v), 0.1, key=key)
        elif f == "slope":
            out[f] = c.selectbox(LABELS[f], list(SLOPE_OPTIONS), format_func=lambda k: SLOPE_OPTIONS[k],
                                 index=list(SLOPE_OPTIONS).index(int(v)), key=key)
        elif f == "ca":
            out[f] = c.selectbox(LABELS[f], [0, 1, 2, 3], index=int(v), key=key)
        elif f == "thal":
            out[f] = c.selectbox(LABELS[f], list(THAL_OPTIONS), format_func=lambda k: THAL_OPTIONS[k],
                                 index=list(THAL_OPTIONS).index(int(v)), key=key)
    return out


def predict_and_explain(model, explainer, vitals: dict):
    row = pd.DataFrame([vitals])[FEATURES]
    risk = float(model.predict_proba(row)[0])
    table = explainer.local_table(row)
    narrative = explainer.narrative(row)
    return risk, risk_category(risk), table, narrative, row


def build_twin(username: str, record: dict, model) -> PatientTwin:
    """Reconstruct a PatientTwin by replaying a patient's logged history, so
    personalised progression rates reflect what has actually been observed."""
    hist = record["history"]
    init = hist[0]["vitals"]
    twin = PatientTwin(username, init, model)
    for i, h in enumerate(hist[1:], start=1):
        twin.observe(i * 0.25, h["vitals"])
    return twin


def live_clock(label: str = "Live") -> None:
    st.caption(f"🟢 {label} · last updated {time.strftime('%H:%M:%S')}")

