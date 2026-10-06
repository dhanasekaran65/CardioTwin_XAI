"""Data loading, synthetic cohort generation (with controllable drift), splitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, SEED, TARGET

_ALIASES = {"thalch": "thalach", "num": TARGET, "condition": TARGET, "output": TARGET,
            "cholesterol": "chol", "sex_male": "sex"}
_STR_MAPS = {
    "sex": {"male": 1, "female": 0},
    "cp": {"asymptomatic": 0, "non-anginal": 1, "atypical angina": 2, "typical angina": 3},
    "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
    "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal": {"normal": 1, "fixed defect": 2, "reversable defect": 3, "reversible defect": 3},
}


def generate_synthetic(n: int = 3000, seed: int = SEED, drift: float = 0.0,
                       missing_rate: float = 0.01) -> pd.DataFrame:
    """UCI-style synthetic cohort.

    `drift` >= 0 simulates non-stationarity: covariate shift (older, higher BP/chol)
    and concept shift (cholesterol/BP become stronger risk drivers).
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(54 + 6 * drift, 9, n), 29, 80)
    sex = rng.binomial(1, 0.65, n)
    cp = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.17, 0.28, 0.10])  # 0 = asymptomatic (highest risk)
    trestbps = np.clip(112 + 0.35 * (age - 50) + rng.normal(0, 13, n) + 8 * drift, 90, 200)
    chol = np.clip(200 + 0.7 * (age - 50) + rng.normal(0, 38, n) + 10 * drift, 120, 480)
    fbs = rng.binomial(1, np.clip(0.08 + 0.004 * (age - 40) + 0.05 * drift, 0, 1))
    restecg = rng.choice([0, 1, 2], n, p=[0.5, 0.47, 0.03])
    thalach = np.clip(205 - 0.75 * age + rng.normal(0, 14, n), 80, 205)
    exang = rng.binomial(1, 0.15 + 0.25 * (cp == 0))
    oldpeak = np.clip(rng.gamma(1.3, 0.7, n) + 0.3 * exang, 0, 6)
    slope = rng.choice([0, 1, 2], n, p=[0.08, 0.47, 0.45])
    ca = np.minimum(3, rng.poisson(np.clip(0.012 * (age - 30), 0, None)))
    thal = rng.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])  # 3 = reversible defect

    w_chol = 0.008 * (1 + 3 * drift)
    w_bp = 0.018 * (1 + 2 * drift)
    z = (-1.6 + 0.045 * (age - 54) + 0.7 * sex + 0.9 * (cp == 0) - 0.4 * (cp == 2)
         + w_bp * (trestbps - 130) + w_chol * (chol - 230) + 0.5 * fbs
         + 0.25 * (restecg == 1) - 0.03 * (thalach - 150) + 0.9 * exang
         + 0.55 * oldpeak - 0.4 * (slope == 2) + 0.8 * ca + 0.9 * (thal == 3)
         + rng.normal(0, 0.4, n))
    y = rng.binomial(1, 1 / (1 + np.exp(-z)))

    df = pd.DataFrame(dict(age=age.round(), sex=sex, cp=cp, trestbps=trestbps.round(),
                           chol=chol.round(), fbs=fbs, restecg=restecg,
                           thalach=thalach.round(), exang=exang, oldpeak=oldpeak.round(1),
                           slope=slope, ca=ca, thal=thal, **{TARGET: y}))
    if missing_rate > 0:  # realistic missingness -> exercises the imputer
        for col in ("chol", "thalach", "oldpeak"):
            df.loc[rng.random(n) < missing_rate, col] = np.nan
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a real UCI/Kaggle heart-disease CSV and coerce it to the project schema."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _ALIASES.items() if k in df.columns})
    for col, mapping in _STR_MAPS.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map(mapping)
    for col in ("fbs", "exang"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0})
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def load_dataset(path: str | Path | None = None, n_synthetic: int = 3000,
                 seed: int = SEED) -> pd.DataFrame:
    return load_csv(path) if path else generate_synthetic(n_synthetic, seed)


def split(df: pd.DataFrame, seed: int = SEED):
    """Stratified 70/15/15 -> (train, val, test) DataFrames."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
