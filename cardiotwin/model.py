"""Risk model: median imputation + tree ensemble (SHAP-TreeExplainer compatible)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer

from .config import FEATURES, SEED

try:  # optional dependency
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

# Clinically motivated monotonic constraints (+1: risk cannot decrease as feature rises)
MONOTONE = {"age": 1, "trestbps": 1, "chol": 1, "thalach": -1, "oldpeak": 1, "ca": 1}

_DEFAULTS = {
    "xgb": dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.9, n_jobs=2, verbosity=0),
    "gb": dict(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8),
    "rf": dict(n_estimators=300, max_depth=8, min_samples_leaf=5, n_jobs=-1),
}


class RiskModel:
    def __init__(self, model_type: str = "xgb", params: dict | None = None, seed: int = SEED):
        if model_type not in _DEFAULTS:
            raise ValueError("model_type must be 'xgb', 'gb' or 'rf'")
        if model_type == "xgb" and XGBClassifier is None:
            raise ImportError("xgboost is not installed; use --model gb or `pip install xgboost`")
        self.model_type = model_type
        self.params = {**_DEFAULTS[model_type], **(params or {})}
        self.seed = seed
        self.imputer: SimpleImputer | None = None
        self.estimator = None

    def _make(self):
        if self.model_type == "xgb":
            mc = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in FEATURES) + ")"
            return XGBClassifier(random_state=self.seed, monotone_constraints=mc, **self.params)
        cls = GradientBoostingClassifier if self.model_type == "gb" else RandomForestClassifier
        return cls(random_state=self.seed, **self.params)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.imputer.transform(X[FEATURES].astype(float))
        return pd.DataFrame(arr, columns=FEATURES, index=X.index)

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "RiskModel":
        self.imputer = SimpleImputer(strategy="median").fit(X[FEATURES].astype(float))
        self.estimator = self._make().fit(self.transform(X), np.asarray(y),
                                          sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.transform(X))[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RiskModel":
        return joblib.load(path)
