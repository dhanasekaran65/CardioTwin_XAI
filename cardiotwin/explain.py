"""SHAP-based feature attribution: local, global, narrative, faithfulness and stability."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import FEATURES, LABELS
from .model import RiskModel


def _positive_class_values(sv) -> np.ndarray:
    """Normalise SHAP output shapes across shap/sklearn versions to (n, features)."""
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    return sv[:, :, -1] if sv.ndim == 3 else sv


class ShapExplainer:
    def __init__(self, model: RiskModel):
        self.model = model
        self.explainer = shap.TreeExplainer(model.estimator)
        self.units = "probability" if model.model_type == "rf" else "log-odds"

    # ---- core -----------------------------------------------------------------------
    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        Xi = self.model.transform(X)
        values = _positive_class_values(self.explainer.shap_values(Xi))
        base = float(np.ravel(self.explainer.expected_value)[-1])
        return shap.Explanation(values=values, base_values=np.full(len(Xi), base),
                                data=Xi.values, feature_names=[LABELS[f] for f in FEATURES])

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        vals = self.explain(X).values
        s = pd.Series(np.abs(vals).mean(0), index=FEATURES, name="mean_abs_shap")
        return s.sort_values(ascending=False)

    def local_table(self, X_row: pd.DataFrame) -> pd.DataFrame:
        e = self.explain(X_row.iloc[[0]])
        df = pd.DataFrame({"feature": FEATURES, "label": [LABELS[f] for f in FEATURES],
                           "value": e.data[0], "shap": e.values[0]})
        return df.reindex(df.shap.abs().sort_values(ascending=False).index).reset_index(drop=True)

    def narrative(self, X_row: pd.DataFrame, top: int = 3) -> str:
        risk = float(self.model.predict_proba(X_row.iloc[[0]])[0])
        t = self.local_table(X_row)
        up, down = t[t.shap > 0].head(top), t[t.shap < 0].head(top)
        fmt = lambda d: ", ".join(f"{r.label} = {r.value:g}" for r in d.itertuples()) or "none"
        return (f"Estimated cardiovascular risk: {risk:.0%}. "
                f"Main factors raising risk: {fmt(up)}. "
                f"Main factors lowering risk: {fmt(down)}. "
                f"(SHAP units: {self.units}; explains the model, not causal effects.)")

    # ---- plots ----------------------------------------------------------------------
    def plot_summary(self, X: pd.DataFrame, path) -> None:
        plt.figure()
        shap.plots.beeswarm(self.explain(X), show=False, max_display=13)
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    def plot_bar(self, X: pd.DataFrame, path) -> None:
        plt.figure()
        shap.plots.bar(self.explain(X), show=False, max_display=13)
        plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    def plot_waterfall(self, X_row: pd.DataFrame, path=None):
        fig = plt.figure()
        shap.plots.waterfall(self.explain(X_row.iloc[[0]])[0], show=False, max_display=10)
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        return fig

    # ---- XAI quality ----------------------------------------------------------------
    def deletion_faithfulness(self, X: pd.DataFrame, k_max: int = 6, seed: int = 0) -> pd.DataFrame:
        """Neutralise top-k SHAP features (set to median) vs random-k; a faithful
        explanation changes predictions more than random deletion."""
        Xi = self.model.transform(X)
        vals = self.explain(X).values
        est = self.model.estimator
        p0 = est.predict_proba(Xi)[:, 1]
        med = Xi.median().values
        n, f = Xi.shape
        rng = np.random.default_rng(seed)
        orders = {"SHAP": np.argsort(-np.abs(vals), axis=1),
                  "Random": np.array([rng.permutation(f) for _ in range(n)])}
        rows = []
        for name, order in orders.items():
            for k in range(k_max + 1):
                M = Xi.values.copy()
                for j in range(k):
                    M[np.arange(n), order[:, j]] = med[order[:, j]]
                p = est.predict_proba(pd.DataFrame(M, columns=Xi.columns))[:, 1]
                rows.append({"method": name, "k": k, "mean_abs_change": float(np.abs(p - p0).mean())})
        return pd.DataFrame(rows)


def explanation_stability(X_train: pd.DataFrame, y_train, X_eval: pd.DataFrame,
                          n_boot: int = 5, k: int = 5, seed: int = 0) -> dict:
    """Refit on bootstrap resamples; report top-k Jaccard overlap and rank correlation."""
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    imps = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X_train), len(X_train))
        m = RiskModel("gb", {"n_estimators": 120}, seed=seed + b).fit(X_train.iloc[idx], y_train[idx])
        imps.append(ShapExplainer(m).global_importance(X_eval.head(300)).reindex(FEATURES))
    tops = [set(s.nlargest(k).index) for s in imps]
    jac = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    rho = pd.concat(imps, axis=1).corr(method="spearman").values
    rho = rho[np.triu_indices(n_boot, 1)]
    return {"top_k": k, "mean_jaccard": float(np.mean(jac)), "mean_spearman": float(np.mean(rho))}
