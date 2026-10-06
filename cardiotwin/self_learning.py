"""Self-learning loop: drift detection -> challenger retraining -> champion promotion.

Protocol (prequential / test-then-train, labels assumed to arrive with the batch):
  1. Score the incoming batch with the current champion (logged before any update).
  2. Monitor covariate drift (PSI) and error drift (Page-Hinkley).
  3. Add batch to a recency-weighted replay buffer (part held out for validation).
  4. If a trigger fires, train a challenger and promote it only if it beats the
     champion on recent held-out data (guards against harmful updates).
  5. On promotion, log how SHAP global attributions shifted (explanation drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, SEED
from .evaluate import evaluate
from .explain import ShapExplainer
from .model import RiskModel

_EPS = 1e-4


class PSIDriftDetector:
    """Population Stability Index per feature vs a reference window."""

    def __init__(self, reference: pd.DataFrame, bins: int = 10, threshold: float = 0.2):
        self.bins, self.threshold = bins, threshold
        self.set_reference(reference)

    def set_reference(self, ref: pd.DataFrame) -> None:
        self.spec, self.ref_dist = {}, {}
        for f in FEATURES:
            x = ref[f].dropna().to_numpy(float)
            if len(np.unique(x)) <= 8:
                cats = np.unique(x)
                self.spec[f] = ("cat", cats)
            else:
                edges = np.unique(np.quantile(x, np.linspace(0, 1, self.bins + 1)))[1:-1]
                self.spec[f] = ("num", edges)
            self.ref_dist[f] = self._dist(f, x)

    def _dist(self, f: str, x: np.ndarray) -> np.ndarray:
        kind, ref = self.spec[f]
        if kind == "cat":
            d = np.array([(x == c).sum() for c in ref], float)
        else:
            d = np.bincount(np.searchsorted(ref, x, side="right"), minlength=len(ref) + 1).astype(float)
        return np.clip(d / max(d.sum(), 1), _EPS, None)

    def psi(self, batch: pd.DataFrame) -> dict:
        out = {}
        for f in FEATURES:
            a, e = self._dist(f, batch[f].dropna().to_numpy(float)), self.ref_dist[f]
            out[f] = float(np.sum((a - e) * np.log(a / e)))
        return out


class PageHinkley:
    """Page-Hinkley test for an upward shift in the mean of a stream (e.g. |y-p|)."""

    def __init__(self, delta: float = 0.005, threshold: float = 15.0, min_n: int = 100):
        self.delta, self.threshold, self.min_n = delta, threshold, min_n
        self.reset()

    def reset(self) -> None:
        self.n, self.mean, self.m, self.m_min = 0, 0.0, 0.0, 0.0

    def update_many(self, xs) -> bool:
        alarm = False
        for x in xs:
            self.n += 1
            self.mean += (x - self.mean) / self.n
            self.m += x - self.mean - self.delta
            self.m_min = min(self.m_min, self.m)
            if self.n >= self.min_n and self.m - self.m_min > self.threshold:
                alarm = True
        return alarm


class SelfLearningManager:
    def __init__(self, champion: RiskModel, X_ref: pd.DataFrame, y_ref, *,
                 replay_size: int = 4000, holdout_frac: float = 0.3, decay: float = 0.7,
                 min_gain: float = 0.002, periodic_every: int = 0,
                 psi_threshold: float = 0.2, ph_threshold: float = 15.0, seed: int = SEED):
        self.champion, self.version = champion, 1
        self.replay_size, self.holdout_frac, self.decay = replay_size, holdout_frac, decay
        self.min_gain, self.periodic_every, self.seed = min_gain, periodic_every, seed
        self.psi_threshold = psi_threshold
        self.rng = np.random.default_rng(seed)
        self.X = X_ref.reset_index(drop=True).copy()
        self.y = pd.Series(np.asarray(y_ref)).reset_index(drop=True)
        self.b = np.full(len(self.X), -1)                  # batch id of each buffered row
        self.hX, self.hy, self.hb = self.X.iloc[0:0], self.y.iloc[0:0], np.array([], int)
        self.psi = PSIDriftDetector(self.X, threshold=psi_threshold)
        self.ph = PageHinkley(threshold=ph_threshold)
        self.batch_idx = 0
        self.log: list[dict] = []
        self.registry = [{"version": 1, "batch": -1, "reason": "initial"}]

    # ---------------------------------------------------------------------------------
    def process_batch(self, X: pd.DataFrame, y) -> dict:
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        i = self.batch_idx

        # 1. prequential evaluation with the current champion
        p = self.champion.predict_proba(X)
        m = evaluate(y, p, threshold=0.5)
        rec = {"batch": i, "n": len(X), "version": self.version, "auroc": m["auroc"],
               "logloss": m["logloss"], "brier": m["brier"], "ece": m["ece"]}

        # 2. drift monitoring
        psi = self.psi.psi(X)
        top = max(psi, key=psi.get)
        psi_alarm = psi[top] > self.psi_threshold
        ph_alarm = self.ph.update_many(np.abs(y.to_numpy() - p))
        periodic = bool(self.periodic_every) and (i + 1) % self.periodic_every == 0
        trigger = psi_alarm or ph_alarm or periodic
        rec.update(psi_max=psi[top], psi_feature=top, psi_alarm=bool(psi_alarm),
                   ph_alarm=bool(ph_alarm), trigger=bool(trigger))

        # 3. update replay buffer + rolling holdout (last two batches)
        perm = self.rng.permutation(len(X))
        n_h = int(len(X) * self.holdout_frac)
        h, t = perm[:n_h], perm[n_h:]
        self.X = pd.concat([self.X, X.iloc[t]], ignore_index=True).tail(self.replay_size)
        self.y = pd.concat([self.y, y.iloc[t]], ignore_index=True).tail(self.replay_size)
        self.b = np.concatenate([self.b, np.full(len(t), i)])[-self.replay_size:]
        self.X = self.X.reset_index(drop=True); self.y = self.y.reset_index(drop=True)
        self.hX = pd.concat([self.hX, X.iloc[h]], ignore_index=True)
        self.hy = pd.concat([self.hy, y.iloc[h]], ignore_index=True)
        self.hb = np.concatenate([self.hb, np.full(len(h), i)])
        keep = self.hb >= i - 1
        self.hX, self.hy, self.hb = self.hX[keep].reset_index(drop=True), self.hy[keep].reset_index(drop=True), self.hb[keep]

        # 4. challenger -> champion decision
        promoted = False
        if trigger and len(self.hX) >= 100:
            w = np.maximum(0.2, self.decay ** (i - self.b))
            challenger = RiskModel(self.champion.model_type, seed=self.seed + i).fit(self.X, self.y, sample_weight=w)
            mc = evaluate(self.hy, self.champion.predict_proba(self.hX), 0.5)
            mn = evaluate(self.hy, challenger.predict_proba(self.hX), 0.5)
            rec.update(hold_champion_logloss=mc["logloss"], hold_challenger_logloss=mn["logloss"])
            if mn["logloss"] < mc["logloss"] - self.min_gain:
                promoted = True
                rec.update(self._promote(challenger, i, mc, mn))
        if trigger:
            self.ph.reset()
        rec["promoted"] = promoted
        self.log.append(rec)
        self.batch_idx += 1
        return rec

    def _promote(self, challenger: RiskModel, i: int, mc: dict, mn: dict) -> dict:
        sample = pd.concat([self.hX, self.X.tail(300)], ignore_index=True)
        old_imp = ShapExplainer(self.champion).global_importance(sample)
        new_imp = ShapExplainer(challenger).global_importance(sample)
        delta = (new_imp - old_imp.reindex(new_imp.index)).sort_values(key=np.abs, ascending=False)
        self.champion, self.version = challenger, self.version + 1
        recent = self.X[self.b >= i - 1]
        self.psi.set_reference(recent if len(recent) >= 300 else self.X.tail(600))
        self.registry.append({"version": self.version, "batch": i, "reason": "drift",
                              "holdout_logloss_before": mc["logloss"],
                              "holdout_logloss_after": mn["logloss"],
                              "top_attribution_shift": delta.index[0],
                              "top_attribution_delta": float(delta.iloc[0])})
        return {"new_version": self.version, "attr_shift_feature": delta.index[0],
                "attr_shift_delta": float(delta.iloc[0])}

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)
