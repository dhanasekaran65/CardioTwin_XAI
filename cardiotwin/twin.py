"""Patient digital twin: state assimilation, personalised progression, what-if simulation.

The twin is a stochastic state-space model:
    x_{t+1} = x_t + drift_i + k * (setpoint - x_t)  [if intervention] + noise
    risk_t  = f_model(x_t)
It self-updates in two ways:
  1. `observe()` assimilates new measurements (Kalman-style gain `alpha`) and
     re-estimates patient-specific progression rates from their history (shrinkage
     towards population priors).
  2. The risk model `f_model` itself is refreshed by `self_learning.py`.

NOTE: what-if results are model-based *associations*, not causal treatment effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BOUNDS, FEATURES, NUMERIC
from .model import RiskModel

DEFAULT_RATES = {"trestbps": 0.6, "chol": 0.4, "thalach": -0.7, "oldpeak": 0.03}  # per year
RATE_LIMITS = {"trestbps": 5.0, "chol": 15.0, "thalach": 5.0, "oldpeak": 0.5}
NOISE_SD = {"trestbps": 3.0, "chol": 8.0, "thalach": 3.0, "oldpeak": 0.10}
RELAX = 0.4       # yearly fraction of the gap to the intervention set-point that closes
SHRINK_K = 3.0    # prior strength (pseudo-observations) for personalised rates


@dataclass(frozen=True)
class Intervention:
    name: str
    deltas: dict = field(default_factory=dict)   # feature -> target change at full adherence
    adherence: float = 0.8
    progression_factor: float = 1.0              # multiplies yearly vessel-progression prob.


SCENARIOS = {
    "No change": Intervention("No change"),
    "BP control": Intervention("BP control", {"trestbps": -15}),
    "Lipid control": Intervention("Lipid control", {"chol": -40}, progression_factor=0.7),
    "Exercise program": Intervention("Exercise program",
                                     {"thalach": 12, "oldpeak": -0.3, "trestbps": -5}),
    "Combined": Intervention("Combined", {"trestbps": -15, "chol": -40, "thalach": 12,
                                          "oldpeak": -0.3}, progression_factor=0.6),
}


class PatientTwin:
    def __init__(self, patient_id: str, state: dict | pd.Series, model: RiskModel,
                 alpha: float = 0.6):
        self.patient_id = patient_id
        self.model = model
        self.alpha = alpha
        self.state = {f: float(state[f]) for f in FEATURES}
        self.t = 0.0
        self.rates = dict(DEFAULT_RATES)
        self.obs_log = {f: [(0.0, self.state[f])] for f in DEFAULT_RATES}

    # ---- self-updating ---------------------------------------------------------------
    def observe(self, t: float, obs: dict) -> dict:
        """Assimilate a new measurement at time `t` (years). Returns innovations."""
        dt = t - self.t
        innov = {}
        for f, v in obs.items():
            if f not in FEATURES:
                continue
            if f in NUMERIC:
                pred = self.state[f] + (self.rates.get(f, 0.0) * dt if f != "age" else dt)
                innov[f] = float(v - pred)
                self.state[f] = float(pred + self.alpha * (v - pred))
            else:
                self.state[f] = float(v)
            if f in self.obs_log:
                self.obs_log[f].append((t, float(v)))
        self.t = t
        self._update_rates()
        return innov

    def _update_rates(self) -> None:
        for f, pts in self.obs_log.items():
            n = len(pts)
            if n >= 3:
                tt, vv = zip(*pts)
                slope = float(np.polyfit(tt, vv, 1)[0])
                w = n / (n + SHRINK_K)
                r = w * slope + (1 - w) * DEFAULT_RATES[f]
                self.rates[f] = float(np.clip(r, -RATE_LIMITS[f], RATE_LIMITS[f]))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.state])[FEATURES]

    def current_risk(self) -> float:
        return float(self.model.predict_proba(self.to_frame())[0])

    # ---- simulation ------------------------------------------------------------------
    def _rollout(self, iv: Intervention, horizon: int, n: int, rng, noise: bool):
        x = pd.concat([self.to_frame()] * n, ignore_index=True)
        x0 = x.copy()
        states, risks = [x.copy()], [self.model.predict_proba(x)]
        for _ in range(horizon):
            x["age"] += 1
            for f, r in self.rates.items():
                x[f] += r
                if f in iv.deltas:
                    x[f] += RELAX * (x0[f] + iv.deltas[f] * iv.adherence - x[f])
                if noise:
                    x[f] += rng.normal(0, NOISE_SD[f], n)
            # stochastic progression of obstructive vessel count, driven by current risk
            p_prog = (0.02 + 0.06 * risks[-1]) * iv.progression_factor
            x["ca"] += (rng.random(n) < p_prog).astype(float) if noise else p_prog
            for f, (lo, hi) in BOUNDS.items():
                x[f] = x[f].clip(lo, hi)
            states.append(x.copy())
            risks.append(self.model.predict_proba(x))
        return states, np.vstack(risks)  # risks: (horizon+1, n)

    def trajectory(self, iv: Intervention | None = None, horizon: int = 10,
                   n_mc: int = 200, seed: int = 0) -> pd.DataFrame:
        """Monte-Carlo risk trajectory with uncertainty band."""
        iv = iv or SCENARIOS["No change"]
        _, risks = self._rollout(iv, horizon, n_mc, np.random.default_rng(seed), noise=True)
        return pd.DataFrame({"year": np.arange(horizon + 1), "risk_mean": risks.mean(1),
                             "risk_p10": np.percentile(risks, 10, axis=1),
                             "risk_p90": np.percentile(risks, 90, axis=1)})

    def expected_state(self, iv: Intervention | None = None, horizon: int = 10) -> pd.DataFrame:
        """Deterministic (noise-free) end-of-horizon state; used for SHAP attribution."""
        iv = iv or SCENARIOS["No change"]
        states, _ = self._rollout(iv, horizon, 1, np.random.default_rng(0), noise=False)
        return states[-1]

    def compare_scenarios(self, horizon: int = 10, n_mc: int = 200) -> dict:
        return {name: self.trajectory(iv, horizon, n_mc) for name, iv in SCENARIOS.items()}
