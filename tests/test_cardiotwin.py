import numpy as np
import pytest

from cardiotwin import data as D
from cardiotwin.config import FEATURES, TARGET
from cardiotwin.evaluate import evaluate
from cardiotwin.explain import ShapExplainer
from cardiotwin.model import RiskModel
from cardiotwin.self_learning import PSIDriftDetector, SelfLearningManager
from cardiotwin.twin import SCENARIOS, PatientTwin


@pytest.fixture(scope="module")
def fitted():
    df = D.generate_synthetic(1500, seed=1)
    tr, va, te = D.split(df)
    return RiskModel("gb", {"n_estimators": 80}).fit(tr[FEATURES], tr[TARGET]), tr, te


def test_data_schema():
    df = D.generate_synthetic(200)
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert 0.2 < df[TARGET].mean() < 0.7


def test_model_beats_chance(fitted):
    m, _, te = fitted
    assert evaluate(te[TARGET], m.predict_proba(te[FEATURES]))["auroc"] > 0.7


@pytest.mark.parametrize("mt", ["gb", "xgb"])
def test_shap_additivity(mt):
    df = D.generate_synthetic(800, seed=2)
    m = RiskModel(mt, {"n_estimators": 60}).fit(df[FEATURES], df[TARGET])
    e = ShapExplainer(m).explain(df[FEATURES].head(50))
    recon = 1 / (1 + np.exp(-(e.base_values + e.values.sum(1))))
    assert np.allclose(recon, m.predict_proba(df[FEATURES].head(50)), atol=1e-3)


def test_shap_beats_random_deletion(fitted):
    m, _, te = fitted
    c = ShapExplainer(m).deletion_faithfulness(te[FEATURES].head(100), k_max=3)
    g = c[c.k == 3].set_index("method").mean_abs_change
    assert g["SHAP"] > g["Random"]


def test_twin_dynamics_and_assimilation(fitted):
    m, _, te = fitted
    twin = PatientTwin("p", te.iloc[0][FEATURES], m)
    trajs = twin.compare_scenarios(horizon=5, n_mc=50)
    assert all(len(t) == 6 for t in trajs.values())
    assert all(((t.risk_p10 <= t.risk_mean) & (t.risk_mean <= t.risk_p90)).all() for t in trajs.values())
    for t in (1, 2, 3):
        twin.observe(t, {"trestbps": twin.state["trestbps"] + 5})
    assert twin.rates["trestbps"] > 0.6  # learned a faster personal BP trend


def test_psi_detects_covariate_drift():
    ref = D.generate_synthetic(2000, seed=3)
    det = PSIDriftDetector(ref[FEATURES])
    assert max(det.psi(D.generate_synthetic(500, seed=4)[FEATURES]).values()) < 0.1
    assert max(det.psi(D.generate_synthetic(500, seed=5, drift=1.5)[FEATURES]).values()) > 0.2


def test_self_learning_runs_and_promotes(fitted):
    m, tr, _ = fitted
    mgr = SelfLearningManager(m, tr[FEATURES], tr[TARGET])
    for b, d in enumerate([0, 0, 1.0, 1.5, 1.5]):
        s = D.generate_synthetic(500, seed=50 + b, drift=d)
        mgr.process_batch(s[FEATURES], s[TARGET])
    log = mgr.log_frame()
    assert len(log) == 5 and log.trigger.any()
