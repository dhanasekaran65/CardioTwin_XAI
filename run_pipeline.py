"""End-to-end CardioTwin-XAI pipeline.

    python run_pipeline.py                      # synthetic cohort
    python run_pipeline.py --data heart.csv     # real UCI/Kaggle heart-disease CSV
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cardiotwin import data as D
from cardiotwin.config import ARTIFACT_DIR, FEATURES, FIG_DIR, MODIFIABLE, OUTPUT_DIR, SEED, TARGET, ensure_dirs
from cardiotwin.evaluate import evaluate
from cardiotwin.explain import ShapExplainer, explanation_stability
from cardiotwin.model import RiskModel
from cardiotwin.self_learning import SelfLearningManager
from cardiotwin.twin import SCENARIOS, PatientTwin
from cardiotwin import viz


def main(args):
    ensure_dirs()
    report = {}

    # 1 ── data + baseline model ────────────────────────────────────────────────────
    df = D.load_dataset(args.data, args.n)
    train, val, test = D.split(df)
    print(f"[data] train={len(train)} val={len(val)} test={len(test)} prevalence={df[TARGET].mean():.2f}")
    model = RiskModel(args.model).fit(train[FEATURES], train[TARGET])
    thr = evaluate(val[TARGET], model.predict_proba(val[FEATURES]))["threshold"]
    p_test = model.predict_proba(test[FEATURES])
    report["test_metrics"] = evaluate(test[TARGET], p_test, threshold=thr)
    print("[model]", {k: round(v, 3) for k, v in report["test_metrics"].items()})
    viz.plot_roc_calibration(test[TARGET], p_test)
    model.save(ARTIFACT_DIR / "champion.joblib")

    # 2 ── explainability ──────────────────────────────────────────────────────────
    ex = ShapExplainer(model)
    ex.plot_summary(test[FEATURES], FIG_DIR / "shap_summary.png")
    ex.plot_bar(test[FEATURES], FIG_DIR / "shap_bar.png")
    imp = ex.global_importance(test[FEATURES])
    imp.to_csv(OUTPUT_DIR / "global_importance.csv")
    report["top_features"] = imp.head(5).round(4).to_dict()
    curve = ex.deletion_faithfulness(test[FEATURES].head(300))
    viz.plot_deletion(curve)
    k3 = curve[curve.k == 3].set_index("method").mean_abs_change
    report["faithfulness_k3"] = {"SHAP": float(k3["SHAP"]), "Random": float(k3["Random"])}
    if not args.fast:
        report["stability"] = explanation_stability(train[FEATURES], train[TARGET], test[FEATURES])
    print("[xai] top features:", list(imp.head(5).index), "| faithfulness@3:", report["faithfulness_k3"])

    # 3 ── digital twin demo ────────────────────────────────────────────────────────
    cand = test[(test.age > 50)].assign(p=p_test[test.age > 50])
    row = cand.iloc[(cand.p - 0.5).abs().argsort().iloc[0]]
    twin = PatientTwin("patient-001", row[FEATURES], model)
    rng = np.random.default_rng(SEED)
    for t in (1, 2, 3):   # three follow-up visits with a worsening trend -> twin adapts
        twin.observe(t, {"trestbps": row.trestbps + 3 * t + rng.normal(0, 2),
                         "chol": row.chol + 5 * t + rng.normal(0, 4),
                         "thalach": row.thalach - 1.5 * t + rng.normal(0, 2), "age": row.age + t})
    print(f"[twin] personalised rates: { {k: round(v, 2) for k, v in twin.rates.items()} } | risk now {twin.current_risk():.2f}")
    trajs = twin.compare_scenarios(horizon=10)
    viz.plot_trajectories(trajs, f"Digital twin risk trajectories ({twin.patient_id})")
    ex.plot_waterfall(twin.to_frame(), FIG_DIR / "twin_waterfall.png")
    report["twin"] = {"rates": twin.rates, "risk_now": twin.current_risk(),
                      "risk_10y": {k: float(v.risk_mean.iloc[-1]) for k, v in trajs.items()},
                      "narrative": ex.narrative(twin.to_frame())}
    base_s = ex.explain(twin.expected_state(SCENARIOS["No change"])).values[0]
    comb_s = ex.explain(twin.expected_state(SCENARIOS["Combined"])).values[0]
    viz.plot_scenario_attribution(pd.Series(comb_s - base_s, index=FEATURES), "Combined intervention")
    print("[twin]", report["twin"]["narrative"])
    print("[twin] 10y risk:", {k: round(v, 3) for k, v in report["twin"]["risk_10y"].items()})

    # 4 ── self-learning under drift ────────────────────────────────────────────────
    frozen = RiskModel(args.model).fit(train[FEATURES], train[TARGET])
    mgr = SelfLearningManager(model, train[FEATURES], train[TARGET])
    schedule = [0.0, 0.0, 0.3, 0.7, 1.0, 1.3, 1.6, 1.6]
    rows = []
    for b, d in enumerate(schedule):
        batch = D.generate_synthetic(args.batch, seed=1000 + b, drift=d)
        rec = mgr.process_batch(batch[FEATURES], batch[TARGET])
        m = evaluate(batch[TARGET], frozen.predict_proba(batch[FEATURES]), 0.5)
        rows.append({"batch": b, "auroc": m["auroc"], "logloss": m["logloss"]})
        print(f"[stream] b{b} drift={d:.1f} v{rec['version']} logloss={rec['logloss']:.3f} "
              f"(frozen {m['logloss']:.3f}) psi={rec['psi_feature']}:{rec['psi_max']:.2f} "
              f"trigger={rec['trigger']} promoted={rec['promoted']}")
    log, frozen_df = mgr.log_frame(), pd.DataFrame(rows)
    log.to_csv(ARTIFACT_DIR / "self_learning_log.csv", index=False)
    (ARTIFACT_DIR / "model_registry.json").write_text(json.dumps(mgr.registry, indent=2))
    mgr.champion.save(ARTIFACT_DIR / "champion_latest.joblib")
    viz.plot_self_learning(log, frozen_df)
    report["self_learning"] = {"final_version": mgr.version,
                               "mean_logloss_adaptive": float(log.logloss.mean()),
                               "mean_logloss_frozen": float(frozen_df.logloss.mean()),
                               "mean_auroc_adaptive": float(log.auroc.mean()),
                               "mean_auroc_frozen": float(frozen_df.auroc.mean())}
    print("[self-learning]", report["self_learning"])

    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nDone. Figures -> {FIG_DIR}\nReport  -> {OUTPUT_DIR / 'report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to real heart-disease CSV")
    ap.add_argument("--n", type=int, default=3000, help="synthetic cohort size")
    ap.add_argument("--batch", type=int, default=500, help="stream batch size")
    ap.add_argument("--model", choices=["xgb", "gb", "rf"], default="xgb")
    ap.add_argument("--fast", action="store_true", help="skip stability bootstrap")
    main(ap.parse_args())
