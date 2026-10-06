# CardioTwin-XAI
**A Self-Learning Digital Twin Framework for Explainable Cardiovascular Risk Prognosis Using SHAP-Based Feature Attribution**

CardioTwin-XAI combines four components in one reproducible Python package:

| Layer | Module | What it does |
|---|---|---|
| Risk model | `model.py` | Imputer + XGBoost (clinical **monotonic constraints**) / GradientBoosting / RandomForest |
| Explainability | `explain.py` | SHAP local (waterfall, narrative), global (beeswarm, bar), **deletion faithfulness**, **bootstrap stability** |
| Digital twin | `twin.py` | Per-patient stochastic state-space model, Kalman-style assimilation of new visits, **personalised progression rates**, Monte-Carlo what-if scenarios with uncertainty bands |
| Self-learning | `self_learning.py` | PSI covariate-drift + Page-Hinkley error-drift monitors → challenger retraining on a recency-weighted replay buffer → **champion/challenger promotion**, model registry, **SHAP attribution-shift logging** |

```
 new patient data ──► Twin.observe() ─► updated state ─► RiskModel ─► risk + SHAP explanation
        │                                                   ▲
        └─► SelfLearningManager: score → drift check → retrain challenger → promote if better
```

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py              # synthetic cohort, full research pipeline (~1 min)
python run_pipeline.py --fast       # skip bootstrap stability
python run_pipeline.py --data heart.csv   # real UCI / Kaggle heart-disease CSV
streamlit run app.py                # Doctor + Patient web application
pytest -q                           # unit tests
```
Outputs land in `outputs/` (`figures/*.png`, `report.json`, `global_importance.csv`,
`artifacts/self_learning_log.csv`, `artifacts/model_registry.json`, saved models).

## The web application (`app.py`)
A role-based Streamlit app sits on top of the research pipeline, so the framework can be
used the way it would be in practice — by a clinician and by their patient, in one login.

**Login.** One login screen for both roles (`store.py` is a small JSON-backed account
store — salted/hashed passwords, no external DB needed). Patients can self-register and
pick their doctor; doctor accounts are onboarded by a clinic admin (or use the demo
accounts below).

| Role | Sees | Can do |
|---|---|---|
| **Doctor** | Full patient roster with risk bands, full 13-feature clinical panel, SHAP tables + waterfall, model-governance panel (drift log, champion version, population feature importance) | Update any patient's vitals, onboard new patients, download a clinical PDF, edit own profile |
| **Patient** | Their own risk only, in plain language, with clinician-only fields (ECG/angiogram-derived: `cp`, `restecg`, `oldpeak`, `slope`, `ca`, `thal`) shown read-only | Log home/wearable readings (BP, HR, cholesterol, glucose flag, exertional angina), run a friendly what-if simulator, download a personal PDF, edit own profile |

**Digital-twin heart visual.** `heart_visual.py` renders an animated SVG heart + ECG
trace whose color, beat rate and jaggedness change with the risk band (`config.RISK_BANDS`:
Low <33%, Moderate 33–66%, High ≥66%) — a glance replaces a number.

**Real-time feel.** Every widget change re-runs the model immediately (Streamlit's normal
reactivity), so a doctor or patient sees the predicted risk update live as they adjust a
value, before choosing to save it. A "Simulate wearable sync" button injects a small
random fluctuation as a stand-in for a connected device feed. Saved readings are written
to the shared JSON store immediately, so a patient's self-logged reading shows up on their
doctor's dashboard on the doctor's next rerun — no page reload needed beyond Streamlit's own.

**Reports.** `reports.py` builds two different PDFs with ReportLab: a clinical report
(vitals, SHAP table, waterfall chart, scenario projection, risk trend) for the doctor, and
a plain-language summary (risk band, narrative, generic lifestyle tips, trend chart) for
the patient — both downloadable with one click.

**Demo accounts** (seeded automatically on first run):
| Role | Username | Password |
|---|---|---|
| Doctor | `dr.asha` | `doctor123` |
| Patient | `patient1` | `patient123` |
| Patient | `patient2` | `patient123` |

The app reuses whatever champion model `run_pipeline.py` last trained
(`outputs/artifacts/champion_latest.joblib`); run the pipeline at least once so the
Model Monitor tab and the risk predictions reflect a real trained model rather than a
quick fallback fit.

## Using real data
`data.load_csv()` accepts the UCI Cleveland schema (`age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal, target`) and common Kaggle variants (`num`, `condition`,
`thalch`, string categories). Target is binarised (`>0` = disease). Note that category encodings
differ between Kaggle versions (e.g. `cp` coding); check them before interpreting SHAP signs.
The synthetic generator uses `cp=0` = asymptomatic (highest risk) and `thal=3` = reversible defect.

## Design notes
* **Prequential evaluation**: each stream batch is scored *before* the model learns from it.
* **Safe updates**: a challenger replaces the champion only if it lowers log-loss on recent held-out
  data by `min_gain`. Labels are assumed to arrive with each batch (delayed-label handling is future work).
* **XAI quality is measured, not assumed**: deletion test (SHAP-ranked vs random feature removal)
  and bootstrap top-k Jaccard / Spearman stability are reported in `report.json`.
* **Monotonic constraints** stop the model producing clinically implausible what-if results
  (e.g. lower blood pressure raising risk).

## Limitations
Research prototype, **not a medical device**. SHAP explains the *model*, not causal mechanisms;
scenario results are associations. Twin dynamics use literature-style priors (`twin.py`) that should
be calibrated on longitudinal cohorts (e.g. Framingham, UK Biobank) before any real-world claim.
Synthetic-data results demonstrate the framework only.

## Extending
Swap in survival/time-to-event models, add fairness audits by sex/age subgroup, use
`shap.Explainer` for non-tree models, replace the replay buffer with reservoir sampling, or
connect `observe()` to an EHR/wearable stream.
