"""Patient-facing dashboard: My Heart (risk card, animated twin, key vitals, SHAP breakdown,
history, care team), Update Health Data (with live model-estimated risk preview),
Digital Twin (10-year projection + personalized guidance), My Report, and Profile management.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import heart_visual
from .config import PATIENT_EDITABLE, risk_category
from .reports import build_patient_report
from .store import store
from .twin import SCENARIOS
from .ui_common import (build_twin, format_last_assessment, inject_custom_css, live_clock,
                        load_model_bundle, patient_health_data_form, predict_and_explain)
from .viz import plot_trajectories

_TIP_MAP = {
    "trestbps": "Your blood pressure is contributing to your risk — ask your doctor about diet, salt intake, activity and whether medication is right for you.",
    "chol": "Your cholesterol is contributing to your risk — a review of your diet or a discussion about cholesterol-lowering treatment may help.",
    "thalach": "Your heart-rate fitness looks like an area to work on — gradual, doctor-approved exercise can improve this over time.",
    "fbs": "Your fasting blood sugar is flagged — managing this with diet, activity and follow-up testing can reduce cardiovascular risk.",
    "exang": "You've reported chest discomfort during exercise — please have this evaluated before starting or changing an exercise routine.",
    "oldpeak": "Your ECG stress-test results are contributing to your risk — your doctor is best placed to interpret this finding with you.",
}

_SCENARIO_FRIENDLY = {
    "No change": "Current trajectory",
    "BP control": "Blood pressure control",
    "Lipid control": "Lipid control",
    "Exercise program": "Exercise improvement",
    "Combined": "Combined"
}


def _risk_badge(cat: str) -> str:
    return {"Low": "🟢 Low", "Moderate": "🟠 Moderate", "High": "🔴 High"}.get(cat, cat)


def _tips(table) -> list[str]:
    ups = table[table.shap > 0].sort_values("shap", ascending=False)
    seen, out = set(), []
    for r in ups.itertuples():
        if r.feature in _TIP_MAP and r.feature not in seen:
            out.append(_TIP_MAP[r.feature])
            seen.add(r.feature)
        if len(out) == 3:
            break
    if not out:
        out.append("Monitor your blood pressure and heart rate regularly at home.")
        out.append("Maintain physical activity as advised by your healthcare provider.")
        out.append("Follow your prescribed heart-health and dietary plan.")
    return out


def render(user: dict) -> None:
    inject_custom_css()
    model, explainer = load_model_bundle()
    username = user["username"]
    record = store.get_patient_record(username)
    if record is None:
        st.error("No health record found yet. Please ask your doctor to complete your onboarding.")
        return

    profile = record["profile"]
    vitals = record["vitals"]
    first_name = profile.get("full_name", username).split()[0]

    st.title(f"❤️ Welcome, {first_name}")
    st.caption("CardioTwin Clinical Risk Monitoring & Digital Twin Simulation")

    # Clean 5-tab patient navigation
    tabs = st.tabs(["🫀 My Heart", "📊 Health Data", "🔮 Digital Twin", "📄 My Report", "⚙️ My Profile"])

    risk, cat, table, narrative, row = predict_and_explain(model, explainer, vitals)
    last_t = record["history"][-1]["t"] if record.get("history") else None
    last_assessment_str = format_last_assessment(last_t)

    # --------------------------------------------------------- 1. My Heart --------
    with tabs[0]:
        # Hero Section: Current Heart Risk & Digital Twin Heart
        c1, c2 = st.columns([1.2, 1.1])
        with c1:
            st.markdown("### Your Current Heart Risk")
            st.markdown(f"# {risk:.0%}")
            st.markdown(f"### {_risk_badge(cat)} RISK")
            st.write("Based on your latest recorded health information.")
            st.caption(f"📅 **Assessment updated:** {last_assessment_str}")

            # Expandable XAI explanation button
            with st.expander("🔍 Why is my risk this value?", expanded=False):
                st.markdown(f"#### Why is my model risk {risk:.0%}?")
                st.write("The risk model evaluated your recorded clinical measurements to estimate cardiovascular risk:")

                pos_factors = table[table.shap > 0].head(3)
                neg_factors = table[table.shap < 0].head(2)

                if not pos_factors.empty:
                    st.markdown("🔴 **Factors contributing to higher model risk:**")
                    for _, r in pos_factors.iterrows():
                        st.markdown(f"- **{r['label']}** ({r['value']:g}) `↑`")

                if not neg_factors.empty:
                    st.markdown("🟢 **Factors contributing to lower model risk:**")
                    for _, r in neg_factors.iterrows():
                        st.markdown(f"- **{r['label']}** ({r['value']:g}) `↓`")

                with st.expander("View Detailed Feature Attribution Plot", expanded=False):
                    fig = explainer.plot_waterfall(row)
                    st.pyplot(fig)

        with c2:
            st.markdown("##### 🫀 Digital Twin Simulated State")
            heart_html = heart_visual.render(risk, subtitle=f"Simulated resting state for {first_name}")
            st.components.v1.html(heart_html, height=310)

        st.divider()

        # Key Health Measurements (Patient-understandable only)
        st.markdown("### 📊 Key Health Measurements")
        m1, m2, m3 = st.columns(3)
        m1.metric("Resting BP", f"{int(vitals.get('trestbps', 120))} mmHg")
        m2.metric("Cholesterol", f"{int(vitals.get('chol', 200))} mg/dL")
        m3.metric("Maximum Heart Rate", f"{int(vitals.get('thalach', 150))} bpm")

        m4, m5 = st.columns(2)
        fbs_status = "High (> 120 mg/dL)" if vitals.get("fbs", 0) else "Normal (≤ 120 mg/dL)"
        m4.metric("Fasting Blood Sugar", fbs_status)
        exang_status = "Yes (Reported)" if vitals.get("exang", 0) else "No (None)"
        m5.metric("Exercise Discomfort / Angina", exang_status)

        st.divider()

        # Risk History
        st.markdown("### 📈 Risk History Over Time")
        hist = [h for h in record.get("history", []) if h.get("risk") is not None]
        if len(hist) >= 2:
            chart_df = pd.DataFrame({
                "Risk (%)": [h["risk"] * 100 for h in hist]
            }, index=[format_last_assessment(h.get("t")) for h in hist])
            st.line_chart(chart_df)
        else:
            st.info("Single recorded assessment. Your trend chart will appear as you log further checkups.")

        st.divider()

        # Care Team Card
        doc = store.get_user(record.get("doctor")) if record.get("doctor") else None
        if doc:
            doc_p = doc["profile"]
            st.markdown("### 👨‍⚕️ Your Care Team")
            st.markdown(f"**{doc_p.get('full_name', 'Dr. Asha')}** · {doc_p.get('specialty', 'Cardiologist')}")
            st.caption(f"🏥 {doc_p.get('hospital', 'Clinic')} · 📞 {doc_p.get('phone', '—')} · ✉️ {doc_p.get('email', '—')}")

    # ------------------------------------------------------ 2. Health Data --------
    with tabs[1]:
        st.subheader("📊 Health Data")
        st.caption("Update your latest home or lab measurements. Your model-estimated risk updates in real time below.")

        col_form, col_preview = st.columns([1.2, 1])

        with col_form:
            edited = patient_health_data_form(vitals, key_prefix="pat_data")
            live_risk, live_cat, live_table, _, live_row = predict_and_explain(model, explainer, edited)

            st.markdown("---")
            st.markdown("### Updated Risk Preview")
            r_col1, r_col2 = st.columns(2)
            r_col1.metric("Current model-estimated risk", f"{risk:.0%}", help=f"Current band: {cat}")
            r_col2.metric("Updated model-estimated risk", f"{live_risk:.0%}", delta=f"{(live_risk - risk):+.0%}",
                          delta_color="inverse", help=f"Updated band: {live_cat}")

            if st.button("💾 Save Health Data", type="primary"):
                store.record_observation(username, edited, source="patient", risk=live_risk)
                st.success("Health data saved! Your doctor can view these updated measurements.")
                st.rerun()

        with col_preview:
            st.markdown("##### 🫀 Live Digital Twin Response")
            st.caption("Digital Twin state adapts instantly to changes in your physiological parameters.")
            preview_html = heart_visual.render(live_risk, subtitle="Simulated state based on updated inputs")
            st.components.v1.html(preview_html, height=310)

    # ----------------------------------------------------- 3. Digital Twin --------
    with tabs[2]:
        st.subheader("🔮 Digital Twin — 10-Year Risk Projection")
        st.caption("Explore how various evidence-based lifestyle interventions can influence your long-term cardiovascular trajectory.")

        twin = build_twin(username, record, model)
        horizon = st.slider("Projection Horizon (years)", 3, 15, 10, key="pat_hz")

        scenario_options = list(_SCENARIO_FRIENDLY.values())
        chosen_labels = st.multiselect(
            "Choose scenarios to compare:",
            options=scenario_options,
            default=scenario_options,
            key="pat_scenarios"
        )

        inv_map = {v: k for k, v in _SCENARIO_FRIENDLY.items()}
        trajs = {}
        for lbl in chosen_labels:
            sc_key = inv_map[lbl]
            trajs[lbl] = twin.trajectory(SCENARIOS[sc_key], horizon, n_mc=150)

        if trajs:
            fig = plot_trajectories(trajs, "Your Projected Risk", save=False)
            st.pyplot(fig)

        st.divider()

        # Current State → Simulated State Comparison
        st.markdown("### 🔀 Current State → Simulated State Comparison")
        st.write("Select an intervention scenario to compare its projected impact against your current baseline:")

        selected_sc_label = st.selectbox("Select Scenario to Inspect:", scenario_options, index=1 if len(scenario_options) > 1 else 0)
        selected_sc_key = inv_map[selected_sc_label]
        sc_obj = SCENARIOS[selected_sc_key]

        # Calculate scenario-adjusted vitals and projected risk
        sim_vitals = dict(vitals)
        for feat, delta_val in sc_obj.deltas.items():
            if feat in sim_vitals:
                sim_vitals[feat] = max(0, sim_vitals[feat] + delta_val)
        sim_risk, sim_cat, sim_table, _, sim_row = predict_and_explain(model, explainer, sim_vitals)

        cmp_c1, cmp_c2 = st.columns(2)
        with cmp_c1:
            st.markdown("#### Current Patient State")
            st.metric("Current Model Risk", f"{risk:.0%}")
            st.markdown(f"**State:** {_risk_badge(cat)} RISK")
            curr_heart = heart_visual.render(risk, subtitle="Current baseline state")
            st.components.v1.html(curr_heart, height=300)

        with cmp_c2:
            st.markdown(f"#### Digital Twin Scenario ({selected_sc_label})")
            st.metric("Projected Model Risk", f"{sim_risk:.0%}", delta=f"{(sim_risk - risk):+.0%}", delta_color="inverse")
            st.markdown(f"**Projected State:** {_risk_badge(sim_cat)} RISK")
            sim_heart = heart_visual.render(sim_risk, subtitle=f"Simulated with {selected_sc_label}")
            st.components.v1.html(sim_heart, height=300)

        # Why Did the Twin Change?
        st.markdown("### 🔍 Why did the Twin change?")
        if selected_sc_key == "No change" or abs(sim_risk - risk) < 0.005:
            st.info("No intervention deltas applied in current trajectory. The digital twin continues on your current baseline trend.")
        else:
            delta_pts = (sim_risk - risk) * 100
            st.write(f"Model-estimated risk changed from **{risk:.0%} → {sim_risk:.0%}** ({delta_pts:+.1f}% points).")
            st.markdown("**Main contributing parameter changes:**")
            for feat, delta_val in sc_obj.deltas.items():
                orig_val = vitals.get(feat, 0)
                new_val = sim_vitals.get(feat, 0)
                lbl = feat.upper()
                if feat == "trestbps": lbl = "Resting BP"
                elif feat == "chol": lbl = "Cholesterol"
                elif feat == "thalach": lbl = "Max Heart Rate"
                elif feat == "oldpeak": lbl = "ST Depression"
                st.markdown(f"- **{lbl}:** `{orig_val:g}` → `{new_val:g}` ({delta_val:+g})")
            st.caption("ℹ️ *These physiological adjustments altered the model-estimated cardiovascular risk profile.*")

        st.divider()

        # Personalized Guidance
        st.markdown("### 💡 Personalized Guidance")
        st.markdown("**Based on your current risk factors:**")
        for tip in _tips(table):
            st.markdown(f"• {tip}")

    # -------------------------------------------------------- 4. My Report --------
    with tabs[3]:
        st.subheader("📄 My Health Report")
        st.caption("Download a personalized summary of your cardiovascular risk profile to keep for your records or share with your care team.")

        clean_narrative = f"Estimated cardiovascular risk: {risk:.0%}. Primary risk contributors evaluated by model: " + \
                          ", ".join(f"{r.label}" for r in table[table.shap > 0].head(3).itertuples())

        pdf_bytes = build_patient_report(
            patient_name=profile.get("full_name", username),
            patient_username=username,
            vitals=record["vitals"],
            risk=risk,
            narrative=clean_narrative,
            tips=_tips(table),
            history=record.get("history", [])
        )
        st.download_button("⬇️ Download My Health Report (PDF)", data=pdf_bytes,
                           file_name=f"{username}_health_report.pdf", mime="application/pdf",
                           type="primary")

    # ------------------------------------------------------- 5. My Profile --------
    with tabs[4]:
        _profile_editor(user, profile)

    st.markdown("---")
    st.caption("CardioTwin-XAI is a clinical research decision-support prototype, not a diagnostic device. Please follow your healthcare professional's advice.")


def _profile_editor(user: dict, profile: dict) -> None:
    st.subheader("My Profile")
    with st.form("patient_profile"):
        full_name = st.text_input("Full name", profile.get("full_name", ""))
        age = st.number_input("Age", 20, 95, int(profile.get("age", 50)))
        sex = st.selectbox("Sex", ["Male", "Female"],
                           index=0 if profile.get("sex", "Male") == "Male" else 1)
        phone = st.text_input("Phone", profile.get("phone", ""))
        email = st.text_input("Email", profile.get("email", ""))
        saved = st.form_submit_button("Save Changes", type="primary")
    if saved:
        store.update_profile(user["username"], {"full_name": full_name, "age": int(age),
                                                 "sex": sex, "phone": phone, "email": email})
        st.success("Profile updated.")
        st.session_state.auth_user = store.get_user(user["username"])
        st.rerun()

    st.divider()
    with st.form("patient_password"):
        st.markdown("**Change Password**")
        pw1 = st.text_input("New password", type="password")
        pw2 = st.text_input("Confirm new password", type="password")
        changed = st.form_submit_button("Update Password")
    if changed:
        if not pw1 or pw1 != pw2:
            st.error("Passwords must match and not be empty.")
        else:
            store.change_password(user["username"], pw1)
            st.success("Password updated.")

