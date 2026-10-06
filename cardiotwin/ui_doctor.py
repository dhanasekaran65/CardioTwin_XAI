"""Doctor-facing dashboard: Overview, My Patients (roster & clinical chart),
Add Patient onboarding, and Profile management.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import heart_visual
from .config import ARTIFACT_DIR, FEATURES, LABELS, risk_category
from .reports import build_doctor_report
from .store import store
from .twin import SCENARIOS
from .ui_common import (build_twin, format_last_assessment, grouped_doctor_vitals_form,
                        inject_custom_css, live_clock, load_model_bundle, model_version,
                        predict_and_explain)
from .viz import plot_trajectories


def _risk_badge(cat: str) -> str:
    return {"Low": "🟢 Low", "Moderate": "🟠 Moderate", "High": "🔴 High"}.get(cat, cat)


def render(user: dict) -> None:
    inject_custom_css()
    model, explainer = load_model_bundle()
    doctor_name = user["profile"].get("full_name", user["username"])
    st.title(f"🩺 Dr. {doctor_name.split()[-1] if ' ' in doctor_name else doctor_name} — Clinician Dashboard")
    live_clock("Connected to CardioTwin model service")

    # Clean 4-tab clinician navigation
    tabs = st.tabs(["🏠 Overview", "👥 My Patients", "➕ Add Patient", "⚙️ My Profile"])

    patients = store.list_patients_for_doctor(user["username"])
    patient_data_list = []
    for p in patients:
        row = pd.DataFrame([p["vitals"]])[FEATURES]
        risk = float(model.predict_proba(row)[0])
        cat = risk_category(risk)
        last_t = p["history"][-1]["t"] if p.get("history") else None
        last_str = format_last_assessment(last_t)
        patient_data_list.append({
            "username": p["username"],
            "name": p["profile"].get("full_name", p["username"]),
            "age": p["vitals"].get("age", 50),
            "sex": "Male" if p["vitals"].get("sex", 1) else "Female",
            "risk_val": risk,
            "risk_str": f"{risk:.0%}",
            "band": _risk_badge(cat),
            "cat": cat,
            "last_assessment": last_str,
            "raw": p
        })

    # ----------------------------------------------------------- 1. Overview ------
    with tabs[0]:
        total_pts = len(patient_data_list)
        high_pts = sum(1 for p in patient_data_list if p["cat"] == "High")
        mod_pts = sum(1 for p in patient_data_list if p["cat"] == "Moderate")
        low_pts = sum(1 for p in patient_data_list if p["cat"] == "Low")

        # Top summary metric cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients", total_pts)
        c2.metric("High Risk", high_pts, help="Patients with predicted risk >= 66%")
        c3.metric("Moderate Risk", mod_pts, help="Patients with predicted risk 33-66%")
        c4.metric("Low Risk", low_pts, help="Patients with predicted risk < 33%")

        st.divider()
        st.subheader("Recent Patient Updates")

        if not patient_data_list:
            st.info("No patients assigned yet. Use **Add Patient** to onboard your first patient.")
        else:
            overview_df = pd.DataFrame([
                {
                    "Patient": p["name"],
                    "Current Risk": p["risk_str"],
                    "Risk Band": p["band"],
                    "Last Assessment": p["last_assessment"]
                }
                for p in sorted(patient_data_list, key=lambda x: x["risk_val"], reverse=True)
            ])
            st.dataframe(overview_df, hide_index=True, width="stretch")

    # -------------------------------------------------------- 2. My Patients ------
    with tabs[1]:
        if not patient_data_list:
            st.info("No patients assigned yet. Use **Add Patient** to onboard your first patient.")
        else:
            roster_df = pd.DataFrame([
                {
                    "Patient": p["name"],
                    "Age": p["age"],
                    "Sex": p["sex"],
                    "Risk": p["risk_str"],
                    "Risk Band": p["band"],
                    "Last Assessment": p["last_assessment"],
                    "_username": p["username"]
                }
                for p in sorted(patient_data_list, key=lambda x: x["risk_val"], reverse=True)
            ])
            st.dataframe(roster_df.drop(columns="_username"), hide_index=True, width="stretch")

            names_map = {p["username"]: f"{p['name']} ({p['risk_str']} - {p['band']})" for p in patient_data_list}
            selected_username = st.selectbox(
                "Open patient record",
                list(names_map.keys()),
                format_func=lambda u: names_map[u],
                key="doc_selected_patient"
            )
            st.divider()
            _patient_detail(selected_username, model, explainer, user)

    # -------------------------------------------------------- 3. Add Patient ------
    with tabs[2]:
        st.subheader("Onboard a new patient")
        with st.form("add_patient"):
            c1, c2 = st.columns(2)
            full_name = c1.text_input("Full name")
            age = c1.number_input("Age", 20, 95, 55)
            sex = c1.selectbox("Sex", ["Male", "Female"])
            username = c2.text_input("Username (for patient login)")
            temp_pw = c2.text_input("Temporary password", value="cardio@123")
            phone = c2.text_input("Phone")
            email = c1.text_input("Email")
            submitted = st.form_submit_button("Create patient account", type="primary")
        if submitted:
            try:
                store.create_user(username, temp_pw, "patient",
                                  {"full_name": full_name, "age": int(age), "sex": sex,
                                   "phone": phone, "email": email}, doctor=user["username"])
                st.success(f"Patient **{full_name}** onboarded. Share login `{username.lower()}` / "
                          f"`{temp_pw}` with them.")
                st.cache_resource.clear()
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # ----------------------------------------------------------- 4. Profile -------
    with tabs[3]:
        _profile_editor(user)


def _patient_detail(username: str, model, explainer, user) -> None:
    record = store.get_patient_record(username)
    if not record:
        st.error("Patient record not found.")
        return

    profile = record["profile"]
    vitals = record["vitals"]
    risk, cat, table, narrative, row = predict_and_explain(model, explainer, vitals)
    last_t = record["history"][-1]["t"] if record.get("history") else None
    last_assessment_str = format_last_assessment(last_t)

    # 1. Patient Information
    st.markdown(f"### 📋 Patient Information — {profile.get('full_name', username)}")
    p1, p2, p3, p4 = st.columns(4)
    p1.write(f"**Patient ID:** `{username}`")
    p2.write(f"**Age / Sex:** {vitals.get('age', 50)} yrs · {'Male' if vitals.get('sex', 1) else 'Female'}")
    p3.write(f"**Phone:** {profile.get('phone', '—')}")
    p4.write(f"**Email:** {profile.get('email', '—')}")

    st.divider()

    # 2. Current Risk Section
    st.markdown("### 🎯 Current Risk")
    r1, r2, r3 = st.columns([1, 1, 1.5])
    r1.metric("Current Predicted Risk", f"{risk:.0%}")
    r2.markdown(f"#### {_risk_badge(cat)} Risk")
    r3.caption(f"📅 **Last assessed:** {last_assessment_str}")

    st.divider()

    # 3. Clinical Measurements & Update Form
    with st.expander("📝 Update Clinical Measurements & Findings", expanded=False):
        st.caption("Adjust clinical parameters from latest visit/lab results and update patient record.")
        with st.form(f"form_clinical_{username}"):
            edited = grouped_doctor_vitals_form(vitals, key_prefix=f"doc_{username}")
            saved = st.form_submit_button("Update Assessment", type="primary")
        if saved:
            new_risk, _, _, _, _ = predict_and_explain(model, explainer, edited)
            store.record_observation(username, edited, source="doctor", risk=new_risk)
            st.success("Assessment updated successfully.")
            st.rerun()

    # 4. Digital Twin Heart & 5. SHAP Explanation
    col_left, col_right = st.columns([1, 1.3])

    with col_left:
        st.markdown("### 🫀 Digital Twin Heart")
        heart_html = heart_visual.render(risk)
        st.components.v1.html(heart_html, height=320)

    with col_right:
        st.markdown(f"### 🔍 Why is this patient's risk {risk:.0%}?")
        st.markdown("**Top contributing factors:**")

        pos_factors = table[table.shap > 0].head(3)
        neg_factors = table[table.shap < 0].head(2)

        if not pos_factors.empty:
            st.markdown("🔴 **Factors increasing risk:**")
            for _, r in pos_factors.iterrows():
                st.markdown(f"- **{r['label']}** ({r['value']:g}) &nbsp; `↑ +{r['shap']:.2f}`")

        if not neg_factors.empty:
            st.markdown("🟢 **Factors lowering risk:**")
            for _, r in neg_factors.iterrows():
                st.markdown(f"- **{r['label']}** ({r['value']:g}) &nbsp; `↓ {r['shap']:.2f}`")

        with st.expander("🔎 View Detailed Explanation (SHAP Waterfall & Table)"):
            fig = explainer.plot_waterfall(row)
            st.pyplot(fig)
            st.dataframe(table[["label", "value", "shap"]].rename(
                columns={"label": "Feature", "value": "Value", "shap": "SHAP contribution"}),
                hide_index=True, width="stretch")

    st.divider()

    # 6. Risk History
    st.markdown("### 📈 Risk History")
    hist = [h for h in record.get("history", []) if h.get("risk") is not None]
    if len(hist) >= 2:
        chart_df = pd.DataFrame({
            "Risk (%)": [h["risk"] * 100 for h in hist]
        }, index=[format_last_assessment(h.get("t")) for h in hist])
        st.line_chart(chart_df)
    else:
        st.info("Single assessment on file. Risk history trend will appear as more visits are recorded.")

    st.divider()

    # 7. Digital Twin — 10-Year Risk Projection
    st.markdown("### 🔮 Digital Twin — 10-Year Risk Projection")
    twin = build_twin(username, record, model)
    horizon = st.slider("Projection Horizon (years)", 3, 15, 10, key=f"hz_{username}")
    
    scenario_display_map = {
        "No change": "Current trajectory",
        "BP control": "Blood pressure control",
        "Lipid control": "Lipid control",
        "Exercise program": "Exercise improvement",
        "Combined": "Combined"
    }
    
    chosen_labels = st.multiselect(
        "Intervention Scenarios",
        options=list(scenario_display_map.values()),
        default=list(scenario_display_map.values()),
        key=f"sc_{username}"
    )
    
    inv_map = {v: k for k, v in scenario_display_map.items()}
    trajs = {}
    for lbl in chosen_labels:
        sc_key = inv_map[lbl]
        trajs[lbl] = twin.trajectory(SCENARIOS[sc_key], horizon, n_mc=150)

    if trajs:
        fig2 = plot_trajectories(trajs, f"{profile.get('full_name', username)} — 10-Year Projected Risk", save=False)
        st.pyplot(fig2)
        st.caption("Longitudinal simulation modeling personalized physiological progression rates and evidence-based interventions.")

    st.divider()

    # 8. Download Clinical Report
    pdf_bytes = build_doctor_report(
        patient_name=profile.get("full_name", username),
        patient_username=username,
        doctor_name=user["profile"].get("full_name", user["username"]),
        vitals=record["vitals"],
        risk=risk,
        threshold=0.5,
        model_version=model_version(),
        shap_table=table,
        narrative=narrative,
        waterfall_fig=explainer.plot_waterfall(row),
        scenario_fig=plot_trajectories(trajs, "Scenario projection", save=False) if trajs else None,
        history=record.get("history", [])
    )
    st.download_button("⬇️ Download Clinical Report (PDF)", data=pdf_bytes,
                       file_name=f"{username}_clinical_report.pdf", mime="application/pdf",
                       key=f"dl_{username}", type="primary")


def _profile_editor(user: dict) -> None:
    st.subheader("My Profile")
    p = user["profile"]
    with st.form("doctor_profile"):
        full_name = st.text_input("Full name", p.get("full_name", ""))
        specialty = st.text_input("Specialty", p.get("specialty", ""))
        hospital = st.text_input("Hospital / Clinic", p.get("hospital", ""))
        phone = st.text_input("Phone", p.get("phone", ""))
        email = st.text_input("Email", p.get("email", ""))
        saved = st.form_submit_button("Save Changes", type="primary")
    if saved:
        store.update_profile(user["username"], {"full_name": full_name, "specialty": specialty,
                                                 "hospital": hospital, "phone": phone, "email": email})
        st.success("Profile updated.")
        st.session_state.auth_user = store.get_user(user["username"])
        st.rerun()

    st.divider()
    with st.form("doctor_password"):
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
