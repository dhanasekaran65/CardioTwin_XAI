"""CardioTwin-XAI web app — role-based login, then a dedicated Doctor or Patient
dashboard. Run with:  streamlit run app.py
"""
import streamlit as st

from cardiotwin.config import ensure_dirs
from cardiotwin.store import store
from cardiotwin import ui_common, ui_doctor, ui_patient

ensure_dirs()
st.set_page_config(page_title="CardioTwin-XAI", page_icon="🫀", layout="wide")
ui_common.inject_custom_css()

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None


def _login_page() -> None:
    st.markdown(
        "<div style='text-align:center; padding: 24px 0 16px 0;'>"
        "<h1 style='color:#1E293B; margin-bottom:6px; font-weight:700;'>🫀 CardioTwin-XAI</h1>"
        "<p style='color:#64748B; font-size:1.05rem; margin:0;'>Self-learning digital twin for explainable "
        "cardiovascular risk prognosis</p>"
        "</div>",
        unsafe_allow_html=True
    )
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        login_tab, register_tab = st.tabs(["🔐 Log in", "🆕 Register (Patient)"])

        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log in", type="primary", width="stretch")
            if submitted:
                user = store.authenticate(username, password)
                if user:
                    st.session_state.auth_user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with register_tab:
            st.caption("New patients can self-register. Doctor accounts are created by "
                      "your clinic administrator.")
            doctors = store.list_doctors()
            with st.form("register_form"):
                full_name = st.text_input("Full name")
                age = st.number_input("Age", 20, 95, 50)
                sex = st.selectbox("Sex", ["Male", "Female"])
                phone = st.text_input("Phone")
                email = st.text_input("Email")
                doctor_choice = st.selectbox(
                    "Choose your doctor", [d["username"] for d in doctors],
                    format_func=lambda u: next(d["profile"]["full_name"] for d in doctors if d["username"] == u)
                ) if doctors else None
                new_username = st.text_input("Choose a username")
                new_password = st.text_input("Choose a password", type="password")
                reg_submitted = st.form_submit_button("Create my account", type="primary",
                                                      width="stretch")
            if reg_submitted:
                try:
                    store.create_user(new_username, new_password, "patient",
                                      {"full_name": full_name, "age": int(age), "sex": sex,
                                       "phone": phone, "email": email}, doctor=doctor_choice)
                    st.success("Account created! Switch to the **Log in** tab to continue.")
                except ValueError as e:
                    st.error(str(e))


def _sign_out_button() -> None:
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Sign out", width="stretch"):
            st.session_state.auth_user = None
            st.rerun()


user = st.session_state.auth_user
if user is None:
    _login_page()
else:
    with st.sidebar:
        st.markdown(f"**{user['profile'].get('full_name', user['username'])}**")
        st.caption(f"Role: {user['role'].capitalize()}")
    if user["role"] == "doctor":
        ui_doctor.render(user)
    else:
        ui_patient.render(user)
    _sign_out_button()
