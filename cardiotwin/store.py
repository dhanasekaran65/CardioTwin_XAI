"""Lightweight persistent store for accounts, profiles and patient vitals history.

This is a small JSON-backed "database" (no external server needed) that gives the
Streamlit app real, cross-session state: accounts survive a restart, a vitals update
a patient logs is immediately visible on the doctor's dashboard, and profile edits
persist. Passwords are salted + PBKDF2-hashed. This is a demo-grade auth layer, not
a substitute for a real identity provider in a production deployment.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone

from .config import APP_DIR, FEATURES, ensure_dirs

DB_PATH = APP_DIR / "app_db.json"

# Reasonable population-average starting vitals for a freshly registered patient
# (used until their doctor or the patient themselves logs real measurements).
DEFAULT_VITALS = {"age": 50, "sex": 1, "cp": 1, "trestbps": 125, "chol": 210, "fbs": 0,
                  "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 0.8, "slope": 1,
                  "ca": 0, "thal": 2}

_ROUND = {"age": 0, "trestbps": 0, "chol": 0, "thalach": 0, "oldpeak": 1}


def _clean_vitals(vitals: dict) -> dict:
    """Round to clinically sensible precision so stored/displayed readings never
    show noisy float artefacts (e.g. from a simulated wearable sync)."""
    out = dict(vitals)
    for k, nd in _ROUND.items():
        if k in out and out[k] is not None:
            out[k] = round(float(out[k]), nd)
            if nd == 0:
                out[k] = int(out[k])
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return digest, salt


class Store:
    """Thread-unsafe-but-fine-for-a-demo JSON store, re-read/written on every call
    so multiple Streamlit sessions on the same machine see each other's updates."""

    def __init__(self, path=DB_PATH):
        self.path = path
        ensure_dirs()
        if not self.path.exists():
            self._write({"users": {}, "patients": {}})
            self._seed_demo_accounts()

    # ---- raw io -----------------------------------------------------------------
    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"users": {}, "patients": {}}

    def _write(self, db: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(db, indent=2))
        os.replace(tmp, self.path)

    # ---- accounts -----------------------------------------------------------------
    def user_exists(self, username: str) -> bool:
        return username.lower() in self._read()["users"]

    def create_user(self, username: str, password: str, role: str, profile: dict,
                    doctor: str | None = None) -> None:
        username = username.lower().strip()
        if not username or not password:
            raise ValueError("Username and password are required.")
        if role not in ("doctor", "patient"):
            raise ValueError("role must be 'doctor' or 'patient'")
        db = self._read()
        if username in db["users"]:
            raise ValueError(f"Username '{username}' is already taken.")
        digest, salt = _hash(password)
        db["users"][username] = {"password_hash": digest, "salt": salt, "role": role,
                                 "profile": profile, "created_at": _now()}
        if role == "patient":
            vitals = dict(DEFAULT_VITALS)
            vitals["age"] = profile.get("age", DEFAULT_VITALS["age"])
            vitals["sex"] = 1 if profile.get("sex", "Male") == "Male" else 0
            db["patients"][username] = {"doctor": doctor, "vitals": vitals, "history": [
                {"t": _now(), "source": "registration", "vitals": vitals, "risk": None}]}
        self._write(db)

    def authenticate(self, username: str, password: str) -> dict | None:
        db = self._read()
        rec = db["users"].get(username.lower().strip())
        if not rec:
            return None
        digest, _ = _hash(password, rec["salt"])
        if secrets.compare_digest(digest, rec["password_hash"]):
            return {"username": username.lower().strip(), **rec}
        return None

    def get_user(self, username: str) -> dict | None:
        rec = self._read()["users"].get(username.lower())
        return {"username": username.lower(), **rec} if rec else None

    def update_profile(self, username: str, fields: dict) -> None:
        db = self._read()
        username = username.lower()
        if username not in db["users"]:
            raise KeyError(username)
        db["users"][username]["profile"].update(fields)
        if db["users"][username]["role"] == "patient" and username in db["patients"]:
            if "age" in fields:
                db["patients"][username]["vitals"]["age"] = fields["age"]
            if "sex" in fields:
                db["patients"][username]["vitals"]["sex"] = 1 if fields["sex"] == "Male" else 0
        self._write(db)

    def change_password(self, username: str, new_password: str) -> None:
        db = self._read()
        digest, salt = _hash(new_password)
        db["users"][username.lower()]["password_hash"] = digest
        db["users"][username.lower()]["salt"] = salt
        self._write(db)

    def list_doctors(self) -> list[dict]:
        db = self._read()
        return [{"username": u, **r} for u, r in db["users"].items() if r["role"] == "doctor"]

    # ---- patients / vitals ---------------------------------------------------------
    def list_patients_for_doctor(self, doctor_username: str) -> list[dict]:
        db = self._read()
        out = []
        for uname, rec in db["patients"].items():
            if rec.get("doctor") == doctor_username.lower():
                profile = db["users"][uname]["profile"]
                out.append({"username": uname, "profile": profile, **rec})
        return out

    def get_patient_record(self, username: str) -> dict | None:
        db = self._read()
        rec = db["patients"].get(username.lower())
        if not rec:
            return None
        return {"username": username.lower(), "profile": db["users"][username.lower()]["profile"], **rec}

    def assign_doctor(self, patient_username: str, doctor_username: str) -> None:
        db = self._read()
        db["patients"][patient_username.lower()]["doctor"] = doctor_username.lower()
        self._write(db)

    def record_observation(self, patient_username: str, vitals: dict, source: str,
                           risk: float | None = None) -> None:
        """Persist a new set of vitals (from a doctor visit, a patient self-log, or a
        simulated wearable sync) and append it to that patient's longitudinal history."""
        db = self._read()
        username = patient_username.lower()
        rec = db["patients"].setdefault(username, {"doctor": None, "vitals": dict(DEFAULT_VITALS),
                                                     "history": []})
        clean = _clean_vitals({k: v for k, v in vitals.items() if k in FEATURES})
        rec["vitals"].update(clean)
        rec["history"].append({"t": _now(), "source": source, "vitals": dict(rec["vitals"]),
                               "risk": risk})
        rec["history"] = rec["history"][-200:]
        self._write(db)

    # ---- demo seed data --------------------------------------------------------------
    def _seed_demo_accounts(self) -> None:
        try:
            self.create_user("dr.asha", "doctor123", "doctor",
                             {"full_name": "Dr. Asha Rao", "specialty": "Cardiology",
                              "hospital": "St. Xavier General Hospital", "phone": "+91-98765-43210",
                              "email": "asha.rao@example-hospital.in"})
            self.create_user("patient1", "patient123", "patient",
                             {"full_name": "Ravi Kumar", "age": 58, "sex": "Male",
                              "phone": "+91-90000-11111", "email": "ravi.kumar@example.com"},
                             doctor="dr.asha")
            self.create_user("patient2", "patient123", "patient",
                             {"full_name": "Fatima Sheikh", "age": 47, "sex": "Female",
                              "phone": "+91-90000-22222", "email": "fatima.sheikh@example.com"},
                             doctor="dr.asha")
            # give the seeded patients slightly different (non-default) vitals so the
            # demo shows a spread of risk levels out of the box
            self.record_observation("patient1", {"trestbps": 140, "chol": 240, "thalach": 140,
                                                  "oldpeak": 1.0, "cp": 1, "exang": 0, "slope": 2,
                                                  "ca": 0, "thal": 2}, source="seed")
            self.record_observation("patient2", {"trestbps": 118, "chol": 190, "thalach": 168,
                                                  "oldpeak": 0.2, "cp": 2, "exang": 0, "ca": 0,
                                                  "thal": 1}, source="seed")
        except ValueError:
            pass  # already seeded


store = Store()
