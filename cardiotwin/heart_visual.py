"""Realistic anatomical Digital Twin heart visualization.

Renders a self-contained HTML/SVG/CSS medical heart component with anatomical
landmarks (Aorta, Pulmonary Artery, Vena Cava, Ventricles, Coronary Sulcus),
smooth anatomical heartbeat animation, and simulated Digital Twin pulse trace.
All visual states dynamically reflect the model-estimated cardiovascular risk.
"""
from __future__ import annotations

from .config import risk_category, risk_color

# Smooth clinical P-Q-R-S-T simulated ECG wave patterns by risk band
_ECG_PATHS = {
    "Low": "M0,25 L35,25 Q45,25 50,22 Q55,25 65,25 L75,25 L80,30 L85,6 L90,38 L95,25 L105,25 Q115,16 125,25 L165,25 Q175,22 185,25 L195,30 L200,6 L205,38 L210,25 L220,25 Q230,16 240,25 L280,25",
    "Moderate": "M0,25 L30,25 Q38,25 43,20 Q48,25 55,25 L63,25 L68,32 L73,4 L78,40 L83,25 L92,25 Q102,14 112,25 L145,25 Q153,20 160,25 L168,25 L173,32 L178,4 L183,40 L188,25 L197,25 Q207,14 217,25 L250,25 Q258,20 265,25 L280,25",
    "High": "M0,25 L20,25 Q26,25 30,18 Q34,25 40,25 L46,25 L50,35 L55,2 L60,44 L65,25 L73,25 Q82,10 92,23 L115,25 Q121,18 126,25 L132,25 L136,35 L141,2 L146,44 L151,25 L159,25 Q168,10 178,23 L200,25 Q206,18 211,25 L217,25 L221,35 L226,2 L231,44 L236,25 L244,25 Q253,10 263,23 L280,25",
}

_BEAT_DURATIONS = {"Low": 1.05, "Moderate": 0.80, "High": 0.58}
_ECG_DURATIONS = {"Low": 2.6, "Moderate": 1.9, "High": 1.3}
_STATUS_TITLES = {
    "Low": "Low Risk Simulated State",
    "Moderate": "Moderate Risk Simulated State",
    "High": "High Risk Simulated State",
}
_STATUS_DESCS = {
    "Low": "Normal resting rate simulation with steady rhythm.",
    "Moderate": "Mildly elevated workload pattern requiring clinical monitoring.",
    "High": "Marked physiological stress response observed in simulation.",
}


def render(risk: float, subtitle: str | None = None, show_card: bool = True) -> str:
    """Return clean HTML/SVG for the realistic anatomical Digital Twin heart.
    
    Dynamically adjusts pulse rhythm, ECG trace velocity, and risk indicators
    based on the model-estimated cardiovascular risk probability.
    """
    cat = risk_category(risk)
    color = risk_color(risk)
    beat_sec = _BEAT_DURATIONS[cat]
    ecg_sec = _ECG_DURATIONS[cat]
    ecg_d = _ECG_PATHS[cat]
    pct = f"{risk * 100:.0f}%"

    bg_badge = "#F0FDF4" if cat == "Low" else ("#FFFBEB" if cat == "Moderate" else "#FEF2F2")
    border_badge = "#BBF7D0" if cat == "Low" else ("#FDE68A" if cat == "Moderate" else "#FECACA")
    status_desc = subtitle or _STATUS_DESCS[cat]

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                background:#FFFFFF;
                border:1px solid #E2E8F0;
                border-radius:12px;
                padding:16px;
                box-shadow:0 1px 3px rgba(0,0,0,0.04);
                max-width:340px;
                margin:0 auto;
                text-align:center;">
      <style>
        @keyframes naturalHeartbeat {{
          0%   {{ transform: scale(1); }}
          14%  {{ transform: scale(1.08) translateY(-1px); }}
          28%  {{ transform: scale(0.98); }}
          42%  {{ transform: scale(1.05) translateY(-0.5px); }}
          60%  {{ transform: scale(1); }}
          100% {{ transform: scale(1); }}
        }}
        @keyframes traceEcg {{
          from {{ stroke-dashoffset: 280; }}
          to   {{ stroke-dashoffset: 0; }}
        }}
        .dt-heart-vector {{
          transform-origin: 100px 105px;
          animation: naturalHeartbeat {beat_sec}s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        }}
        .dt-ecg-path {{
          stroke-dasharray: 280;
          animation: traceEcg {ecg_sec}s linear infinite;
        }}
      </style>

      <!-- Anatomical Digital Twin Medical Heart -->
      <svg class="dt-heart-vector" width="140" height="145" viewBox="0 0 200 200" style="display:block;margin:0 auto 6px auto;">
        <defs>
          <!-- Muscle Gradients with subtle 3D lighting -->
          <radialGradient id="myoGrad" cx="45%" cy="40%" r="60%">
            <stop offset="0%" stop-color="#E11D48"/>
            <stop offset="45%" stop-color="#BE123C"/>
            <stop offset="85%" stop-color="#881337"/>
            <stop offset="100%" stop-color="#4C0519"/>
          </radialGradient>
          <linearGradient id="aortaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#FB7185"/>
            <stop offset="40%" stop-color="#E11D48"/>
            <stop offset="100%" stop-color="#9F1239"/>
          </linearGradient>
          <linearGradient id="pulmGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#38BDF8"/>
            <stop offset="50%" stop-color="#0284C7"/>
            <stop offset="100%" stop-color="#0369A1"/>
          </linearGradient>
          <linearGradient id="venaCavaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#60A5FA"/>
            <stop offset="70%" stop-color="#2563EB"/>
            <stop offset="100%" stop-color="#1D4ED8"/>
          </linearGradient>
          <!-- Subtle shadow filter -->
          <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.12"/>
          </filter>
        </defs>

        <g filter="url(#softShadow)">
          <!-- Superior Vena Cava (Right) -->
          <path d="M58,28 C58,15 70,12 72,26 L72,60 C68,60 62,56 58,52 Z" fill="url(#venaCavaGrad)" stroke="#1E40AF" stroke-width="0.8"/>

          <!-- Aorta Arch with 3 Supra-Aortic Branch Arteries -->
          <path d="M96,62 C96,30 112,14 135,16 C155,18 160,38 156,58 L142,60 C146,45 142,28 132,28 C122,28 112,38 112,62 Z" fill="url(#aortaGrad)" stroke="#881337" stroke-width="0.8"/>
          <!-- Brachiocephalic trunk -->
          <path d="M116,22 L114,8 C114,6 119,6 120,8 L123,20 Z" fill="url(#aortaGrad)" stroke="#881337" stroke-width="0.6"/>
          <!-- Left common carotid -->
          <path d="M128,18 L129,7 C129,5 134,5 134,7 L134,18 Z" fill="url(#aortaGrad)" stroke="#881337" stroke-width="0.6"/>
          <!-- Left subclavian artery -->
          <path d="M140,21 L144,9 C144,7 149,8 148,10 L145,23 Z" fill="url(#aortaGrad)" stroke="#881337" stroke-width="0.6"/>

          <!-- Pulmonary Artery Trunk (Crossing in front of aorta) -->
          <path d="M82,56 C82,42 98,36 115,44 C120,47 122,54 120,62 L100,72 C94,68 86,64 82,56 Z" fill="url(#pulmGrad)" stroke="#0369A1" stroke-width="0.8"/>
          <path d="M115,44 L128,40 C130,39 133,43 131,46 L120,52 Z" fill="url(#pulmGrad)" stroke="#0369A1" stroke-width="0.6"/>

          <!-- Right Atrium Auricle -->
          <path d="M54,58 C42,65 40,82 46,98 C50,105 60,108 68,106 C70,95 72,75 66,62 Z" fill="#BE123C" stroke="#881337" stroke-width="0.8"/>

          <!-- Left Atrium Auricle -->
          <path d="M142,66 C155,72 160,86 156,98 C152,106 142,108 136,104 C138,92 140,78 142,66 Z" fill="#9F1239" stroke="#4C0519" stroke-width="0.8"/>

          <!-- Main Ventricular Muscle (Left & Right Ventricles + Apex) -->
          <path d="M58,95 C45,115 54,152 82,176 C94,186 102,190 108,190 C114,190 125,182 142,164 C162,142 166,110 150,90 C136,75 118,78 102,82 C85,78 68,80 58,95 Z" fill="url(#myoGrad)" stroke="#4C0519" stroke-width="1.2"/>

          <!-- Anterior Interventricular Sulcus & Coronary Arteries -->
          <path d="M102,82 Q96,120 108,188" fill="none" stroke="#881337" stroke-width="2.5" opacity="0.6"/>
          <!-- Left Anterior Descending Artery (LAD) -->
          <path d="M100,86 Q95,118 106,184" fill="none" stroke="#F43F5E" stroke-width="1.6"/>
          <!-- Diagonal / Marginal Arterial Branches -->
          <path d="M98,105 Q85,114 74,124" fill="none" stroke="#FDA4AF" stroke-width="1.1"/>
          <path d="M100,128 Q88,138 78,150" fill="none" stroke="#FDA4AF" stroke-width="1.0"/>
          <path d="M101,112 Q115,124 130,132" fill="none" stroke="#FDA4AF" stroke-width="1.1"/>
          <path d="M103,138 Q118,148 128,156" fill="none" stroke="#FDA4AF" stroke-width="0.9"/>

          <!-- Cardiac Vein highlights (Blue branches) -->
          <path d="M104,90 Q100,122 110,182" fill="none" stroke="#38BDF8" stroke-width="1.0" opacity="0.85"/>
          <path d="M104,115 Q120,128 134,136" fill="none" stroke="#38BDF8" stroke-width="0.8" opacity="0.85"/>

          <!-- Muscle Volume Light Refraction (Depth reflection) -->
          <path d="M62,102 C58,118 64,142 84,166" fill="none" stroke="#FFFFFF" stroke-width="2.5" opacity="0.22" stroke-linecap="round"/>
        </g>
      </svg>

      <!-- Digital Twin Simulated Pulse Trace -->
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px 8px;margin:8px 0 10px 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
          <span style="font-size:0.72rem;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.4px;">Simulated ECG / Pulse</span>
          <span style="font-size:0.72rem;font-weight:600;color:{color};">{pct} Risk</span>
        </div>
        <svg width="100%" height="32" viewBox="0 0 280 50" preserveAspectRatio="none" style="display:block;">
          <line x1="0" y1="25" x2="280" y2="25" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="3,3"/>
          <path class="dt-ecg-path" d="{ecg_d}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <!-- Risk State Badge & Description -->
      <div style="display:inline-block;padding:4px 12px;border-radius:20px;background:{bg_badge};border:1px solid {border_badge};font-size:0.82rem;font-weight:700;color:{color};letter-spacing:0.4px;margin-bottom:4px;">
        ● {_STATUS_TITLES[cat].upper()}
      </div>
      <div style="font-size:0.80rem;color:#64748B;line-height:1.35;margin-top:4px;">
        {status_desc}
      </div>
    </div>
    """

