"""PDF report generation — a clinical version for doctors, a plain-language version
for patients. Both are built with ReportLab so they download as real, shareable PDFs.
"""
from __future__ import annotations

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .config import LABELS, risk_category

styles = getSampleStyleSheet()
_H1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#1E293B"))
_H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#2563EB"))
_BODY = ParagraphStyle("Body", parent=styles["BodyText"], textColor=colors.HexColor("#1E293B"))
_SMALL = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, textColor=colors.HexColor("#64748B"))
_BANNER = {"Low": colors.HexColor("#16A34A"), "Moderate": colors.HexColor("#D97706"),
          "High": colors.HexColor("#DC2626")}


def _fig_to_image(fig, width=15 * cm) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)
    h = width * 0.55
    return Image(buf, width=width, height=h)


def _trend_figure(history: list[dict]):
    pts = [h for h in history if h.get("risk") is not None]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    xs = list(range(len(pts)))
    ys = [p["risk"] * 100 for p in pts]
    ax.plot(xs, ys, "o-", color="#2563EB", lw=2)
    ax.axhspan(0, 33, color="#16A34A", alpha=0.10)
    ax.axhspan(33, 66, color="#D97706", alpha=0.10)
    ax.axhspan(66, 100, color="#DC2626", alpha=0.10)
    ax.set(ylim=(0, 100), ylabel="Predicted risk (%)", xlabel="Recorded visit #",
          title="Risk over time")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    ax.tick_params(colors="#64748B", labelsize=8)
    ax.xaxis.label.set_color("#1E293B")
    ax.yaxis.label.set_color("#1E293B")
    ax.title.set_color("#1E293B")
    return fig


def _vitals_table(vitals: dict) -> Table:
    rows = [["Measurement", "Value"]] + [[LABELS.get(k, k), f"{v:g}" if isinstance(v, (int, float)) else str(v)]
                                          for k, v in vitals.items()]
    t = Table(rows, colWidths=[7 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white]),
    ]))
    return t


# ------------------------------------------------------------------------------------
def build_doctor_report(*, patient_name: str, patient_username: str, doctor_name: str,
                        vitals: dict, risk: float, threshold: float, model_version: int,
                        shap_table, narrative: str, waterfall_fig, scenario_fig,
                        history: list[dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    cat = risk_category(risk)
    story = [Paragraph("CardioTwin-XAI — Clinical Risk Report", _H1),
            Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}  ·  Attending: {doctor_name}", _SMALL),
            Spacer(1, 10),
            Paragraph(f"Patient: <b>{patient_name}</b>  (ID: {patient_username})", _BODY),
            Spacer(1, 6)]

    band = Table([[f"Predicted CV risk: {risk:.0%}  —  {cat.upper()} RISK BAND"]], colWidths=[16 * cm])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _BANNER[cat]),
                              ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                              ("FONTSIZE", (0, 0), (-1, -1), 13), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                              ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [band, Spacer(1, 10),
             Paragraph(f"Decision threshold in use: {threshold:.2f} · Model (champion) version: v{model_version}", _SMALL),
             Spacer(1, 12), Paragraph("Current vitals", _H2), _vitals_table(vitals), Spacer(1, 14),
             Paragraph("Model explanation (SHAP)", _H2),
             Paragraph(narrative, _BODY), Spacer(1, 8)]

    rows = [["Feature", "Value", "SHAP contribution", "Direction"]]
    for r in shap_table.itertuples():
        rows.append([r.label, f"{r.value:g}", f"{r.shap:+.3f}", "↑ raises risk" if r.shap > 0 else "↓ lowers risk"])
    t = Table(rows, colWidths=[5 * cm, 3 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FAFC"), colors.white])]))
    story += [t, Spacer(1, 12)]
    if waterfall_fig is not None:
        story += [Paragraph("Local feature attribution (waterfall)", _H2), _fig_to_image(waterfall_fig)]
    if scenario_fig is not None:
        story += [Spacer(1, 10), Paragraph("Digital-twin scenario projection", _H2), _fig_to_image(scenario_fig)]
    tf = _trend_figure(history)
    if tf is not None:
        story += [Spacer(1, 10), Paragraph("Longitudinal risk trend", _H2), _fig_to_image(tf)]
    story += [Spacer(1, 16),
             Paragraph("Research prototype for decision support only — not a diagnostic device. "
                      "SHAP values explain the model's output, not verified causal mechanisms; "
                      "correlate with clinical judgement and current guidelines before acting.", _SMALL)]
    doc.build(story)
    return buf.getvalue()


def build_patient_report(*, patient_name: str, patient_username: str, vitals: dict, risk: float,
                         narrative: str, tips: list[str], history: list[dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    cat = risk_category(risk)
    story = [Paragraph("Your Heart Health Summary", _H1),
            Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} for {patient_name}", _SMALL),
            Spacer(1, 12)]
    band = Table([[f"Your current heart-risk level:  {cat.upper()}  ({risk:.0%})"]], colWidths=[16 * cm])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _BANNER[cat]),
                              ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                              ("FONTSIZE", (0, 0), (-1, -1), 14), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                              ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [band, Spacer(1, 14), Paragraph("What this means", _H2), Paragraph(narrative, _BODY), Spacer(1, 10)]
    if tips:
        story += [Paragraph("Suggestions to discuss with your doctor", _H2)]
        for tip in tips:
            story.append(Paragraph(f"• {tip}", _BODY))
        story.append(Spacer(1, 10))
    tf = _trend_figure(history)
    if tf is not None:
        story += [Paragraph("Your risk over time", _H2), _fig_to_image(tf), Spacer(1, 10)]
    story += [Paragraph("Your recorded measurements", _H2), _vitals_table(vitals), Spacer(1, 16),
             Paragraph("This summary is generated by an AI-assisted research tool and is not a medical "
                      "diagnosis. Always discuss results and any lifestyle or treatment changes with your "
                      "doctor before acting on them.", _SMALL)]
    doc.build(story)
    return buf.getvalue()
