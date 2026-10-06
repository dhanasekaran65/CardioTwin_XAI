"""Matplotlib figures for the pipeline."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from .config import FIG_DIR, LABELS


SCENARIO_COLORS = {
    "No change": "#DC2626",
    "Current trajectory": "#DC2626",
    "BP control": "#2563EB",
    "Blood pressure control": "#2563EB",
    "Lipid control": "#0F766E",
    "Exercise program": "#0284C7",
    "Exercise improvement": "#0284C7",
    "Combined": "#16A34A"
}
PALETTE = ["#2563EB", "#0F766E", "#0284C7", "#16A34A", "#64748B"]


def _apply_clinical_style(ax):
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
    ax.tick_params(colors="#64748B", labelsize=8.5)
    ax.xaxis.label.set_color("#1E293B")
    ax.yaxis.label.set_color("#1E293B")
    ax.title.set_color("#1E293B")


def _save(fig, name, save=True):
    fig.patch.set_facecolor("#FFFFFF")
    fig.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(fig)
        return None
    return fig


def plot_roc_calibration(y, p, name="performance.png", save=True):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2, color="#2563EB", label="Model")
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="#64748B", lw=1, label="Chance")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    _apply_clinical_style(ax[0])
    ax[0].legend(fontsize=8)

    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax[1].plot(mean, frac, "o-", color="#0F766E", lw=2, label="Observed")
    ax[1].plot([0, 1], [0, 1], linestyle="--", color="#64748B", lw=1, label="Perfect")
    ax[1].set(xlabel="Predicted risk", ylabel="Observed frequency", title="Calibration (test)")
    _apply_clinical_style(ax[1])
    ax[1].legend(fontsize=8)
    return _save(fig, name, save)


def plot_trajectories(trajs: dict, title: str, name="twin_trajectories.png", save=True):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (label, df) in enumerate(trajs.items()):
        color = SCENARIO_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.plot(df.year, df.risk_mean, lw=2, label=label, color=color)
        ax.fill_between(df.year, df.risk_p10, df.risk_p90, alpha=0.10, color=color)
    ax.set(xlabel="Years ahead", ylabel="Predicted CV risk", title=title, ylim=(0, 1))
    _apply_clinical_style(ax)
    ax.legend(fontsize=8, framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0")
    return _save(fig, name, save)


def plot_scenario_attribution(delta, scenario: str, name="scenario_attribution.png", save=True):
    d = delta.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#0F766E" if v < 0 else "#DC2626" for v in d.values]
    ax.barh([LABELS[i] for i in d.index], d.values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#64748B", lw=0.8, linestyle="--")
    ax.set(xlabel="Δ SHAP (log-odds) vs 'No change'", title=f"Why risk differs at horizon: {scenario}")
    _apply_clinical_style(ax)
    return _save(fig, name, save)


def plot_deletion(curve, name="faithfulness.png", save=True):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    color_map = {"SHAP": "#2563EB", "Random": "#64748B"}
    for m, g in curve.groupby("method"):
        ax.plot(g.k, g.mean_abs_change, "o-", label=m, color=color_map.get(m, "#0F766E"), lw=2)
    ax.set(xlabel="# top features neutralised", ylabel="Mean |Δ risk|", title="Deletion faithfulness")
    _apply_clinical_style(ax)
    ax.legend(fontsize=8, framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0")
    return _save(fig, name, save)


def plot_self_learning(adaptive, frozen, name="self_learning.png", save=True):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, col, ttl in zip(ax, ("logloss", "auroc"), ("Log-loss (lower better)", "AUROC (higher better)")):
        a.plot(adaptive.batch, adaptive[col], "o-", label="Self-learning twin", color="#2563EB", lw=2)
        a.plot(frozen.batch, frozen[col], "s--", label="Frozen baseline", color="#64748B", lw=1.5)
        for r in adaptive[adaptive.promoted].itertuples():
            a.axvline(r.batch, color="#16A34A", alpha=0.3, lw=4)
        a.set(xlabel="Stream batch", title=ttl)
        _apply_clinical_style(a)
    ax[0].legend(fontsize=8, framealpha=0.9, facecolor="#FFFFFF", edgecolor="#E2E8F0")
    fig.suptitle("Prequential performance under drift (green = model promoted)", color="#1E293B", fontsize=11)
    return _save(fig, name, save)
