"""Matplotlib chart builders (all values come from real data/model outputs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from risk_analysis import risk_color

INK, MUTED, GRID = "#1f2a44", "#7a869a", "#e6eaf1"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def gauge(value, title, color="#3b6fd4"):
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(aspect="equal"))
    th = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(th), np.sin(th), lw=16, color=GRID, solid_capstyle="round")
    t2 = np.linspace(np.pi, np.pi - np.pi * np.clip(value, 0, 100) / 100, 100)
    ax.plot(np.cos(t2), np.sin(t2), lw=16, color=color, solid_capstyle="round")
    ax.text(0, 0.05, f"{value:.0f}", ha="center", fontsize=28, weight="bold", color=INK)
    ax.text(0, -0.3, title, ha="center", fontsize=10, color=MUTED)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.5, 1.2); ax.axis("off")
    return fig


def radar(values: dict, title=""):
    labels = list(values); vals = [values[k] for k in labels]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]; ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), subplot_kw=dict(polar=True))
    ax.plot(ang, vals, color="#3b6fd4", lw=2); ax.fill(ang, vals, color="#3b6fd4", alpha=0.25)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(0, 10); ax.set_yticks([2.5, 5, 7.5, 10]); ax.set_yticklabels([])
    ax.grid(color=GRID); ax.spines["polar"].set_color(GRID)
    ax.set_title(title or "Level of concern (0 = low, 10 = high)", fontsize=9, color=MUTED, pad=14)
    return fig


def prob_bars(probs: dict):
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    cats = list(probs); v = [probs[c] * 100 for c in cats]
    ax.barh(cats[::-1], v[::-1], color=[risk_color(c) for c in cats[::-1]])
    for i, x in enumerate(v[::-1]):
        ax.text(x + 1, i, f"{x:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 110); _style(ax); ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    return fig


def contributions(contrib: dict, labels: dict, title):
    items = sorted(contrib.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.barh([labels[k] for k, _ in items], [v * 100 for _, v in items],
            color=["#d9534f" if v > 0 else "#2e9e6b" for _, v in items])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Change in High-risk probability (percentage points)", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left"); _style(ax)
    return fig


def importance(imp: dict, labels: dict, title):
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.barh([labels[k] for k, _ in items], [v for _, v in items], color="#3b6fd4")
    ax.set_title(title, fontsize=10, color=INK, loc="left"); _style(ax)
    return fig


def trend_line(x, series: dict, title, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    for (name, y), c in zip(series.items(), ["#3b6fd4", "#d9534f", "#2e9e6b", "#e0a030"]):
        ax.plot(x, y, marker="o", lw=2, label=name, color=c)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(list(x)); ax.set_xlabel("Session #", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left"); _style(ax)
    return fig


def timeline(categories):
    fig, ax = plt.subplots(figsize=(6, 1.6))
    lvl = {"Low": 0, "Moderate": 1, "High": 2}
    for i, c in enumerate(categories):
        ax.scatter(i + 1, lvl[c], s=220, color=risk_color(c), zorder=3)
    ax.plot(range(1, len(categories) + 1), [lvl[c] for c in categories], color=GRID, zorder=1)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Low", "Moderate", "High"]); ax.set_ylim(-0.5, 2.5)
    ax.set_xticks(range(1, len(categories) + 1)); _style(ax)
    return fig


def confusion(cm, classes, title):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(title, fontsize=10, color=INK)
    return fig


def model_comparison(metrics: dict):
    names = list(metrics); keys = ["accuracy", "precision", "recall", "f1"]
    w = 0.8 / len(names); fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, n in enumerate(names):
        ax.bar(np.arange(len(keys)) + i * w, [metrics[n][k] for k in keys], w, label=n,
               color=["#3b6fd4", "#8fb0ea"][i % 2])
    ax.set_xticks(np.arange(len(keys)) + w * (len(names) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"]); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8); _style(ax)
    return fig
