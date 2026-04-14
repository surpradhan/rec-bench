import sys
sys.path.insert(0, '.')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

RESULTS_CSV = Path("results/benchmark_results.csv")
OUTPUT_PATH = Path("results/plots/benchmark_comparison.png")

MODEL_ORDER = [
    "Hybrid", "ALS", "UserCF", "ItemCF", "Popularity", "SVD", "NCF", "ContentBased"
]

# Tier-based colouring: meaning over decoration
TIERS = {
    "Hybrid":       "strong",
    "ALS":          "strong",
    "UserCF":       "mid",
    "ItemCF":       "mid",
    "Popularity":   "mid",
    "SVD":          "mid",
    "NCF":          "mid",
    "ContentBased": "weak",
}

TIER_COLOR = {
    "strong": "#4F46E5",   # indigo  — top performers
    "mid":    "#0EA5E9",   # sky     — middle pack
    "weak":   "#94A3B8",   # slate   — trailing
}

NDCG_COLOR  = "#4F46E5"   # indigo  — NDCG@10 dot
PREC_COLOR  = "#F59E0B"   # amber   — Precision@10 dot

BG     = "#FFFFFF"
PANEL  = "#F8FAFC"
GRID   = "#E2E8F0"
TEXT   = "#0F172A"
MUTED  = "#64748B"


def plot_benchmark_comparison(csv_path: Path = RESULTS_CSV,
                               output_path: Path = OUTPUT_PATH) -> None:
    df = pd.read_csv(csv_path).set_index("model")
    df_sorted = df.loc[MODEL_ORDER] if all(m in df.index for m in MODEL_ORDER) \
        else df.sort_values("NDCG@10", ascending=False)

    models  = df_sorted.index.tolist()
    ndcg    = df_sorted["NDCG@10"].tolist()
    prec    = df_sorted["Precision@10"].tolist()
    n       = len(models)
    y_pos   = list(range(n - 1, -1, -1))   # top model at top

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Grid ──────────────────────────────────────────────────────────────────
    x_max = max(ndcg) * 1.28
    ax.set_xlim(-0.003, x_max)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0, pad=6)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks([])

    # ── Tier separator bands ───────────────────────────────────────────────────
    tier_breaks = [
        (6.5, 7.5, "STRONG",  "#EEF2FF"),   # between ALS and UserCF
        (-0.5, 0.5, "WEAK",   "#F8FAFC"),   # ContentBased row
    ]
    # Shade every row subtly by tier
    for i, (model, y) in enumerate(zip(models, y_pos)):
        tier = TIERS[model]
        row_color = {"strong": "#EEF2FF", "mid": PANEL, "weak": "#F8FAFC"}[tier]
        ax.barh(y, x_max, height=0.88, left=0, color=row_color,
                edgecolor="none", zorder=1)

    # Horizontal divider between tiers
    ax.axhline(y=5.5, color=GRID, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(y=0.5, color=GRID, linewidth=1.2, linestyle="--", zorder=2)

    # ── Dumbbell stems ────────────────────────────────────────────────────────
    for y, nd, pr, model in zip(y_pos, ndcg, prec, models):
        lo, hi = min(nd, pr), max(nd, pr)
        ax.plot([lo, hi], [y, y], color="#CBD5E1", linewidth=2.0,
                solid_capstyle="round", zorder=3)

    # ── Dots ──────────────────────────────────────────────────────────────────
    dot_kw = dict(zorder=5, linewidths=1.5)
    ax.scatter(ndcg, y_pos, s=110, color=NDCG_COLOR,
               edgecolors="white", **dot_kw)
    ax.scatter(prec, y_pos, s=110, color=PREC_COLOR,
               edgecolors="white", **dot_kw)

    # ── Value labels ──────────────────────────────────────────────────────────
    for y, nd, pr in zip(y_pos, ndcg, prec):
        ax.text(max(nd, pr) + x_max * 0.013, y,
                f"{nd:.4f}", va="center", ha="left",
                fontsize=8.2, color=NDCG_COLOR, fontweight="bold")
        # Precision label below if values are close
        offset = -0.32 if abs(nd - pr) < 0.003 else 0
        ax.text(max(nd, pr) + x_max * 0.013, y + offset,
                f"{pr:.4f}", va="center", ha="left",
                fontsize=8.2, color=PREC_COLOR, fontweight="bold")

    # ── Model name labels + rank badges ───────────────────────────────────────
    for i, (model, y) in enumerate(zip(models, y_pos)):
        rank = i + 1
        tier = TIERS[model]
        tc   = TIER_COLOR[tier]

        # Rank badge
        ax.text(-0.0025, y, f"#{rank}",
                va="center", ha="right", fontsize=8,
                color=tc, fontweight="bold",
                fontfamily="monospace")

        # Model name
        ax.text(0.0, y + 0.32, model,
                va="bottom", ha="left", fontsize=10.5,
                color=TEXT, fontweight="bold" if tier == "strong" else "normal")

    # ── Tier labels (right margin) ─────────────────────────────────────────────
    for label, y_center, color in [
        ("TOP TIER",    6.5,  TIER_COLOR["strong"]),
        ("MID TIER",    3.0,  TIER_COLOR["mid"]),
        ("BOTTOM TIER", 0.0,  TIER_COLOR["weak"]),
    ]:
        ax.text(x_max * 0.995, y_center, label,
                va="center", ha="right", fontsize=7.5,
                color=color, fontweight="bold", alpha=0.6,
                rotation=0)

    # ── Winner callout ────────────────────────────────────────────────────────
    ax.annotate(
        "  Best overall\n  NDCG@10: 0.1166",
        xy=(ndcg[0], y_pos[0]),
        xytext=(ndcg[0] + x_max * 0.08, y_pos[0] + 1.1),
        fontsize=8.5, color=NDCG_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=NDCG_COLOR,
                        lw=1.2, connectionstyle="arc3,rad=-0.25"),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EEF2FF",
                  edgecolor=NDCG_COLOR, linewidth=1.0),
        zorder=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=NDCG_COLOR,
                      markersize=9, markeredgecolor="white", label="NDCG@10"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=PREC_COLOR,
                      markersize=9, markeredgecolor="white", label="Precision@10"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              framealpha=0.95, edgecolor=GRID, fontsize=9.5,
              handletextpad=0.6, borderpad=0.7)

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.055, 0.96,
             "Hybrid and ALS dominate; Content-Based trails by 14×",
             fontsize=14, fontweight="bold", color=TEXT, va="top")
    fig.text(0.055, 0.915,
             "MovieLens 100K  ·  8 algorithms  ·  Final test-set results  "
             "·  Relevance threshold ≥ 4  ·  K = 10",
             fontsize=9, color=MUTED, va="top")

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.055, 0.015,
             "Evaluation: time-based train/val/test split  ·  "
             "Tuned models retrained on train+val  ·  "
             "MAP denominator = min(|relevant|, K)",
             fontsize=7.5, color=MUTED, va="bottom", style="italic")

    ax.set_ylim(-0.75, n - 0.25)
    plt.subplots_adjust(left=0.055, right=0.97, top=0.88, bottom=0.07)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    plot_benchmark_comparison()
