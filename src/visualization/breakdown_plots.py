"""
Breakdown analysis visualisations.

Layout
------
Row 1 (taller): two heatmaps side-by-side, tightly coupled.
  Left  — NDCG@10 by item popularity bucket (model labels here only)
  Right — NDCG@10 by user activity tier    (no duplicate y-labels)
Row 2 (shorter, full width): win / loss horizontal bar.

Design decisions
----------------
- Shared vmax across both heatmaps so colour darkness is directly comparable.
- Colorbars removed — values are labelled in every cell; colorbars add ink, not information.
- Right heatmap omits y-axis labels (reader carries them from the left panel).
- Heatmaps visually separated into "strong" / "mid" / "weak" tier zones.
- Win/loss separates model bars from "tie" and "none" with a gap and different style.
- "none" (39 % floor) foregrounded as the headline finding, not buried.
- n_users shown as a small subscript in every cell for reliability context.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

OUTPUT_PATH = Path("results/plots/breakdown_overview.png")

MODEL_ORDER = ["Hybrid", "ALS", "UserCF", "ItemCF", "Popularity", "SVD", "NCF", "ContentBased"]

MODEL_COLOR = {
    "Hybrid":       "#F5C518",
    "ALS":          "#e07b32",
    "UserCF":       "#5b9bd5",
    "ItemCF":       "#7b6fd4",
    "Popularity":   "#52b788",
    "SVD":          "#d45bb0",
    "NCF":          "#d47a5b",
    "ContentBased": "#6b9e7a",
}

# Tier bands for visual grouping on heatmaps
STRONG_MODELS = {"Hybrid", "ALS"}
WEAK_MODELS   = {"ContentBased"}

TIER_BAND = {
    "strong": "#16161f",   # faint gold tint row
    "mid":    "#0f0f1a",
    "weak":   "#0d0d17",
}

BG    = "#08080E"
PANEL = "#0f0f1a"
GRID  = "#2a2a3a"
SEP   = "#3a3a52"        # separator lines
TEXT  = "#e8e8f0"
MUTED = "#7a7a9a"
DIM   = "#4a4a6a"

BUCKET_LABELS = {
    "head":  "Head\n(top 20 %)",
    "torso": "Torso\n(mid 30 %)",
    "tail":  "Tail\n(btm 50 %)",
}
TIER_LABELS = {
    "cold":     "Cold\n(≤33rd pct)",
    "moderate": "Moderate\n(33–66th)",
    "active":   "Active\n(>66th pct)",
}


def _ordered_models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present]


def _row_band(model: str) -> str:
    if model in STRONG_MODELS:
        return "strong"
    if model in WEAK_MODELS:
        return "weak"
    return "mid"


def _draw_heatmap(
    ax,
    pivot_v: pd.DataFrame,
    pivot_n: pd.DataFrame,
    col_labels: list[str],
    title: str,
    vmax: float,
    show_yticklabels: bool,
    ncf_flag_cols: list[str] | None = None,
) -> None:
    models = pivot_v.index.tolist()
    cols   = list(pivot_v.columns)
    data_v = pivot_v.values
    data_n = pivot_n.values
    n_rows, n_cols = data_v.shape

    # Custom dark→gold colormap to match cinema theme
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cinema_gold", ["#16161f", "#c9a01a", "#F5C518"]
    )
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # ── Tier row bands ────────────────────────────────────────────────────
    for r, model in enumerate(models):
        band_color = TIER_BAND[_row_band(model)]
        ax.barh(r, n_cols, left=-0.5, height=1.0,
                color=band_color, edgecolor="none", zorder=0)

    # ── Cells ─────────────────────────────────────────────────────────────
    for r in range(n_rows):
        for c in range(n_cols):
            val  = data_v[r, c]
            n    = int(data_n[r, c])
            rgba = cmap(norm(val))

            rect = plt.Rectangle(
                [c - 0.46, r - 0.42], 0.92, 0.84,
                facecolor=rgba, edgecolor="none",
                zorder=1, clip_on=False,
            )
            ax.add_patch(rect)

            # Luminance-based text colour — on dark bg, bright cells get dark text
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            tc      = "#08080E" if lum > 0.35 else TEXT
            tc_dim  = "#1a1a2a" if lum > 0.35 else MUTED

            ax.text(c, r + 0.10, f"{val:.4f}",
                    ha="center", va="center",
                    fontsize=8.8, color=tc, fontweight="bold", zorder=2)
            ax.text(c, r - 0.26, f"n={n}",
                    ha="center", va="center",
                    fontsize=6.2, color=tc_dim, zorder=2)

    # ── NCF=0 flag ────────────────────────────────────────────────────────
    if ncf_flag_cols and "NCF" in models:
        r_ncf = models.index("NCF")
        for col_name in ncf_flag_cols:
            if col_name in cols:
                c = cols.index(col_name)
                ax.text(c + 0.41, r_ncf + 0.39, "▲",
                        ha="center", va="center",
                        fontsize=6.5, color="#DC2626", zorder=3,
                        fontweight="bold")

    # ── Separator lines between tiers ─────────────────────────────────────
    # Between ALS (row 1) and UserCF (row 2)
    ax.axhline(1.5, color=SEP, linewidth=1.0, linestyle="--", zorder=3)
    # Between NCF (row 6) and ContentBased (row 7)
    ax.axhline(6.5, color=SEP, linewidth=1.0, linestyle="--", zorder=3)

    # ── Axes config ───────────────────────────────────────────────────────
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9.0, color=MUTED, ha="center")

    ax.set_yticks(range(n_rows))
    if show_yticklabels:
        ax.set_yticklabels(models, fontsize=10.0)
        for tick, model in zip(ax.get_yticklabels(), models):
            tick.set_color(MODEL_COLOR.get(model, TEXT))
            tick.set_fontweight("bold" if model in STRONG_MODELS else "normal")
    else:
        ax.set_yticklabels([])

    # Panel title
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=TEXT, pad=10)

    # Thin outer border
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SEP)
        ax.spines[side].set_linewidth(0.6)


def _draw_win_bar(ax, win_df: pd.DataFrame, n_total: int) -> None:
    # Split into models vs meta-entries (none / tie)
    model_rows = win_df[~win_df["model"].isin(["none", "tie"])].copy()
    meta_rows  = win_df[win_df["model"].isin(["none", "tie"])].copy()

    model_rows = model_rows.sort_values("wins", ascending=True)
    meta_rows  = meta_rows.sort_values("wins", ascending=True)

    # Stack: meta at bottom, gap, models above
    GAP    = 0.9
    y_meta = list(range(len(meta_rows)))
    y_model = [y + len(meta_rows) + GAP for y in range(len(model_rows))]

    all_models = meta_rows["model"].tolist() + model_rows["model"].tolist()
    all_wins   = meta_rows["wins"].tolist()  + model_rows["wins"].tolist()
    all_y      = y_meta + y_model

    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_visible(False)

    max_w = max(all_wins)

    for y, model, w in zip(all_y, all_models, all_wins):
        is_meta = model in ("none", "tie")
        color   = MODEL_COLOR.get(model, DIM) if not is_meta else DIM
        alpha   = 0.45 if is_meta else 1.0
        height  = 0.50 if is_meta else 0.58

        ax.barh(y, w, height=height, color=color, alpha=alpha,
                edgecolor="none", zorder=2)

        # Value label
        rate = w / n_total
        label_color = DIM if is_meta else MODEL_COLOR.get(model, MUTED)
        ax.text(w + max_w * 0.012, y,
                f"{w}   {rate:.1%}",
                va="center", ha="left", fontsize=9.0,
                color=label_color,
                fontweight="normal" if is_meta else "bold")

    # Separator line between meta and model rows
    sep_y = len(meta_rows) + GAP / 2
    ax.axhline(sep_y, color=SEP, linewidth=1.0, linestyle="--", zorder=1)

    # Y-tick labels
    ax.set_yticks(all_y)
    labels = meta_rows["model"].tolist() + model_rows["model"].tolist()
    ax.set_yticklabels(labels, fontsize=10.0)
    for tick, model in zip(ax.get_yticklabels(), labels):
        is_meta = model in ("none", "tie")
        tick.set_color(DIM if is_meta else MODEL_COLOR.get(model, TEXT))
        tick.set_fontweight("normal" if is_meta else
                            ("bold" if model in STRONG_MODELS else "normal"))

    ax.tick_params(axis="both", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_w * 1.28)

    # ── Headline callout: 39 % floor ─────────────────────────────────────
    n_none = int(win_df.loc[win_df["model"] == "none", "wins"].sum()) \
        if "none" in win_df["model"].values else 0
    n_tie  = int(win_df.loc[win_df["model"] == "tie",  "wins"].sum()) \
        if "tie"  in win_df["model"].values else 0
    floor_pct = n_none / n_total

    ax.text(max_w * 1.27, sep_y - 0.05,
            f"{floor_pct:.0%} of users: every model scored 0\n"
            f"A further {n_tie / n_total:.0%} had two+ models tied",
            va="center", ha="right", fontsize=8.2, color="#f87171",
            style="italic", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#2a0a0a",
                      edgecolor="#7f1d1d", linewidth=0.8))

    ax.set_xlabel("Number of test users  (out of 943)", fontsize=9.5,
                  color=MUTED, labelpad=7)
    ax.set_title("Win / Loss  —  which model ranks a relevant item highest per user?",
                 fontsize=11.5, fontweight="bold", color=TEXT, pad=10)

    # ── Shared colour scale legend for heatmaps ───────────────────────────
    # (shown here as a compact inline note rather than a redundant colourbar)
    ax.text(0.0, 1.06,
            "Heatmap colour scale: 0.00",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=DIM)
    # Draw a tiny gradient rectangle
    grad_ax = ax.inset_axes([0.133, 1.04, 0.12, 0.025])
    grad_data = np.linspace(0, 1, 256).reshape(1, -1)
    cinema_gold = mcolors.LinearSegmentedColormap.from_list(
        "cinema_gold", ["#16161f", "#c9a01a", "#F5C518"]
    )
    grad_ax.imshow(grad_data, aspect="auto", cmap=cinema_gold,
                   vmin=0, vmax=1, origin="lower")
    grad_ax.set_axis_off()
    ax.text(0.255, 1.06, "→ max NDCG@10",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=DIM)


# ── Main entry point ──────────────────────────────────────────────────────────

def plot_breakdown(
    pop_df:      pd.DataFrame,
    activity_df: pd.DataFrame,
    win_df:      pd.DataFrame,
    user_df:     pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    plt.rcParams.update({
        "text.color":          TEXT,
        "axes.labelcolor":     MUTED,
        "xtick.color":         MUTED,
        "ytick.color":         TEXT,
    })

    fig = plt.figure(figsize=(15, 11.5), facecolor=BG)

    # ── Global title ──────────────────────────────────────────────────────
    fig.text(0.5, 0.985,
             "Breakdown Analysis  —  Where do models win and fail?",
             ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.957,
             "MovieLens 100K  ·  NDCG@10  ·  Models sorted best → worst  "
             "·  Cell subscript = users evaluated  "
             "·  ▲ = NCF scores 0 (head-only recommender)",
             ha="center", va="top",
             fontsize=8.8, color=MUTED)

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.55, 0.85],
        left=0.10, right=0.97,
        top=0.945, bottom=0.06,
        hspace=0.42,
        wspace=0.04,   # heatmaps visually coupled
    )

    ax_pop = fig.add_subplot(gs[0, 0])
    ax_act = fig.add_subplot(gs[0, 1])
    ax_win = fig.add_subplot(gs[1, :])

    # ── Shared vmax ───────────────────────────────────────────────────────
    vmax = max(pop_df["ndcg_at_10"].max(),
               activity_df["ndcg_at_10"].max()) * 1.05

    # ── Panel 1: popularity buckets ───────────────────────────────────────
    models = _ordered_models(pop_df)
    pv = pop_df.pivot(index="model", columns="bucket", values="ndcg_at_10") \
               .reindex(index=models, columns=["head", "torso", "tail"]).fillna(0)
    pn = pop_df.pivot(index="model", columns="bucket", values="n_users") \
               .reindex(index=models, columns=["head", "torso", "tail"]).fillna(0).astype(int)

    _draw_heatmap(
        ax_pop, pv, pn,
        col_labels=[BUCKET_LABELS[c] for c in ["head", "torso", "tail"]],
        title="NDCG@10 by Item Popularity",
        vmax=vmax,
        show_yticklabels=True,
        ncf_flag_cols=["torso", "tail"],
    )

    # ── Panel 2: user activity tiers ─────────────────────────────────────
    models = _ordered_models(activity_df)
    av = activity_df.pivot(index="model", columns="tier", values="ndcg_at_10") \
                    .reindex(index=models, columns=["cold", "moderate", "active"]).fillna(0)
    an = activity_df.pivot(index="model", columns="tier", values="n_users") \
                    .reindex(index=models, columns=["cold", "moderate", "active"]).fillna(0).astype(int)

    _draw_heatmap(
        ax_act, av, an,
        col_labels=[TIER_LABELS[c] for c in ["cold", "moderate", "active"]],
        title="NDCG@10 by User Activity",
        vmax=vmax,
        show_yticklabels=False,   # shared with left panel
    )

    # Tier labels between the two heatmaps (right edge of left panel)
    for label, y, color in [
        ("STRONG", 0.5,  "#4F46E5"),
        ("MID",    3.5,  MUTED),
        ("WEAK",   7.25, DIM),
    ]:
        ax_pop.text(
            3.58, y, label,
            ha="left", va="center",
            fontsize=6.5, color=color,
            fontweight="bold", alpha=0.65,
            transform=ax_pop.transData,
        )

    # ── Panel 3: win / loss ───────────────────────────────────────────────
    _draw_win_bar(ax_win, win_df, n_total=len(user_df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {output_path}")
