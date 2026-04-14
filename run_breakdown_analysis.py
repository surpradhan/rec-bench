"""
Week 4 — Item 3: Error / breakdown analysis.

Fits all 8 models on train+val (same setup as run_final_benchmark.py),
collects per-user recommendations, then runs three analyses:

  1. Per-popularity-bucket  — NDCG@10 for head / torso / tail items
  2. Per-user-activity      — NDCG@10 for cold / moderate / active users
  3. Win / loss             — which model ranks best per test user?

Outputs
-------
  results/analysis/popularity_bucket.csv
  results/analysis/user_activity.csv
  results/analysis/win_loss_summary.csv
  results/analysis/win_loss_per_user.csv
  results/plots/breakdown_overview.png
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import pandas as pd
import numpy as np

from src.data_prep.loader import load_and_split, build_interaction_matrix
from src.models import (
    PopularityRecommender, ContentBasedRecommender,
    UserCFRecommender, ItemCFRecommender,
    SVDRecommender, ALSRecommender, NCFRecommender, HybridRecommender,
)
from src.evaluation.evaluator import build_ground_truth
from src.tuning.tuner import load_best_params
from src.analysis.breakdown import (
    bucket_items_by_popularity,
    bucket_users_by_activity,
    popularity_bucket_breakdown,
    user_activity_breakdown,
    win_loss_breakdown,
)
from src.visualization.breakdown_plots import plot_breakdown

ANALYSIS_DIR = Path("results/analysis")
K = 10


# ── Model setup (mirrors run_final_benchmark.py) ──────────────────────────────

def _build_models(best: dict) -> list:
    def tuned(key, cls):
        params = best.get(key, {}).get("params", {})
        if params:
            print(f"    {key}: tuned params {params}")
        else:
            print(f"    {key}: defaults (no tuning results found)")
        return cls(**params, random_state=42)

    return [
        PopularityRecommender(),
        ContentBasedRecommender(),
        UserCFRecommender(),
        ItemCFRecommender(),
        tuned("svd",    SVDRecommender),
        tuned("als",    ALSRecommender),
        tuned("ncf",    NCFRecommender),
        tuned("hybrid", HybridRecommender),
    ]


# ── Collect recommendations ───────────────────────────────────────────────────

def collect_recs(model, test_users: list[int], max_k: int = K) -> dict[int, list[int]]:
    recs = {}
    for uid in test_users:
        raw = model.recommend(uid, n=max_k, exclude_seen=True)
        recs[uid] = [int(x) for x in raw]
    return recs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Week 4 · Item 3 — Breakdown Analysis")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    data = load_and_split()
    train_val = pd.concat([data["train"], data["val"]], ignore_index=True)
    assert not train_val.duplicated(subset=["user_id", "item_id"]).any()

    combined_matrix, _, _ = build_interaction_matrix(
        train_val,
        user_ids=data["user_ids"],
        item_ids=data["item_ids"],
    )
    fit_kwargs = {
        "movies":         data["movies"],
        "train_matrix":   combined_matrix,
        "user_id_to_idx": data["user_id_to_idx"],
        "item_id_to_idx": data["item_id_to_idx"],
    }

    test_users: list[int] = [int(u) for u in data["test"]["user_id"].unique()]
    ground_truth = build_ground_truth(data["test"])

    # ── 2. Build bucket / tier mappings ──────────────────────────────────
    print("\n[2/5] Building item popularity buckets and user activity tiers …")
    # Use train_val (same data models trained on) for counting
    item_buckets = bucket_items_by_popularity(train_val)
    user_tiers   = bucket_users_by_activity(train_val)

    bucket_counts = pd.Series(item_buckets).value_counts()
    tier_counts   = pd.Series(user_tiers).value_counts()
    print(f"  Item buckets : {dict(bucket_counts)}")
    print(f"  User tiers   : {dict(tier_counts)}")

    # ── 3. Fit models + collect recommendations ───────────────────────────
    print("\n[3/5] Fitting models on train+val and collecting recommendations …")
    best = load_best_params()
    models = _build_models(best)

    recs_by_model: dict[str, dict[int, list[int]]] = {}
    for model in models:
        print(f"  → {model.name} …", end=" ", flush=True)
        model.fit(train_val, **fit_kwargs)
        recs_by_model[model.name] = collect_recs(model, test_users)
        print("done")

    # ── 4. Run analyses ───────────────────────────────────────────────────
    print("\n[4/5] Running analyses …")

    pop_df      = popularity_bucket_breakdown(recs_by_model, ground_truth, item_buckets)
    activity_df = user_activity_breakdown(recs_by_model, ground_truth, user_tiers)
    user_df, win_df = win_loss_breakdown(recs_by_model, ground_truth)

    # ── 5. Save outputs ───────────────────────────────────────────────────
    print("\n[5/5] Saving outputs …")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    pop_df.to_csv(ANALYSIS_DIR / "popularity_bucket.csv", index=False)
    activity_df.to_csv(ANALYSIS_DIR / "user_activity.csv", index=False)
    win_df.to_csv(ANALYSIS_DIR / "win_loss_summary.csv", index=False)
    user_df.to_csv(ANALYSIS_DIR / "win_loss_per_user.csv", index=False)

    plot_breakdown(pop_df, activity_df, win_df, user_df)

    # ── Print summaries ───────────────────────────────────────────────────
    print("\n── Item popularity bucket breakdown (NDCG@10) ──────────────────")
    pivot_pop = pop_df.pivot(index="model", columns="bucket", values="ndcg_at_10")
    pivot_pop = pivot_pop.reindex(
        index=[m for m in ["Hybrid","ALS","UserCF","ItemCF","Popularity","SVD","NCF","ContentBased"]
               if m in pivot_pop.index],
        columns=["head", "torso", "tail"]
    )
    print(pivot_pop.round(4).to_string())

    print("\n── User activity tier breakdown (NDCG@10) ──────────────────────")
    pivot_act = activity_df.pivot(index="model", columns="tier", values="ndcg_at_10")
    pivot_act = pivot_act.reindex(
        index=[m for m in ["Hybrid","ALS","UserCF","ItemCF","Popularity","SVD","NCF","ContentBased"]
               if m in pivot_act.index],
        columns=["cold", "moderate", "active"]
    )
    print(pivot_act.round(4).to_string())

    print("\n── Win / loss summary ──────────────────────────────────────────")
    print(win_df.sort_values("wins", ascending=False).to_string(index=False))

    n_total = len(user_df)
    n_none = int(win_df.loc[win_df["model"] == "tie/none", "wins"].sum()) \
        if "tie/none" in win_df["model"].values else 0
    print(f"\n  Total test users : {n_total}")
    print(f"  No model won     : {n_none}  ({n_none/n_total:.1%})")

    print("\nDone. All outputs saved to results/analysis/ and results/plots/")


if __name__ == "__main__":
    main()
