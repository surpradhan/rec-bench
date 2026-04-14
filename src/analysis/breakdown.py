"""
Post-hoc breakdown analysis for the recommendation benchmark.

Three analyses:
  1. Per-popularity-bucket  — NDCG@10 split by item head / torso / tail
  2. Per-user-activity      — NDCG@10 split by user cold / moderate / active
  3. Win / loss             — per-user best model; Hybrid/ALS vs UserCF deep-dive

All functions are pure (no side effects, no file I/O) and operate on
pre-collected recommendation dicts so models are fitted only once.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import ndcg_at_k

RELEVANCE_THRESHOLD = 4.0
K = 10

# ── Bucket / tier definitions ─────────────────────────────────────────────────
# Items: top 20 % by train rating count = head, bottom 50 % = tail, rest = torso
ITEM_HEAD_PCT = 0.20
ITEM_TAIL_PCT = 0.50

# Users: 33rd / 66th percentile of train rating count
USER_TIER_QUANTILES = (1 / 3, 2 / 3)

ITEM_BUCKETS  = ["head", "torso", "tail"]
USER_TIERS    = ["cold", "moderate", "active"]
MODEL_ORDER   = ["Hybrid", "ALS", "UserCF", "ItemCF", "Popularity", "SVD", "NCF", "ContentBased"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _relevant(raw_gt: dict[int, float]) -> set[int]:
    return {iid for iid, r in raw_gt.items() if r >= RELEVANCE_THRESHOLD}


# ── Analysis 1: item popularity buckets ──────────────────────────────────────

def bucket_items_by_popularity(train: pd.DataFrame) -> dict[int, str]:
    """
    Assign each item to 'head', 'torso', or 'tail' based on its rating count
    in the training set.

    head  = top ITEM_HEAD_PCT (20 %) most-rated items
    tail  = bottom ITEM_TAIL_PCT (50 %) least-rated items
    torso = everything in between
    """
    counts = train["item_id"].value_counts().sort_values(ascending=False)
    n = len(counts)
    n_head = max(1, int(np.ceil(n * ITEM_HEAD_PCT)))
    n_tail = max(1, int(np.ceil(n * ITEM_TAIL_PCT)))

    buckets: dict[int, str] = {}
    for rank, item_id in enumerate(counts.index):
        if rank < n_head:
            buckets[int(item_id)] = "head"
        elif rank >= n - n_tail:
            buckets[int(item_id)] = "tail"
        else:
            buckets[int(item_id)] = "torso"
    return buckets


def popularity_bucket_breakdown(
    recs_by_model: dict[str, dict[int, list[int]]],
    ground_truth: dict[int, dict[int, float]],
    item_buckets: dict[int, str],
) -> pd.DataFrame:
    """
    For each (model, bucket) pair, compute mean NDCG@10 restricted to relevant
    items that fall in that bucket.  Users with no relevant items in a bucket
    are excluded from that bucket's mean (not scored as 0).

    Returns a DataFrame with columns: model, bucket, ndcg_at_10, n_users.
    """
    rows = []
    for model_name, recs in recs_by_model.items():
        for bucket in ITEM_BUCKETS:
            scores = []
            for uid, raw_gt in ground_truth.items():
                if uid not in recs:
                    continue
                bucket_relevant = {
                    iid for iid, r in raw_gt.items()
                    if r >= RELEVANCE_THRESHOLD and item_buckets.get(int(iid)) == bucket
                }
                if not bucket_relevant:
                    continue
                scores.append(ndcg_at_k(recs[uid], bucket_relevant, K))
            rows.append({
                "model":       model_name,
                "bucket":      bucket,
                "ndcg_at_10":  float(np.mean(scores)) if scores else 0.0,
                "n_users":     len(scores),
            })
    return pd.DataFrame(rows)


# ── Analysis 2: user activity tiers ──────────────────────────────────────────

def bucket_users_by_activity(train: pd.DataFrame) -> dict[int, str]:
    """
    Assign each user to 'cold', 'moderate', or 'active' based on their rating
    count in the provided DataFrame (caller passes train+val), using the 33rd
    and 66th percentile boundaries.
    """
    counts = train.groupby("user_id").size()
    q33, q66 = counts.quantile(USER_TIER_QUANTILES).values

    tiers: dict[int, str] = {}
    for uid, cnt in counts.items():
        if cnt <= q33:
            tiers[int(uid)] = "cold"
        elif cnt <= q66:
            tiers[int(uid)] = "moderate"
        else:
            tiers[int(uid)] = "active"
    return tiers


def user_activity_breakdown(
    recs_by_model: dict[str, dict[int, list[int]]],
    ground_truth: dict[int, dict[int, float]],
    user_tiers: dict[int, str],
) -> pd.DataFrame:
    """
    For each (model, tier) pair, compute mean NDCG@10.
    Users not in user_tiers (e.g. pure-train users) are excluded.

    Returns a DataFrame with columns: model, tier, ndcg_at_10, n_users.
    """
    tier_to_users: dict[str, set[int]] = {t: set() for t in USER_TIERS}
    for uid, tier in user_tiers.items():
        tier_to_users[tier].add(uid)

    rows = []
    for model_name, recs in recs_by_model.items():
        for tier in USER_TIERS:
            scores = []
            for uid in tier_to_users[tier]:
                if uid not in recs or uid not in ground_truth:
                    continue
                relevant = _relevant(ground_truth[uid])
                scores.append(ndcg_at_k(recs[uid], relevant, K))
            rows.append({
                "model":      model_name,
                "tier":       tier,
                "ndcg_at_10": float(np.mean(scores)) if scores else 0.0,
                "n_users":    len(scores),
            })
    return pd.DataFrame(rows)


# ── Analysis 3: win / loss ────────────────────────────────────────────────────

def win_loss_breakdown(
    recs_by_model: dict[str, dict[int, list[int]]],
    ground_truth: dict[int, dict[int, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each test user, determine which model achieves the highest NDCG@10.

    winner labels:
      <model_name> — one model strictly leads
      'tie'        — two or more models share the best non-zero score
      'none'       — every model scored 0 (no relevant item in top-K)

    Returns
    -------
    user_df   : one row per user — winner + per-model NDCG@10 scores
    win_df    : win count and win rate per model, sorted by wins descending
    """
    model_names = list(recs_by_model.keys())
    all_users   = sorted(ground_truth.keys())

    rows = []
    for uid in all_users:
        raw_gt   = ground_truth[uid]
        relevant = _relevant(raw_gt)

        scores: dict[str, float] = {
            m: ndcg_at_k(recs_by_model[m][uid], relevant, K)
            if uid in recs_by_model[m] else 0.0
            for m in model_names
        }

        best = max(scores.values())
        if best == 0.0:
            winner = "none"
        else:
            best_models = [m for m in model_names if scores[m] == best]
            winner = best_models[0] if len(best_models) == 1 else "tie"

        row = {"user_id": uid, "winner": winner, "best_ndcg": best}
        row.update({f"ndcg_{m}": scores[m] for m in model_names})
        rows.append(row)

    user_df = pd.DataFrame(rows)

    n_total = len(user_df)
    win_counts = user_df["winner"].value_counts()
    win_df = pd.DataFrame({
        "model":    win_counts.index,
        "wins":     win_counts.values,
        "win_rate": win_counts.values / n_total,
    }).reset_index(drop=True)

    return user_df, win_df
