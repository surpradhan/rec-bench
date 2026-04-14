"""
Ranking and rating-prediction evaluation metrics.

All ranking metrics operate on:
  recommended : list of item_ids (ranked, best first)
  relevant    : set of item_ids the user actually interacted with in the test set

Supported metrics
-----------------
  Precision@K, Recall@K, NDCG@K, MAP@K   — ranking quality
  RMSE, MAE                               — explicit rating prediction
"""

import warnings
import numpy as np

# Expected rating scale for MovieLens. Models that output predictions outside
# this range are likely mis-scaled; RMSE/MAE would be meaningless.
_RATING_MIN = 1.0
_RATING_MAX = 5.0


# ──────────────────────────────────────────────
# Per-user ranking metrics
# ──────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items that appear in the top-K."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Binary relevance (item is relevant or not).
    """
    if not relevant or k == 0:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Average Precision at K for a single user.

    Denominator convention: min(|relevant|, K).
    This matches the standard IR definition where AP@K rewards both precision
    and coverage up to the cutoff.  If you prefer to divide by |relevant|
    alone (ignoring K), swap the denominator below.
    """
    if not relevant or k == 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant), k)


# ──────────────────────────────────────────────
# Aggregate over all users
# ──────────────────────────────────────────────

def evaluate_ranking(
    recommendations: dict[int, list],
    ground_truth: dict[int, set],
    k_values: list[int] = (5, 10),
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Compute mean Precision, Recall, NDCG, and MAP over all users.

    Parameters
    ----------
    recommendations : {user_id: [item_id, ...]}  ranked lists
    ground_truth    : {user_id: set(item_id)}     relevant items per user
                      If values are floats (raw ratings), items with
                      rating >= relevance_threshold are treated as relevant.
    k_values        : list of K cutoffs to evaluate
    relevance_threshold : minimum rating to count as relevant (for raw dicts)

    Returns
    -------
    dict with keys like "Precision@5", "Recall@10", etc.
    """
    results = {f"Precision@{k}": [] for k in k_values}
    results.update({f"Recall@{k}": [] for k in k_values})
    results.update({f"NDCG@{k}": [] for k in k_values})
    results.update({f"MAP@{k}": [] for k in k_values})

    common_users = set(recommendations.keys()) & set(ground_truth.keys())

    for user_id in common_users:
        recs = recommendations[user_id]
        raw = ground_truth[user_id]

        # Support both set[int] and dict{item_id: rating}
        if isinstance(raw, set):
            relevant = raw
        else:
            relevant = {iid for iid, r in raw.items() if r >= relevance_threshold}

        # Users with no relevant items in the test set score 0 on all metrics.
        # Skipping them would inflate averages by excluding zero-score users.
        for k in k_values:
            results[f"Precision@{k}"].append(precision_at_k(recs, relevant, k))
            results[f"Recall@{k}"].append(recall_at_k(recs, relevant, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recs, relevant, k))
            results[f"MAP@{k}"].append(average_precision_at_k(recs, relevant, k))

    return {metric: float(np.mean(vals)) if vals else 0.0
            for metric, vals in results.items()}


# ──────────────────────────────────────────────
# Rating prediction metrics
# ──────────────────────────────────────────────

def _check_prediction_scale(y_pred: np.ndarray, model_name: str = "") -> None:
    """
    Warn if predicted ratings fall outside the expected 1–5 scale.
    A model outputting 0–1 probabilities or 0–100 scores would produce
    RMSE values that are incomparable to a correctly scaled model.
    """
    valid = y_pred[~np.isnan(y_pred)]
    if len(valid) == 0:
        return
    lo, hi = float(valid.min()), float(valid.max())
    tag = f"[{model_name}] " if model_name else ""
    if lo < _RATING_MIN - 0.5 or hi > _RATING_MAX + 0.5:
        warnings.warn(
            f"{tag}Predicted ratings are outside the expected {_RATING_MIN}–{_RATING_MAX} "
            f"scale (observed range: [{lo:.3f}, {hi:.3f}]). "
            f"RMSE/MAE may not be comparable to other models."
        )


def rmse(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> float:
    """Root Mean Squared Error, ignoring NaN predictions."""
    _check_prediction_scale(y_pred, model_name)
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask].astype(float) - y_pred[mask].astype(float)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> float:
    """Mean Absolute Error, ignoring NaN predictions."""
    _check_prediction_scale(y_pred, model_name)
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask].astype(float) - y_pred[mask].astype(float))))
