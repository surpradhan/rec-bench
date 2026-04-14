"""
Evaluator — runs any BaseRecommender through the full benchmark protocol.

Measures:
  - Ranking metrics (Precision, Recall, NDCG, MAP at K=5 and K=10)
  - Rating prediction (RMSE, MAE) — only for models that declare support
  - Training time
  - Inference latency (per-user median and p95)
  - Peak memory usage

Design guarantees
-----------------
- Each model runs inside a try/except; a failed model records the error and
  the benchmark continues with the remaining models.
- model.recommend() output is validated before metrics are computed.
- All user ids are normalised to Python int before being passed to any model
  method, so models that use ids as dict keys never see numpy int64.
- Models that do not support rating prediction are not queried for it;
  their RMSE/MAE columns read "N/A" instead of NaN to distinguish from a
  model that attempted prediction but failed.
- Results are persisted to CSV after every model so progress is not lost.
"""

import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from .metrics import evaluate_ranking, rmse, mae
from src.models.base import BaseRecommender

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Minimum number of latency samples for p95 to be meaningful.
_MIN_LATENCY_SAMPLES_FOR_P95 = 20


def build_ground_truth(test: pd.DataFrame) -> dict[int, dict[int, float]]:
    """
    Build {user_id: {item_id: rating}} from the test split.

    Raises ValueError if duplicate (user_id, item_id) pairs are present —
    a duplicate would silently overwrite the first rating, corrupting
    RMSE/MAE ground truth with whichever row happened to appear last.
    """
    dupes = test.duplicated(subset=["user_id", "item_id"])
    if dupes.any():
        n = int(dupes.sum())
        raise ValueError(
            f"Test set contains {n} duplicate (user_id, item_id) pair(s). "
            f"Deduplicate before evaluation to avoid corrupted RMSE/MAE ground truth."
        )
    gt: dict[int, dict[int, float]] = {}
    for row in test.itertuples(index=False):
        gt.setdefault(int(row.user_id), {})[int(row.item_id)] = row.rating
    return gt


def _validate_recommendations(
    recs: object, model_name: str, user_id: int, max_k: int
) -> list[int]:
    """
    Assert that model.recommend() returned a valid ranked list.
    Raises ValueError with a descriptive message on any violation.
    """
    if not isinstance(recs, list):
        raise ValueError(
            f"[{model_name}] recommend({user_id}) returned {type(recs).__name__}, expected list"
        )
    if len(recs) > max_k:
        raise ValueError(
            f"[{model_name}] recommend({user_id}) returned {len(recs)} items, expected <= {max_k}"
        )
    non_int = [x for x in recs if not isinstance(x, (int, np.integer))]
    if non_int:
        raise ValueError(
            f"[{model_name}] recommend({user_id}) contains non-integer items: {non_int[:5]}"
        )
    return [int(x) for x in recs]


def evaluate(
    model: BaseRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_values: list[int] = (5, 10),
    relevance_threshold: float = 4.0,
    n_latency_samples: int = 500,
    random_state: int = 42,
    fit_kwargs: dict | None = None,
) -> dict:
    """
    Fit the model on `train`, evaluate on `test`.

    Parameters
    ----------
    fit_kwargs : extra keyword arguments forwarded to model.fit().
                 Use this to pass movies metadata, pre-built matrices, or
                 index dicts that specific models require:

                     evaluate(model, train, test,
                              fit_kwargs={"movies": data["movies"],
                                          "train_matrix": data["train_matrix"],
                                          "user_id_to_idx": data["user_id_to_idx"],
                                          "item_id_to_idx": data["item_id_to_idx"]})

    Returns a results dict suitable for a single row in the comparison table.
    On failure the dict contains an "error" key and zeroed/null metrics.
    """
    fit_kwargs = fit_kwargs or {}
    # ── Input validation ──────────────────────────────────────────────────
    if not k_values or not all(isinstance(k, int) and k > 0 for k in k_values):
        raise ValueError(
            f"k_values must be a non-empty list of positive integers, got {k_values!r}"
        )

    results: dict = {"model": model.name}
    max_k = max(k_values)

    # Normalise all test user ids to plain Python int once, upfront.
    # This ensures every model.recommend() call receives the same type
    # regardless of whether pandas returns int64 or int32 on the platform.
    test_users: list[int] = [int(u) for u in test["user_id"].unique()]

    # ── Training ──────────────────────────────────────────────────────────
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        model.fit(train, **fit_kwargs)
        train_time = time.perf_counter() - t0
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except Exception as exc:
        tracemalloc.stop()
        results["error"] = f"fit() failed: {exc}"
        return results

    results["train_time_s"] = round(train_time, 3)
    results["peak_memory_mb"] = round(mem_peak / 1024 / 1024, 2)

    # ── Ranking evaluation ────────────────────────────────────────────────
    try:
        ground_truth = build_ground_truth(test)

        recommendations: dict[int, list[int]] = {}
        for uid in test_users:
            raw = model.recommend(uid, n=max_k, exclude_seen=True)
            recommendations[uid] = _validate_recommendations(raw, model.name, uid, max_k)

        ranking_metrics = evaluate_ranking(
            recommendations, ground_truth,
            k_values=list(k_values),
            relevance_threshold=relevance_threshold,
        )
        results.update(ranking_metrics)
    except Exception as exc:
        results["error"] = f"recommend() failed: {exc}"
        return results

    # ── Inference latency (sampled) ───────────────────────────────────────
    rng = np.random.default_rng(random_state)
    n_sample = min(n_latency_samples, len(test_users))
    sample_users: list[int] = [
        int(u) for u in rng.choice(test_users, size=n_sample, replace=False)
    ]
    latencies: list[float] = []
    for uid in sample_users:
        t = time.perf_counter()
        model.recommend(uid, n=max_k, exclude_seen=True)
        latencies.append(time.perf_counter() - t)

    results["latency_median_ms"] = round(float(np.median(latencies)) * 1000, 3)

    if len(latencies) < _MIN_LATENCY_SAMPLES_FOR_P95:
        warnings.warn(
            f"[{model.name}] p95 latency estimated from only {len(latencies)} samples "
            f"(< {_MIN_LATENCY_SAMPLES_FOR_P95}); estimate may be unreliable."
        )
    results["latency_p95_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 3)

    # ── Rating prediction (only if model declares support) ────────────────
    if model.supports_rating_prediction:
        try:
            y_true, y_pred = [], []
            for row in test.itertuples(index=False):
                y_true.append(row.rating)
                y_pred.append(model.predict_rating(int(row.user_id), int(row.item_id)))
            y_true_arr = np.array(y_true, dtype=float)
            y_pred_arr = np.array(y_pred, dtype=float)
            results["RMSE"] = round(rmse(y_true_arr, y_pred_arr, model.name), 4)
            results["MAE"] = round(mae(y_true_arr, y_pred_arr, model.name), 4)
        except Exception as exc:
            results["RMSE"] = "N/A"
            results["MAE"] = "N/A"
            warnings.warn(f"[{model.name}] predict_rating() failed: {exc}")
    else:
        results["RMSE"] = "N/A"
        results["MAE"] = "N/A"

    return results


def benchmark(
    models: list[BaseRecommender],
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_values: list[int] = (5, 10),
    relevance_threshold: float = 4.0,
    n_latency_samples: int = 500,
    random_state: int = 42,
    fit_kwargs: dict | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Run evaluate() for every model in `models` and return a comparison DataFrame.

    Results are saved to CSV after every model so progress survives a crash.

    Parameters
    ----------
    models      : list of BaseRecommender instances (unfitted)
    fit_kwargs  : extra keyword arguments forwarded to every model's fit().
                  Pass movies, matrices, and index dicts here so all models
                  have access without modifying the base interface.
    save_path   : where to write the running CSV; defaults to results/benchmark_results.csv
    """
    if save_path is None:
        save_path = RESULTS_DIR / "benchmark_results.csv"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in models:
        print(f"  Evaluating {model.name} ...", flush=True)
        row = evaluate(
            model, train, test,
            k_values=k_values,
            relevance_threshold=relevance_threshold,
            n_latency_samples=n_latency_samples,
            random_state=random_state,
            fit_kwargs=fit_kwargs,
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(save_path, index=False)
        if "error" in row:
            warnings.warn(f"Model '{model.name}' failed: {row['error']}")
        else:
            p10 = row.get("Precision@10")
            ndcg10 = row.get("NDCG@10")
            metric_str = (
                f"P@10={p10:.4f} | NDCG@10={ndcg10:.4f}"
                if isinstance(p10, float) and isinstance(ndcg10, float)
                else "(ranking metrics unavailable)"
            )
            print(f"    done. Train {row['train_time_s']}s | {metric_str}")

    df = pd.DataFrame(rows)
    print(f"\nResults saved to {save_path}")
    return df
