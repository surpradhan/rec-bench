"""
Week 4 — Item 1: Final benchmark with tuned models retrained on train+val.

Tuned Week 3 models were selected on val but trained only on train.
Standard practice is to retrain on train+val combined using best params before
the final test evaluation, so the final model has seen more data.

Steps:
  1. Load data (train / val / test splits + full id universe).
  2. Combine train+val into a single DataFrame.
  3. Rebuild the interaction matrix from the combined split, keeping the same
     user_ids / item_ids universe so no index mismatches occur.
  4. Instantiate all 8 models:
       - Week 1 & 2: default hyperparameters (not tuned).
       - Week 3 (SVD, ALS, NCF, Hybrid): best params from results/tuning/best_params.json.
  5. Run benchmark() against the held-out test split.
  6. Save results to results/benchmark_results.csv (overwriting the old file).
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import pandas as pd

from src.data_prep.loader import load_and_split, build_interaction_matrix
from src.models import (
    PopularityRecommender,
    ContentBasedRecommender,
    UserCFRecommender,
    ItemCFRecommender,
    SVDRecommender,
    ALSRecommender,
    NCFRecommender,
    HybridRecommender,
)
from src.evaluation.evaluator import benchmark
from src.tuning.tuner import load_best_params


def main() -> None:
    print("Loading data …")
    data = load_and_split()

    # ── Combine train + val ───────────────────────────────────────────────
    train_val = pd.concat([data["train"], data["val"]], ignore_index=True)
    print(
        f"  train rows : {len(data['train']):,}\n"
        f"  val rows   : {len(data['val']):,}\n"
        f"  combined   : {len(train_val):,}"
    )

    # Sanity check: time-based splitting must never assign the same (user, item)
    # pair to both train and val. If this fires, split_data() has a bug.
    assert not train_val.duplicated(subset=["user_id", "item_id"]).any(), (
        "train and val share (user_id, item_id) pairs — split_data() invariant violated"
    )

    # ── Rebuild interaction matrix for train+val ──────────────────────────
    # Re-use the shared universe (all_user_ids / all_item_ids) so every split
    # and every model works in the same row/column index space.
    combined_matrix, _, _ = build_interaction_matrix(
        train_val,
        user_ids=data["user_ids"],
        item_ids=data["item_ids"],
    )
    print(f"  combined matrix shape: {combined_matrix.shape}")

    # ── Load tuned params for Week 3 models ──────────────────────────────
    best = load_best_params()
    if not best:
        print(
            "\nWARNING: no tuning results found at results/tuning/best_params.json.\n"
            "Week 3 models will use default hyperparameters.\n"
            "Run tuning first with src/tuning/tuner.py to get best params.\n"
        )

    def tuned(key: str, cls: type) -> object:
        """Instantiate a model with its tuned params (or defaults if absent).

        random_state=42 is passed explicitly to match the fixed seed used
        during tuning (tune_model() always forces random_state=42 and does not
        include it in best_params.json). Without this, a future change to any
        model's default random_state would silently diverge from tuning.
        """
        params = best.get(key, {}).get("params", {})
        if params:
            print(f"  {key}: using tuned params {params}")
        else:
            print(f"  {key}: no tuned params found, using defaults")
        return cls(**params, random_state=42)

    print("\nInstantiating models …")
    models = [
        # Week 1 & 2 — no hyperparameter tuning
        PopularityRecommender(),
        ContentBasedRecommender(),
        UserCFRecommender(),
        ItemCFRecommender(),
        # Week 3 — retrained with tuned params on train+val
        tuned("svd",    SVDRecommender),
        tuned("als",    ALSRecommender),
        tuned("ncf",    NCFRecommender),
        tuned("hybrid", HybridRecommender),
    ]

    # ── fit_kwargs for the combined train ─────────────────────────────────
    # The key name "train_matrix" is what all Week 3 model fit() signatures
    # expect — but note it holds the combined train+val matrix here, not
    # train-only. Models use whatever matrix is passed under this key.
    fit_kwargs = {
        "movies":          data["movies"],
        "train_matrix":    combined_matrix,   # train+val combined
        "user_id_to_idx":  data["user_id_to_idx"],
        "item_id_to_idx":  data["item_id_to_idx"],
    }

    # Load previous results for delta comparison (before overwriting).
    save_path = Path("results/benchmark_results.csv")
    prev_df = pd.read_csv(save_path) if save_path.exists() else None

    print("\nRunning final benchmark (train+val → test) …\n")
    results_df = benchmark(
        models,
        train=train_val,
        test=data["test"],
        fit_kwargs=fit_kwargs,
    )

    print("\nFinal results:")
    cols = ["model", "NDCG@5", "NDCG@10", "Precision@5", "Precision@10",
            "Recall@5", "Recall@10", "MAP@5", "MAP@10", "RMSE", "MAE",
            "train_time_s"]
    display_cols = [c for c in cols if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))

    # ── Delta vs previous run ─────────────────────────────────────────────
    if prev_df is not None and "NDCG@10" in prev_df.columns:
        print("\nNDCG@10 delta vs previous run (train-only):")
        print(f"  {'Model':<14} {'Previous':>10} {'Final':>10} {'Delta':>10}")
        print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
        prev_map = dict(zip(prev_df["model"], pd.to_numeric(prev_df["NDCG@10"], errors="coerce")))
        for _, row in results_df.iterrows():
            model_name = row["model"]
            new_val = pd.to_numeric(row.get("NDCG@10"), errors="coerce")
            old_val = prev_map.get(model_name)
            if pd.isna(new_val):
                continue
            if old_val is not None and not pd.isna(old_val):
                delta = new_val - old_val
                sign = "+" if delta >= 0 else ""
                print(f"  {model_name:<14} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f}")
            else:
                print(f"  {model_name:<14} {'N/A':>10} {new_val:>10.4f} {'N/A':>10}")


if __name__ == "__main__":
    main()
