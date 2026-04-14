"""
Bayesian hyperparameter tuning for Week 3 recommenders.

Uses scikit-optimize's Gaussian Process minimiser to search each model's
parameter space, maximising NDCG@10 on the *validation* split.  The test
split is never touched during tuning — only at final benchmark time.

Usage
-----
# Tune all four Week 3 models with default call budgets:
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data_prep.loader import load_and_split
from src.tuning.tuner import tune_all
data = load_and_split()
tune_all(data['train'], data['val'],
         fit_kwargs={'movies': data['movies'],
                     'train_matrix': data['train_matrix'],
                     'user_id_to_idx': data['user_id_to_idx'],
                     'item_id_to_idx': data['item_id_to_idx']})
"

Results are saved to results/tuning/best_params.json.
A convergence plot is saved to results/plots/tuning_convergence.png.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluator import evaluate
from src.tuning.spaces import PARAM_SPACES

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def tune_model(
    model_class: type,
    param_space: list,
    train: pd.DataFrame,
    val: pd.DataFrame,
    fit_kwargs: dict | None = None,
    n_calls: int = 25,
    n_initial_points: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[dict, float, object]:
    """
    Bayesian search over `param_space` to maximise NDCG@10 on `val`.

    Parameters
    ----------
    model_class     : BaseRecommender subclass to instantiate and tune.
    param_space     : list of (param_name, skopt Dimension) pairs.
    train / val     : DataFrames from load_and_split().  Model is fit on
                      `train` and evaluated on `val` — test is never used.
    fit_kwargs      : extra keyword args forwarded to model.fit() (matrices,
                      id dicts, movies metadata).
    n_calls         : total number of (fit + evaluate) calls, including
                      n_initial_points random exploration calls.
    n_initial_points: number of random calls before GP takes over.
    random_state    : seed for both the GP surrogate and the model itself.
    verbose         : print each call's params and NDCG@10.

    Returns
    -------
    best_params : dict of param_name → best value
    best_ndcg   : NDCG@10 achieved by best_params on val
    skopt_result: raw OptimizeResult from gp_minimize (for plotting)
    """
    try:
        from skopt import gp_minimize
    except ImportError as exc:
        raise ImportError(
            "tune_model() requires scikit-optimize. "
            "Install it with: pip install scikit-optimize"
        ) from exc

    fit_kwargs = fit_kwargs or {}
    param_names = [name for name, _ in param_space]
    dimensions = [dim for _, dim in param_space]

    call_idx = [0]

    def objective(params: list) -> float:
        call_idx[0] += 1
        kwargs = dict(zip(param_names, params))

        # Fix random_state so stochastic variation doesn't corrupt the signal.
        model = model_class(**kwargs, random_state=random_state)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = evaluate(model, train, val, fit_kwargs=fit_kwargs)

        if "error" in result:
            ndcg = 0.0
            if verbose:
                print(
                    f"  [{call_idx[0]:2d}/{n_calls}] {kwargs} "
                    f"→ ERROR: {result['error']}"
                )
        else:
            ndcg = float(result.get("NDCG@10", 0.0))
            if verbose:
                print(
                    f"  [{call_idx[0]:2d}/{n_calls}] {kwargs} "
                    f"→ NDCG@10={ndcg:.4f}"
                )

        # gp_minimize minimises; negate NDCG so it maximises it.
        # A failed call returns 0.0 (the worst possible negated NDCG, since
        # all valid NDCGs are > 0, meaning -NDCG < 0 < 0.0).
        return -ndcg

    n_initial = min(n_initial_points, n_calls)
    skopt_result = gp_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=random_state,
        verbose=False,   # we handle our own logging above
    )

    # skopt returns numpy scalar types (int64, float64) for Integer/Real
    # dimensions. Convert to native Python so callers can pass them directly
    # to model __init__ and so json.dump serialises them without error.
    best_params = {
        k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v
        for k, v in zip(param_names, skopt_result.x)
    }
    best_ndcg = float(-skopt_result.fun)
    return best_params, best_ndcg, skopt_result


def tune_all(
    train: pd.DataFrame,
    val: pd.DataFrame,
    fit_kwargs: dict | None = None,
    n_calls_override: int | None = None,
    random_state: int = 42,
    save_dir: Path | None = None,
    models: list[str] | None = None,
) -> dict[str, dict]:
    """
    Tune Week 3 models and persist results.

    Parameters
    ----------
    n_calls_override : if set, overrides the per-model n_calls from spaces.py
                       (useful for quick smoke tests, e.g. n_calls_override=3).
    save_dir         : directory for best_params.json; defaults to results/tuning/
    models           : list of model keys to tune, e.g. ['als', 'hybrid'].
                       If None, all models in PARAM_SPACES are tuned.
                       Previously saved results for untouched models are preserved.

    Returns
    -------
    dict mapping model key → {'params': {...}, 'ndcg_at_10': float}
    """
    save_dir = Path(save_dir) if save_dir else RESULTS_DIR / "tuning"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results so untouched models are preserved in the output file.
    all_results: dict[str, dict] = load_best_params(save_dir)
    convergence: dict[str, list[float]] = {}

    keys_to_tune = models if models is not None else list(PARAM_SPACES.keys())
    unknown = [k for k in keys_to_tune if k not in PARAM_SPACES]
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}. Available: {list(PARAM_SPACES.keys())}")

    for key, (model_class, param_space, default_n_calls) in PARAM_SPACES.items():
        if key not in keys_to_tune:
            continue
        n_calls = n_calls_override if n_calls_override is not None else default_n_calls
        print(
            f"\n{'='*60}\n"
            f"Tuning {model_class.__name__}  "
            f"({len(param_space)} param(s), {n_calls} calls)\n"
            f"{'='*60}"
        )

        best_params, best_ndcg, skopt_result = tune_model(
            model_class=model_class,
            param_space=param_space,
            train=train,
            val=val,
            fit_kwargs=fit_kwargs,
            n_calls=n_calls,
            random_state=random_state,
        )

        all_results[key] = {"params": best_params, "ndcg_at_10": round(best_ndcg, 6)}
        # Running best NDCG at each call (for convergence plot)
        running_best = np.minimum.accumulate(-np.array(skopt_result.func_vals))
        convergence[key] = (-running_best).tolist()

        print(f"  → best params: {best_params}")
        print(f"  → best NDCG@10 on val: {best_ndcg:.4f}")

    # ── Persist results ───────────────────────────────────────────────────
    out_path = save_dir / "best_params.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBest params saved to {out_path}")

    # ── Summary table — show tuned models; mark preserved ones ───────────
    print("\nTuning summary (val NDCG@10):")
    print(f"  {'Model':<12} {'NDCG@10':>10}  Best params")
    print(f"  {'-'*12} {'-'*10}  {'-'*40}")
    for k, entry in all_results.items():
        tag = "" if k in keys_to_tune else "  (preserved)"
        print(f"  {k:<12} {entry['ndcg_at_10']:>10.4f}  {entry['params']}{tag}")

    # ── Convergence plot — only for models tuned this run ─────────────────
    if convergence:
        _plot_convergence(convergence, save_dir)

    return all_results


def _plot_convergence(
    convergence: dict[str, list[float]],
    save_dir: Path,
) -> None:
    """Save a convergence plot (best NDCG@10 vs call number), one panel per model."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; skipping convergence plot.")
        return

    n = len(convergence)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Bayesian Tuning Convergence — NDCG@10 on val", fontsize=13)

    for ax, (key, ndcg_curve) in zip(axes.flat, convergence.items()):
        calls = range(1, len(ndcg_curve) + 1)
        ax.plot(calls, ndcg_curve, marker="o", markersize=3, linewidth=1.5)
        ax.set_title(key.upper())
        ax.set_xlabel("Call #")
        ax.set_ylabel("Best NDCG@10")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplot panels
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    plot_path = save_dir.parent / "plots" / "tuning_convergence.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot saved to {plot_path}")


def load_best_params(save_dir: Path | None = None) -> dict[str, dict]:
    """
    Load previously saved tuning results.

    Returns dict mapping model key → {'params': {...}, 'ndcg_at_10': float},
    or an empty dict if no results file exists yet.
    """
    save_dir = Path(save_dir) if save_dir else RESULTS_DIR / "tuning"
    path = save_dir / "best_params.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
