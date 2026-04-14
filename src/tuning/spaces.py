"""
Hyperparameter search spaces for each Week 3 model.

Each entry in PARAM_SPACES maps a model key to a tuple of:
    (ModelClass, [(param_name, skopt_dimension), ...], n_calls)

Parameter ranges are chosen to cover the practical operating range for
MovieLens 100K while staying within wall-clock budget:
  - SVD / ALS / Hybrid are fast (<1s–5s per fit), so more calls are affordable.
  - NCF is ~25s per fit; n_calls is kept low accordingly.

random_state is intentionally excluded from all spaces — it is fixed to 42
during tuning so that stochastic variation does not corrupt the signal.

mlp_layers is excluded from the NCF space because structural hyperparameters
(number of layers, per-layer width) form a non-continuous space that GP
handles poorly. The default (64, 32, 16) is a reasonable NeuMF architecture
and is kept fixed; emb_size, lr, and n_epochs are tuned instead.
"""

from skopt.space import Integer, Real

from src.models.svd import SVDRecommender
from src.models.als import ALSRecommender
from src.models.ncf import NCFRecommender
from src.models.hybrid import HybridRecommender


PARAM_SPACES: dict[str, tuple[type, list, int]] = {
    "svd": (
        SVDRecommender,
        [
            ("n_factors", Integer(10, 150, name="n_factors")),
        ],
        25,   # n_calls — 1D space converges quickly
    ),
    "als": (
        ALSRecommender,
        [
            ("n_factors",      Integer(10, 150,    name="n_factors")),
            # Upper bound extended from 1.0 → 10.0: previous run hit reg=1.0 (boundary).
            ("regularization", Real(1e-4, 10.0,    prior="log-uniform", name="regularization")),
            ("iterations",     Integer(10, 50,     name="iterations")),
            # Lower bound extended from 10.0 → 1.0: previous run hit alpha=10.0 (boundary).
            ("alpha",          Real(1.0, 100.0,    prior="log-uniform", name="alpha")),
        ],
        40,   # n_calls — extended space; extra 5 calls over original
    ),
    "ncf": (
        NCFRecommender,
        [
            ("emb_size", Integer(8, 64,    name="emb_size")),
            ("n_epochs", Integer(5, 20,    name="n_epochs")),
            ("lr",       Real(1e-4, 1e-2,  prior="log-uniform", name="lr")),
        ],
        20,   # n_calls — NCF is ~25s per fit; 20 calls ≈ 8 min
    ),
    "hybrid": (
        HybridRecommender,
        [
            ("no_components", Integer(10, 100,  name="no_components")),
            # Upper bound extended from 50 → 100: previous run hit epochs=50 (boundary).
            ("epochs",        Integer(10, 100,  name="epochs")),
        ],
        30,   # n_calls — extended space; extra 5 calls over original
    ),
}
