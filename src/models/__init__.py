"""
Model registry.

Add every new recommender here.  The benchmark runner imports from this
dict so nothing else needs to change when a new model is added.

Usage
-----
from src.models import MODELS, get_model

all_models = [cls() for cls in MODELS.values()]
popularity  = get_model("popularity")
"""

from .popularity import PopularityRecommender
from .content_based import ContentBasedRecommender
from .user_cf import UserCFRecommender
from .item_cf import ItemCFRecommender
from .svd import SVDRecommender
from .als import ALSRecommender
from .ncf import NCFRecommender
from .hybrid import HybridRecommender

MODELS: dict[str, type] = {
    "popularity":    PopularityRecommender,
    "content_based": ContentBasedRecommender,
    "user_cf":       UserCFRecommender,
    "item_cf":       ItemCFRecommender,
    # Week 3
    "svd":           SVDRecommender,
    "als":           ALSRecommender,
    "ncf":           NCFRecommender,
    "hybrid":        HybridRecommender,
}


def get_model(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in MODELS:
        raise KeyError(
            f"Unknown model {name!r}. Available: {list(MODELS.keys())}"
        )
    return MODELS[name](**kwargs)
