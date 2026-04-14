"""
Popularity Baseline Recommender.

Two modes:
  - 'count'      : rank items by number of ratings received
  - 'mean_rating': rank items by mean rating (with a Bayesian smoothing floor)

The popularity list is global — every user gets the same recommendations
minus items they have already seen (when exclude_seen=True).
"""

import numpy as np
import pandas as pd
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):

    name = "Popularity"

    def __init__(self, mode: str = "count", min_ratings: int = 10) -> None:
        """
        Parameters
        ----------
        mode        : 'count' or 'mean_rating'
        min_ratings : minimum number of ratings for an item to be ranked
                      (used only in mean_rating mode for Bayesian smoothing)
        """
        if mode not in ("count", "mean_rating"):
            raise ValueError("mode must be 'count' or 'mean_rating'")
        self.mode = mode
        self.min_ratings = min_ratings
        self._ranked_items: list[int] = []
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, train: pd.DataFrame, **kwargs) -> "PopularityRecommender":
        # Build per-user seen sets
        self._user_seen = (
            train.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        if self.mode == "count":
            scores = train.groupby("item_id").size().rename("score")
        else:
            stats = train.groupby("item_id")["rating"].agg(["count", "mean"])
            # Bayesian average: (n * mean + C * global_mean) / (n + C)
            global_mean = train["rating"].mean()
            C = self.min_ratings
            stats["score"] = (
                (stats["count"] * stats["mean"] + C * global_mean)
                / (stats["count"] + C)
            )
            scores = stats["score"]

        # Store as plain Python ints to satisfy the base class contract
        self._ranked_items: list[int] = [
            int(i) for i in scores.sort_values(ascending=False).index
        ]
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        recs = [item for item in self._ranked_items if item not in seen]
        return recs[:n]
