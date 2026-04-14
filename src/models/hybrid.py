"""
Hybrid Recommender (LightFM).

Uses the `lightfm` library to combine collaborative filtering latent factors
with content-based item features (MovieLens genre flags).  WARP loss is used
for implicit-feedback ranking; only ratings >= 4 are treated as positive
interactions, aligning with the benchmark's relevance_threshold=4.0.

This is a hybrid in the classic LightFM sense: the model learns separate latent
representations for users and items that combine an MF component with a linear
content component over the item feature vectors.

Requires movies, user_id_to_idx, and item_id_to_idx via fit_kwargs.
"""

import numpy as np
import pandas as pd
from .base import BaseRecommender
from .content_based import GENRE_COLS


class HybridRecommender(BaseRecommender):

    name = "Hybrid"

    def __init__(
        self,
        no_components: int = 50,
        loss: str = "warp",
        epochs: int = 20,
        n_threads: int = 4,
        random_state: int = 42,
    ) -> None:
        self.no_components = no_components
        self.loss = loss
        self.epochs = epochs
        self.n_threads = n_threads
        self.random_state = random_state
        self._user_seen: dict[int, set[int]] = {}
        self._user_id_map: dict[int, int] = {}
        self._item_id_map: dict[int, int] = {}
        self._idx_to_item_id: dict[int, int] = {}

    def fit(
        self,
        train: pd.DataFrame,
        *,
        movies: pd.DataFrame = None,
        user_id_to_idx: dict = None,
        item_id_to_idx: dict = None,
        **kwargs,
    ) -> "HybridRecommender":
        if movies is None or user_id_to_idx is None or item_id_to_idx is None:
            raise ValueError(
                "HybridRecommender requires movies, user_id_to_idx, and "
                "item_id_to_idx. Pass them via fit_kwargs."
            )

        try:
            from lightfm import LightFM
            from lightfm.data import Dataset
        except ImportError as exc:
            raise ImportError(
                "HybridRecommender requires the `lightfm` package. "
                "Install it with: pip install lightfm"
            ) from exc

        genre_cols = [c for c in GENRE_COLS if c in movies.columns]

        # ── Build LightFM Dataset ─────────────────────────────────────────
        dataset = Dataset()
        dataset.fit(
            users=list(user_id_to_idx.keys()),
            items=list(item_id_to_idx.keys()),
            item_features=genre_cols,
        )

        # Only use ratings >= 4 as positive interactions, aligning with the
        # benchmark's relevance_threshold=4.0.  WARP loss has no concept of
        # negative ratings; including 1-star interactions as positives trains
        # the model to rank disliked items higher, contradicting what is measured.
        positives = train[train["rating"] >= 4.0]
        interaction_pairs = list(
            zip(positives["user_id"].astype(int), positives["item_id"].astype(int))
        )
        interactions, _weights = dataset.build_interactions(interaction_pairs)

        # ── Build item feature matrix ─────────────────────────────────────
        # Filter to items in the benchmark universe, then build (item_id, [genres])
        # tuples without iterrows: stack genre columns into a boolean matrix,
        # then read active genre names per row via a list comprehension over
        # the column index — O(n_items × n_genres) but all in numpy/pandas.
        movies_in_universe = movies[movies["item_id"].isin(item_id_to_idx)]
        genre_matrix = movies_in_universe[genre_cols].values.astype(bool)
        genre_cols_arr = np.array(genre_cols)
        item_feature_tuples = [
            (int(iid), list(genre_cols_arr[row_mask]))
            for iid, row_mask in zip(
                movies_in_universe["item_id"].astype(int), genre_matrix
            )
        ]
        item_features = dataset.build_item_features(item_feature_tuples)

        # ── Fit LightFM model ─────────────────────────────────────────────
        model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            random_state=self.random_state,
        )
        # Note: num_threads > 1 requires LightFM compiled with OpenMP.
        # If the installed wheel lacks OpenMP support it silently falls back
        # to a single thread; n_threads is kept as a parameter for environments
        # where a multi-threaded build is available.
        model.fit(
            interactions,
            item_features=item_features,
            epochs=self.epochs,
            num_threads=self.n_threads,
            verbose=False,
        )

        # ── Store state for recommend() ───────────────────────────────────
        user_id_map, _, item_id_map, _ = dataset.mapping()
        self._user_id_map = user_id_map          # external_id → internal_idx
        self._item_id_map = item_id_map
        self._idx_to_item_id = {v: k for k, v in item_id_map.items()}
        self._item_features = item_features
        self._model = model
        self._n_items = len(item_id_map)
        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        u_internal = self._user_id_map.get(user_id)
        if u_internal is None:
            return []

        item_indices = np.arange(self._n_items)
        scores = self._model.predict(
            u_internal,
            item_indices,
            item_features=self._item_features,
        )

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        order = np.argsort(scores)[::-1]
        recs: list[int] = []
        for idx in order:
            iid = int(self._idx_to_item_id[int(idx)])
            if iid not in seen:
                recs.append(iid)
                if len(recs) == n:
                    break
        return recs
