"""
Content-Based Recommender.

Builds a genre-based feature vector for each item (19 binary MovieLens genre
flags), then constructs a user profile as the rating-weighted average of the
feature vectors of items the user rated.  Recommendations are the unseen items
with highest cosine similarity to the user profile.
"""

import numpy as np
import pandas as pd
from .base import BaseRecommender


GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


class ContentBasedRecommender(BaseRecommender):

    name = "ContentBased"

    def __init__(self) -> None:
        self._item_ids: np.ndarray = np.array([])          # (n_items,) int
        self._item_features: np.ndarray = np.array([])     # (n_items, n_genres) L2-normalised
        self._item_id_to_feat_idx: dict[int, int] = {}
        self._user_profiles: dict[int, np.ndarray] = {}   # user_id → L2-normalised vector
        self._user_seen: dict[int, set[int]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        *,
        movies: pd.DataFrame = None,
        # item_id_to_idx and other matrix kwargs are accepted but unused:
        # content-based builds its own item index from movies metadata.
        **kwargs,
    ) -> "ContentBasedRecommender":
        if movies is None:
            raise ValueError(
                "ContentBasedRecommender requires a movies DataFrame. "
                "Pass it via fit_kwargs={'movies': data['movies']}."
            )

        # ── Build item feature matrix ─────────────────────────────────────
        genre_cols = [c for c in GENRE_COLS if c in movies.columns]
        feat_df = (
            movies[["item_id"] + genre_cols]
            .dropna(subset=genre_cols)
            .set_index("item_id")
        )
        self._item_ids = feat_df.index.to_numpy(dtype=int)
        feat_matrix = feat_df.values.astype(np.float32)

        # L2-normalise rows; items with no genre get a zero vector
        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._item_features = feat_matrix / norms

        self._item_id_to_feat_idx = {
            int(iid): i for i, iid in enumerate(self._item_ids)
        }

        # ── Build per-user seen sets ──────────────────────────────────────
        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        # ── Build user profiles ───────────────────────────────────────────
        # Use (rating − global_mean) as weight so items the user dislikes
        # pull the profile *away* from their genres rather than toward them.
        # With raw weights, a rating of 1 contributes a positive genre component
        # even though the user actively disliked that item.
        global_mean = float(train["rating"].mean())

        n_genres = feat_matrix.shape[1]
        for user_id, group in train.groupby("user_id"):
            item_ids = group["item_id"].values
            raw_weights = group["rating"].values.astype(np.float32)
            weights = raw_weights - global_mean   # centred: dislikes → negative

            feat_indices = np.array(
                [self._item_id_to_feat_idx.get(int(iid), -1) for iid in item_ids]
            )
            valid = feat_indices >= 0
            if not valid.any():
                self._user_profiles[int(user_id)] = np.zeros(n_genres, dtype=np.float32)
                continue

            vi = feat_indices[valid]
            vw = weights[valid]
            profile = (vw[:, None] * self._item_features[vi]).sum(axis=0)
            # No division by sum: signed weights can sum to ~0; the L2-norm
            # below handles scale. A zero-norm profile (all ratings at global
            # mean) gets a zero vector → no recommendations, which is correct.

            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm

            self._user_profiles[int(user_id)] = profile

        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        profile = self._user_profiles.get(user_id)
        if profile is None:
            return []

        # Cosine similarity: both sides are L2-normalised
        scores = self._item_features @ profile  # (n_items,)

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()

        order = np.argsort(scores)[::-1]
        recs: list[int] = []
        for idx in order:
            iid = int(self._item_ids[idx])
            if iid not in seen:
                recs.append(iid)
                if len(recs) == n:
                    break
        return recs
