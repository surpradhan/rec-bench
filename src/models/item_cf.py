"""
Item-Based Collaborative Filtering.

Computes item–item cosine similarity from the transposed user–item matrix.
For a target user, unseen items are scored as the dot product of their
similarity vectors with the user's rating vector, giving a weighted sum of
similarities to all items the user has rated.  The full item–item similarity
matrix is precomputed at fit time (1349×1349 ≈ 7 MB for this dataset).

Requires train_matrix, user_id_to_idx, and item_id_to_idx via fit_kwargs.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .base import BaseRecommender


class ItemCFRecommender(BaseRecommender):

    name = "ItemCF"

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors
        self._item_sim: np.ndarray = np.array([])       # (n_items, n_items)
        self._train_dense: np.ndarray = np.array([])    # (n_users, n_items)
        self._user_id_to_idx: dict[int, int] = {}
        self._item_ids: np.ndarray = np.array([])       # col-index → item_id
        self._user_seen: dict[int, set[int]] = {}

    def fit(
        self,
        train: pd.DataFrame,
        *,
        train_matrix: csr_matrix = None,
        user_id_to_idx: dict = None,
        item_id_to_idx: dict = None,
        **kwargs,
    ) -> "ItemCFRecommender":
        if train_matrix is None or user_id_to_idx is None or item_id_to_idx is None:
            raise ValueError(
                "ItemCFRecommender requires train_matrix, user_id_to_idx, and "
                "item_id_to_idx. Pass them via fit_kwargs."
            )

        self._user_id_to_idx = user_id_to_idx
        self._item_ids = np.array(
            [iid for iid, _ in sorted(item_id_to_idx.items(), key=lambda x: x[1])]
        )

        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        # ── Build item–item cosine similarity ─────────────────────────────
        mat = train_matrix.astype(np.float32).toarray()  # (n_users, n_items)
        self._train_dense = mat

        item_mat = mat.T                                  # (n_items, n_users)

        # Mean-centre each item vector over its rated entries only.
        # Without this, globally popular items share high dot-product
        # similarity due to rating magnitude, not co-preference — an item
        # rated [5,5,1] and one rated [1,1,5] would appear 40% similar.
        rated_mask = item_mat > 0                         # (n_items, n_users)
        rated_counts = rated_mask.sum(axis=1, keepdims=True).astype(np.float32)
        rated_counts[rated_counts == 0] = 1.0
        item_means = item_mat.sum(axis=1, keepdims=True) / rated_counts
        item_mat_c = item_mat.copy()
        item_mat_c[rated_mask] -= np.repeat(
            item_means[:, 0], rated_mask.sum(axis=1)
        )

        norms = np.linalg.norm(item_mat_c, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = item_mat_c / norms                  # (n_items, n_users)

        # Keep only top-n_neighbors similarities per item to limit noise
        sim_full = normalized @ normalized.T             # (n_items, n_items)
        np.fill_diagonal(sim_full, 0.0)

        # Zero out all but the top n_neighbors similarities for each item
        if self.n_neighbors < sim_full.shape[0]:
            threshold_idx = np.argpartition(sim_full, -self.n_neighbors, axis=1)[
                :, : -self.n_neighbors
            ]
            rows = np.arange(sim_full.shape[0])[:, None].repeat(
                sim_full.shape[0] - self.n_neighbors, axis=1
            )
            sim_full[rows, threshold_idx] = 0.0

        self._item_sim = sim_full
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        u_idx = self._user_id_to_idx.get(user_id)
        if u_idx is None:
            return []

        user_ratings = self._train_dense[u_idx].copy()   # (n_items,)
        rated = user_ratings > 0
        if not rated.any():
            return []

        # Mean-centre the user's ratings before scoring so that users who
        # rate everything highly don't get different item rankings from users
        # with identical relative preferences but a lower overall rating level.
        # Without this, score[j] gains a user_mean × Σ sim(j, seen) bias that
        # varies per item and changes ranking order (affects 96% of users).
        user_ratings[rated] -= user_ratings[rated].mean()

        # score[j] = Σ_i sim[j,i] * centred_rating[i]  for seen items i
        scores = self._item_sim @ user_ratings            # (n_items,)

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
