"""
User-Based Collaborative Filtering.

For each target user, finds the n_neighbors most similar users (mean-centred
cosine similarity), then scores unseen items by a weighted sum
of neighbour ratings.  The pre-built user–user similarity matrix is stored
at fit time so recommend() is O(n_items) per call.

Requires train_matrix, user_id_to_idx, and item_id_to_idx via fit_kwargs.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .base import BaseRecommender


class UserCFRecommender(BaseRecommender):

    name = "UserCF"

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors
        self._user_sim: np.ndarray = np.array([])       # (n_users, n_users)
        self._centered: np.ndarray = np.array([])       # mean-centred dense matrix
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
    ) -> "UserCFRecommender":
        if train_matrix is None or user_id_to_idx is None or item_id_to_idx is None:
            raise ValueError(
                "UserCFRecommender requires train_matrix, user_id_to_idx, and "
                "item_id_to_idx. Pass them via fit_kwargs."
            )

        self._user_id_to_idx = user_id_to_idx
        # item_ids ordered by column index
        self._item_ids = np.array(
            [iid for iid, _ in sorted(item_id_to_idx.items(), key=lambda x: x[1])]
        )

        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        # ── Mean-centre ratings (only over rated items) ───────────────────
        mat = train_matrix.astype(np.float32).toarray()  # (n_users, n_items)
        rated_mask = mat > 0
        row_counts = rated_mask.sum(axis=1).astype(np.float32)
        row_counts[row_counts == 0] = 1.0
        user_means = mat.sum(axis=1) / row_counts         # (n_users,)

        centered = mat.copy()
        centered[rated_mask] -= np.repeat(user_means, rated_mask.sum(axis=1))

        # ── L2-normalise for cosine similarity ────────────────────────────
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = centered / norms                     # (n_users, n_items)

        # ── User–user similarity matrix ───────────────────────────────────
        self._user_sim = normalized @ normalized.T        # (n_users, n_users)
        np.fill_diagonal(self._user_sim, 0.0)

        self._centered = centered
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        u_idx = self._user_id_to_idx.get(user_id)
        if u_idx is None:
            return []

        sims = self._user_sim[u_idx]                     # (n_users,)
        k = min(self.n_neighbors, len(sims))

        # Top-k neighbours with positive similarity
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_idx = top_k_idx[sims[top_k_idx] > 0]
        if len(top_k_idx) == 0:
            return []

        neighbor_sims = sims[top_k_idx]                  # (k,)
        scores = neighbor_sims @ self._centered[top_k_idx]  # (n_items,)

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
