"""
SVD-based Matrix Factorization.

Applies truncated SVD to the mean-centred user–item rating matrix.
The reconstructed matrix (U Σ Vᵀ + user_means) gives a predicted
rating for every (user, item) pair, which is used both for ranking
and for explicit rating prediction.

Requires train_matrix, user_id_to_idx, and item_id_to_idx via fit_kwargs.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from .base import BaseRecommender


class SVDRecommender(BaseRecommender):

    name = "SVD"
    supports_rating_prediction = True

    def __init__(self, n_factors: int = 50, random_state: int = 42) -> None:
        self.n_factors = n_factors
        self.random_state = random_state
        self._predicted: np.ndarray = np.array([])
        self._user_means: np.ndarray = np.array([])
        self._user_id_to_idx: dict[int, int] = {}
        self._item_id_to_idx: dict[int, int] = {}
        self._item_ids: np.ndarray = np.array([])
        self._user_seen: dict[int, set[int]] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        train: pd.DataFrame,
        *,
        train_matrix: csr_matrix = None,
        user_id_to_idx: dict = None,
        item_id_to_idx: dict = None,
        **kwargs,
    ) -> "SVDRecommender":
        if train_matrix is None or user_id_to_idx is None or item_id_to_idx is None:
            raise ValueError(
                "SVDRecommender requires train_matrix, user_id_to_idx, and "
                "item_id_to_idx. Pass them via fit_kwargs."
            )

        self._user_id_to_idx = user_id_to_idx
        self._item_id_to_idx = item_id_to_idx
        self._item_ids = np.array(
            [iid for iid, _ in sorted(item_id_to_idx.items(), key=lambda x: x[1])]
        )
        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        mat = train_matrix.astype(np.float64).toarray()  # (n_users, n_items)
        rated_mask = mat > 0

        # Per-user mean over rated entries only
        row_counts = rated_mask.sum(axis=1).astype(np.float64)
        row_counts[row_counts == 0] = 1.0
        self._user_means = mat.sum(axis=1) / row_counts    # (n_users,)
        self._global_mean = float(train["rating"].mean())

        # Mean-centre rated entries; unrated entries stay at 0.
        # np.repeat replicates each user's mean once per rated item so the
        # subtraction is fully vectorised — the same approach used in UserCF.
        centered = mat.copy()
        centered[rated_mask] -= np.repeat(
            self._user_means, rated_mask.sum(axis=1)
        )

        # Truncated SVD — svds returns factors in ascending singular-value order.
        # np.random.seed does not seed ARPACK; reproducibility requires passing
        # an explicit v0 starting vector drawn from a seeded RNG.
        k = min(self.n_factors, min(centered.shape) - 1)
        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(centered.shape))
        U, sigma, Vt = svds(centered, k=k, v0=v0)

        # Reverse to descending order
        idx = np.argsort(sigma)[::-1]
        U = U[:, idx]
        sigma = sigma[idx]
        Vt = Vt[idx, :]

        # Reconstruct and add user means back
        self._predicted = (U * sigma) @ Vt + self._user_means[:, None]
        self._predicted = np.clip(self._predicted, 1.0, 5.0).astype(np.float32)
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        u_idx = self._user_id_to_idx.get(user_id)
        if u_idx is None:
            return []

        scores = self._predicted[u_idx]
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

    def predict_rating(self, user_id: int, item_id: int) -> float:
        u_idx = self._user_id_to_idx.get(user_id)
        i_idx = self._item_id_to_idx.get(item_id)
        if u_idx is None or i_idx is None:
            return self._global_mean
        # _predicted is already clipped to [1, 5] during fit(); no re-clip needed.
        return float(self._predicted[u_idx, i_idx])
