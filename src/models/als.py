"""
ALS-based Collaborative Filtering (implicit feedback).

Uses the `implicit` library's AlternatingLeastSquares.  Only ratings >= 4
are treated as positive interactions, aligning with the benchmark's
relevance_threshold=4.0.  Those ratings are then converted to confidence
values (c_ui = 1 + alpha * r_ui) so that a 5-star rating carries more
weight than a 4-star, but both are strictly positive.

Design note — confidence weighting on filtered positives
---------------------------------------------------------
The standard Hu et al. (2008) formula c_ui = 1 + alpha * r_ui was designed
for implicit counts (e.g. play counts), where higher values mean stronger
preference.  After filtering to ratings >= 4 the same formula is still
sensible: a 4-star gives c = 1 + 40*4 = 161 and a 5-star gives c = 201,
preserving the relative preference signal within the positive set.
alpha=40 is the Hu et al. default and works well on rating-scale data in
this range; it can be tuned if needed.

Requires train_matrix, user_id_to_idx, and item_id_to_idx via fit_kwargs.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .base import BaseRecommender


class ALSRecommender(BaseRecommender):

    name = "ALS"

    def __init__(
        self,
        n_factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 40.0,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self._user_id_to_idx: dict[int, int] = {}
        self._item_ids: np.ndarray = np.array([])
        self._user_seen: dict[int, set[int]] = {}
        self._conf_matrix: csr_matrix | None = None

    def fit(
        self,
        train: pd.DataFrame,
        *,
        train_matrix: csr_matrix = None,
        user_id_to_idx: dict = None,
        item_id_to_idx: dict = None,
        **kwargs,
    ) -> "ALSRecommender":
        if train_matrix is None or user_id_to_idx is None or item_id_to_idx is None:
            raise ValueError(
                "ALSRecommender requires train_matrix, user_id_to_idx, and "
                "item_id_to_idx. Pass them via fit_kwargs."
            )

        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ImportError(
                "ALSRecommender requires the `implicit` package. "
                "Install it with: pip install implicit"
            ) from exc

        self._user_id_to_idx = user_id_to_idx
        self._item_ids = np.array(
            [iid for iid, _ in sorted(item_id_to_idx.items(), key=lambda x: x[1])]
        )
        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        # Only treat ratings >= 4 as positive interactions, aligning with the
        # benchmark's relevance_threshold=4.0.  A 1-star rating is not a
        # positive signal; including it as a high-confidence entry trains the
        # model to recommend items a user disliked, directly contradicting the
        # evaluation criterion.
        positives = train[train["rating"] >= 4.0]

        rows = positives["user_id"].map(user_id_to_idx).values.astype(np.int32)
        cols = positives["item_id"].map(item_id_to_idx).values.astype(np.int32)
        # Confidence: c_ui = 1 + alpha * r_ui  (ratings are already >= 4)
        data = (1.0 + self.alpha * positives["rating"].values).astype(np.float32)

        n_users = len(user_id_to_idx)
        n_items = len(item_id_to_idx)
        self._conf_matrix = csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_items)
        )

        model = AlternatingLeastSquares(
            factors=self.n_factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
        )
        model.fit(self._conf_matrix)
        self._als = model
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        u_idx = self._user_id_to_idx.get(user_id)
        if u_idx is None:
            return []

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        # Request extra candidates to account for post-filtering seen items
        n_request = n + len(seen)

        ids, _scores = self._als.recommend(
            u_idx,
            self._conf_matrix[u_idx],
            N=n_request,
            filter_already_liked_items=False,
        )

        recs: list[int] = []
        for i_idx in ids:
            iid = int(self._item_ids[int(i_idx)])
            if iid not in seen:
                recs.append(iid)
                if len(recs) == n:
                    break
        return recs
