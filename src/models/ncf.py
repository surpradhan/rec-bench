"""
Neural Collaborative Filtering (NCF).

Implements the He et al. (2017) NeuMF architecture: a Generalised Matrix
Factorisation (GMF) path and an MLP path are trained jointly and their
output vectors are concatenated before a final linear prediction layer.

Design note — MSE vs BPR
------------------------
The original NeuMF paper targets implicit feedback and uses log-loss or BPR
(Bayesian Personalised Ranking).  Here we train with MSE on explicit ratings
(1–5 stars) instead, making NCF behave as a rating predictor that is then
repurposed for ranking.  This choice allows apples-to-apples RMSE/MAE
comparison with SVD, and keeps the training signal consistent with the
rest of the benchmark (all explicit ratings, same train split).  The
trade-off is that the ranking loss is not directly optimised; NDCG/MAP
scores reflect rating-prediction quality applied to ranking, not a
ranking-specialised objective.

Requires user_id_to_idx and item_id_to_idx via fit_kwargs.
"""

import numpy as np
import pandas as pd
from .base import BaseRecommender


def _build_neumf(n_users, n_items, emb_size, mlp_layers):
    """
    Construct the NeuMF model.  Imported lazily so that importing this module
    does not require PyTorch to be installed — consistent with als.py and
    hybrid.py, which also lazy-import their optional dependencies inside fit().
    """
    import torch.nn as nn

    class _NeuMF(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb_gmf = nn.Embedding(n_users, emb_size)
            self.item_emb_gmf = nn.Embedding(n_items, emb_size)
            self.user_emb_mlp = nn.Embedding(n_users, emb_size)
            self.item_emb_mlp = nn.Embedding(n_items, emb_size)

            mlp_parts = []
            in_size = emb_size * 2
            for out_size in mlp_layers:
                mlp_parts.append(nn.Linear(in_size, out_size))
                mlp_parts.append(nn.ReLU())
                in_size = out_size
            self.mlp = nn.Sequential(*mlp_parts)

            self.output_layer = nn.Linear(emb_size + mlp_layers[-1], 1)

            for emb in (self.user_emb_gmf, self.item_emb_gmf,
                        self.user_emb_mlp, self.item_emb_mlp):
                nn.init.normal_(emb.weight, std=0.01)

        def forward(self, user_ids, item_ids):
            import torch
            u_gmf = self.user_emb_gmf(user_ids)
            i_gmf = self.item_emb_gmf(item_ids)
            gmf_out = u_gmf * i_gmf

            u_mlp = self.user_emb_mlp(user_ids)
            i_mlp = self.item_emb_mlp(item_ids)
            mlp_out = self.mlp(torch.cat([u_mlp, i_mlp], dim=1))

            combined = torch.cat([gmf_out, mlp_out], dim=1)
            return self.output_layer(combined).squeeze(1)

    return _NeuMF()


class NCFRecommender(BaseRecommender):

    name = "NCF"
    supports_rating_prediction = True

    def __init__(
        self,
        emb_size: int = 32,
        mlp_layers: tuple[int, ...] = (64, 32, 16),
        n_epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.emb_size = emb_size
        self.mlp_layers = tuple(mlp_layers)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self._user_id_to_idx: dict[int, int] = {}
        self._item_id_to_idx: dict[int, int] = {}
        self._item_ids: np.ndarray = np.array([])
        self._user_seen: dict[int, set[int]] = {}
        self._global_mean: float = 3.0

    def fit(
        self,
        train: pd.DataFrame,
        *,
        user_id_to_idx: dict = None,
        item_id_to_idx: dict = None,
        **kwargs,
    ) -> "NCFRecommender":
        if user_id_to_idx is None or item_id_to_idx is None:
            raise ValueError(
                "NCFRecommender requires user_id_to_idx and item_id_to_idx. "
                "Pass them via fit_kwargs."
            )

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "NCFRecommender requires PyTorch. "
                "Install it with: pip install torch"
            ) from exc

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._user_id_to_idx = user_id_to_idx
        self._item_id_to_idx = item_id_to_idx
        self._item_ids = np.array(
            [iid for iid, _ in sorted(item_id_to_idx.items(), key=lambda x: x[1])]
        )
        self._user_seen = (
            train.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self._global_mean = float(train["rating"].mean())

        n_users = len(user_id_to_idx)
        n_items = len(item_id_to_idx)

        self._model = _build_neumf(n_users, n_items, self.emb_size, self.mlp_layers)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        user_idx_t = torch.tensor(
            [user_id_to_idx[int(uid)] for uid in train["user_id"]], dtype=torch.long
        )
        item_idx_t = torch.tensor(
            [item_id_to_idx[int(iid)] for iid in train["item_id"]], dtype=torch.long
        )
        ratings_t = torch.tensor(train["rating"].values, dtype=torch.float32)

        dataset = TensorDataset(user_idx_t, item_idx_t, ratings_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for _ in range(self.n_epochs):
            for u_batch, i_batch, r_batch in loader:
                optimizer.zero_grad()
                preds = self._model(u_batch, i_batch)
                loss = loss_fn(preds, r_batch)
                loss.backward()
                optimizer.step()

        self._model.eval()
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        u_idx = self._user_id_to_idx.get(user_id)
        if u_idx is None:
            return []

        import torch
        n_items = len(self._item_ids)
        user_t = torch.full((n_items,), u_idx, dtype=torch.long)
        item_t = torch.arange(n_items, dtype=torch.long)

        with torch.no_grad():
            scores = self._model(user_t, item_t).numpy()

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
        import torch
        u_t = torch.tensor([u_idx], dtype=torch.long)
        i_t = torch.tensor([i_idx], dtype=torch.long)
        with torch.no_grad():
            pred = self._model(u_t, i_t).item()
        return float(np.clip(pred, 1.0, 5.0))
