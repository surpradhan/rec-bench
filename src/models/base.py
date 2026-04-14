"""
Abstract base class that every recommender must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """
    All recommenders share this interface so they plug into the same
    evaluation pipeline without modification.

    Subclass contract
    -----------------
    1. Set `name` to a short human-readable string.
    2. If the model produces explicit rating predictions, set
       `supports_rating_prediction = True` and override `predict_rating()`.
    3. If the model uses any randomness (SGD, random init, dropout, …),
       accept a `random_state` parameter in __init__ and use it to seed
       every stochastic operation.  This ensures reproducibility across runs.

    Passing metadata to fit()
    -------------------------
    Models that need more than the ratings DataFrame (e.g. content-based models
    requiring genre vectors, or matrix models requiring the pre-built sparse
    matrix) should accept these via keyword arguments to fit():

        def fit(self, train, *, movies=None, train_matrix=None, **kwargs):
            ...

    The evaluator calls model.fit(train) by default.  To pass extra context,
    use the `fit_kwargs` parameter of evaluate() / benchmark():

        benchmark([model], train, test, fit_kwargs={"movies": data["movies"]})

    This keeps the base-class signature minimal while allowing each model to
    declare exactly which extra data it needs.
    """

    name: str = "BaseRecommender"

    # Set to True in subclasses that override predict_rating().
    # The evaluator uses this flag to skip the RMSE/MAE computation for
    # ranking-only models instead of calling predict_rating() ~9,500 times
    # just to collect NaNs.
    supports_rating_prediction: bool = False

    @abstractmethod
    def fit(self, train: pd.DataFrame, **kwargs) -> "BaseRecommender":
        """
        Train the model on the train split.

        Parameters
        ----------
        train    : pd.DataFrame  columns [user_id, item_id, rating, timestamp]
        **kwargs : optional context — models may declare keyword-only args:
                   movies         pd.DataFrame  — for content-based models
                   train_matrix   csr_matrix    — for matrix factorization models
                   user_id_to_idx dict          — for fast index lookups
                   item_id_to_idx dict          — for fast index lookups
                   user_ids       np.ndarray    — full id universe
                   item_ids       np.ndarray    — full id universe

        Must return `self` so callers can chain: model = MyModel().fit(train)
        """

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[int]:
        """
        Return a ranked list of **at most** n item_ids for user_id.

        Requirements
        ------------
        - Return type must be list[int] (plain Python ints, not numpy int64).
        - Length must be <= n.
        - If exclude_seen=True, omit items the user rated during training.
        - Return an empty list for cold-start users rather than raising.
        """

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating a user would give an item.

        Override this method AND set supports_rating_prediction = True in
        subclasses that produce explicit rating predictions.
        The default raises NotImplementedError to catch accidental calls.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support rating prediction. "
            "Set supports_rating_prediction = True and override predict_rating()."
        )
