"""
MovieLens 100K data loader and preprocessor.
Loads ratings, movies, and users; builds the interaction matrix;
produces consistent train/val/test splits.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "ml-100k"


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.data and return a tidy ratings DataFrame."""
    path = Path(data_dir) / "u.data"
    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int, "rating": float, "timestamp": int},
    )
    return ratings


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.item and return a movies DataFrame with genre columns."""
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    columns = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_names

    path = Path(data_dir) / "u.item"
    movies = pd.read_csv(
        path,
        sep="|",
        names=columns,
        encoding="latin-1",
        usecols=["item_id", "title", "release_date"] + genre_names,
    )
    movies["item_id"] = movies["item_id"].astype(int)
    # Parse release year from title or date
    movies["year"] = movies["release_date"].str.extract(r"(\d{4})").astype("Int64")
    return movies


def load_users(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.user and return a users DataFrame."""
    path = Path(data_dir) / "u.user"
    users = pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return users


def split_data(
    ratings: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    strategy: str = "time",
    min_ratings_for_holdout: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train / val / test sets.

    strategy='time'     — per-user, hold out the most recent interactions
    strategy='random'   — global random split

    Parameters
    ----------
    min_ratings_for_holdout : users with fewer interactions than this are kept
        entirely in train (no val/test holdout). Must match the
        min_ratings_per_user filter applied upstream, or be set explicitly.
        Default 5 matches the load_and_split() default.
    """
    ratings = ratings.copy()

    if strategy == "time":
        # item_id is a deterministic tiebreaker: 76% of MovieLens 100K ratings
        # share a (user_id, timestamp) pair, so without it the assignment of
        # same-second items to train vs test depends on arbitrary row order.
        ratings = ratings.sort_values(["user_id", "timestamp", "item_id"])
        train_rows, val_rows, test_rows = [], [], []

        for _, group in ratings.groupby("user_id"):
            n = len(group)
            n_test = max(1, int(np.floor(n * test_ratio)))
            n_val = max(1, int(np.floor(n * val_ratio)))
            # Require at least min_ratings_for_holdout interactions to create
            # a meaningful holdout.  This threshold must be kept in sync with
            # the min_ratings_per_user filter in load_and_split().
            if n < min_ratings_for_holdout:
                train_rows.append(group)
                continue
            train_rows.append(group.iloc[: n - n_val - n_test])
            val_rows.append(group.iloc[n - n_val - n_test : n - n_test])
            test_rows.append(group.iloc[n - n_test :])

        _empty = pd.DataFrame(columns=ratings.columns)
        train = pd.concat(train_rows).reset_index(drop=True) if train_rows else _empty
        val   = pd.concat(val_rows).reset_index(drop=True)   if val_rows   else _empty.copy()
        test  = pd.concat(test_rows).reset_index(drop=True)  if test_rows  else _empty.copy()

        if val.empty or test.empty:
            warnings.warn(
                f"split_data: val or test split is empty (val={len(val)}, test={len(test)}). "
                f"All users may be below min_ratings_for_holdout={min_ratings_for_holdout}. "
                f"Evaluation will not be possible."
            )

    elif strategy == "random":
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(ratings))
        n = len(ratings)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test = ratings.iloc[idx[:n_test]].reset_index(drop=True)
        val = ratings.iloc[idx[n_test : n_test + n_val]].reset_index(drop=True)
        train = ratings.iloc[idx[n_test + n_val :]].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'time' or 'random'.")

    return train, val, test


def build_interaction_matrix(
    ratings: pd.DataFrame,
    user_ids: np.ndarray | None = None,
    item_ids: np.ndarray | None = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str = "rating",
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Build a sparse user-item interaction matrix from a ratings DataFrame.

    Parameters
    ----------
    ratings  : DataFrame of interactions to encode as matrix entries
    user_ids : Optional pre-defined universe of user ids (determines row space).
               If None, derived from `ratings` itself.
    item_ids : Optional pre-defined universe of item ids (determines col space).
               If None, derived from `ratings` itself.

    Passing `user_ids` / `item_ids` derived from the full dataset (train+val+test)
    ensures all splits share the same index space, preventing out-of-bounds lookups
    when models map unseen val/test users or items.

    Returns
    -------
    matrix   : csr_matrix  shape (n_users, n_items)
    user_ids : np.ndarray  row index → user_id
    item_ids : np.ndarray  col index → item_id
    """
    if user_ids is None:
        user_ids = np.sort(ratings[user_col].unique())
    if item_ids is None:
        item_ids = np.sort(ratings[item_col].unique())

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    # Rows/cols for entries that exist in the id universe; drop unknowns with dropna
    rows = ratings[user_col].map(user_index).dropna().astype(int)
    cols = ratings[item_col].map(item_index).dropna().astype(int)
    valid = rows.index.intersection(cols.index)

    n_dropped = len(ratings) - len(valid)
    if n_dropped > 0:
        warnings.warn(
            f"build_interaction_matrix: {n_dropped} rating(s) were dropped because "
            f"their user_id or item_id is not in the provided universe. "
            f"This is expected when building val/test matrices from a train-derived universe."
        )

    data = ratings.loc[valid, rating_col].values.astype(np.float32)
    rows_arr = rows.loc[valid].values
    cols_arr = cols.loc[valid].values

    # Detect duplicate (user, item) pairs before building the matrix.
    # scipy.sparse sums duplicates silently, turning e.g. two ratings of
    # 5.0 and 3.0 into a matrix value of 8.0, which is meaningless.
    if len(rows_arr) > 0:
        pairs = np.stack([rows_arr, cols_arr], axis=1)
        n_unique = len(np.unique(pairs, axis=0))
        if n_unique < len(rows_arr):
            warnings.warn(
                f"build_interaction_matrix: {len(rows_arr) - n_unique} duplicate "
                f"(user, item) pair(s) detected. scipy.sparse will sum their ratings, "
                f"producing nonsensical matrix values. Deduplicate the input DataFrame first."
            )

    if len(rows_arr) > 0:
        assert rows_arr.min() >= 0 and rows_arr.max() < len(user_ids), (
            f"Row index out of bounds: [{rows_arr.min()}, {rows_arr.max()}] "
            f"not in [0, {len(user_ids) - 1}]"
        )
        assert cols_arr.min() >= 0 and cols_arr.max() < len(item_ids), (
            f"Col index out of bounds: [{cols_arr.min()}, {cols_arr.max()}] "
            f"not in [0, {len(item_ids) - 1}]"
        )

    matrix = csr_matrix(
        (data, (rows_arr, cols_arr)),
        shape=(len(user_ids), len(item_ids)),
    )
    return matrix, user_ids, item_ids


def load_and_split(
    data_dir: Path = DATA_DIR,
    min_ratings_per_user: int = 5,
    min_ratings_per_item: int = 5,
    strategy: str = "time",
    random_state: int = 42,
) -> dict:
    """
    Full pipeline: load → filter → split → build matrices.

    The user/item id universe is derived from the full (filtered) ratings set
    BEFORE splitting, so train, val, and test all share the same index space.
    This prevents silent index mismatches when models look up val/test users
    or items that happen not to appear in the training split.

    Returns a dict with keys:
        ratings, movies, users,
        train, val, test,
        train_matrix, val_matrix, test_matrix,
        user_ids, item_ids
    """
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)

    # Filter sparse users and items
    user_counts = ratings["user_id"].value_counts()
    item_counts = ratings["item_id"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    valid_items = item_counts[item_counts >= min_ratings_per_item].index
    ratings = ratings[
        ratings["user_id"].isin(valid_users) & ratings["item_id"].isin(valid_items)
    ].reset_index(drop=True)

    # ── Movies metadata validation ────────────────────────────────────────
    # Warn if any rated item is absent from movies.
    # Content-based models (Week 2) join on item_id; missing rows produce
    # silent NaN features that quietly degrade recommendation quality.
    missing_items = set(ratings["item_id"].unique()) - set(movies["item_id"])
    if missing_items:
        warnings.warn(
            f"Movies metadata is missing {len(missing_items)} item(s) that appear in ratings. "
            f"Content-based models may produce NaN features for these items. "
            f"Missing ids (first 10): {sorted(missing_items)[:10]}"
        )

    # Validate movies content: duplicate ids, binary genre flags, NaN genres.
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    if not movies["item_id"].is_unique:
        warnings.warn("Movies metadata contains duplicate item_ids. "
                      "Content-based joins will produce unexpected rows.")
    genre_cols_present = [c for c in genre_names if c in movies.columns]
    if genre_cols_present:
        nan_genres = movies[genre_cols_present].isna().any().any()
        if nan_genres:
            warnings.warn("Movies metadata has NaN values in genre columns. "
                          "Content-based feature vectors will contain NaN.")
        non_binary = ~movies[genre_cols_present].isin([0, 1]).all().all()
        if non_binary:
            warnings.warn("Movies genre columns contain values other than 0/1. "
                          "Content-based models expect binary genre indicators.")

    # Build the shared id universe from ALL filtered ratings, BEFORE splitting.
    # Using train-only ids here was the original data-leakage bug.
    # The names all_user_ids / all_item_ids make the scope unambiguous;
    # these are returned as "user_ids" / "item_ids" in the output dict so
    # callers always work with the full universe, not a train-only subset.
    all_user_ids = np.sort(ratings["user_id"].unique())
    all_item_ids = np.sort(ratings["item_id"].unique())

    train, val, test = split_data(
        ratings,
        strategy=strategy,
        min_ratings_for_holdout=min_ratings_per_user,
        random_state=random_state,
    )

    train_matrix, _, _ = build_interaction_matrix(
        train, user_ids=all_user_ids, item_ids=all_item_ids
    )
    val_matrix, _, _ = build_interaction_matrix(
        val, user_ids=all_user_ids, item_ids=all_item_ids
    )
    test_matrix, _, _ = build_interaction_matrix(
        test, user_ids=all_user_ids, item_ids=all_item_ids
    )

    # Pre-built reverse-lookup dicts so models don't re-implement O(n) searches.
    # Every CF / SVD / ALS model needs: row = user_id_to_idx[user_id]
    user_id_to_idx = {int(uid): i for i, uid in enumerate(all_user_ids)}
    item_id_to_idx = {int(iid): i for i, iid in enumerate(all_item_ids)}

    return {
        "ratings": ratings,
        "movies": movies,
        "users": users,
        "train": train,
        "val": val,
        "test": test,
        "train_matrix": train_matrix,
        "val_matrix": val_matrix,
        "test_matrix": test_matrix,
        # Full universe — not train-only. All splits share this index space.
        "user_ids": all_user_ids,
        "item_ids": all_item_ids,
        # O(1) reverse lookups: user_id / item_id → matrix row / col index
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
    }
