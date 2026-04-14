"""
Exploratory data analysis helpers.
Run as a script to print a summary report and save plots.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from src.data_prep.loader import load_and_split

PLOTS_DIR = _PROJECT_ROOT / "results" / "plots"


def print_summary(data: dict) -> None:
    ratings = data["ratings"]
    train, val, test = data["train"], data["val"], data["test"]
    matrix = data["train_matrix"]

    print("=" * 60)
    print("DATASET SUMMARY — MovieLens 100K")
    print("=" * 60)
    print(f"  Total ratings   : {len(ratings):,}")
    print(f"  Unique users    : {ratings['user_id'].nunique():,}")
    print(f"  Unique items    : {ratings['item_id'].nunique():,}")
    print(f"  Rating range    : {ratings['rating'].min()} – {ratings['rating'].max()}")
    print(f"  Mean rating     : {ratings['rating'].mean():.3f}")
    print()
    print("SPLITS")
    print(f"  Train           : {len(train):,} ({100*len(train)/len(ratings):.1f}%)")
    print(f"  Val             : {len(val):,}  ({100*len(val)/len(ratings):.1f}%)")
    print(f"  Test            : {len(test):,}  ({100*len(test)/len(ratings):.1f}%)")
    print()
    sparsity = 1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print("INTERACTION MATRIX (train)")
    print(f"  Shape           : {matrix.shape}")
    print(f"  Sparsity        : {sparsity:.4%}")
    print("=" * 60)


def plot_eda(data: dict, save_dir: Path = PLOTS_DIR) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ratings = data["ratings"]
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MovieLens 100K — EDA", fontsize=15, fontweight="bold")

    # Rating distribution
    ax = axes[0, 0]
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

    # Ratings per user
    ax = axes[0, 1]
    user_counts = ratings.groupby("user_id").size()
    ax.hist(user_counts, bins=40, color="teal", edgecolor="white")
    ax.set_title("Ratings per User")
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of users")

    # Ratings per item
    ax = axes[1, 0]
    item_counts = ratings.groupby("item_id").size()
    ax.hist(item_counts, bins=40, color="coral", edgecolor="white")
    ax.set_title("Ratings per Item")
    ax.set_xlabel("Number of ratings")
    ax.set_ylabel("Number of items")

    # Timestamp distribution (activity over time)
    ax = axes[1, 1]
    ts = pd.to_datetime(ratings["timestamp"], unit="s")
    ts.dt.to_period("M").value_counts().sort_index().plot(ax=ax, color="purple")
    ax.set_title("Rating Activity Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ratings")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = Path(save_dir) / "eda_overview.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"EDA plot saved → {out}")


if __name__ == "__main__":
    try:
        data = load_and_split()
    except FileNotFoundError as e:
        print(
            f"ERROR: Could not load MovieLens data — {e}\n"
            "Make sure you have downloaded the dataset:\n"
            "  cd data/raw && curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip "
            "&& unzip ml-100k.zip"
        )
        sys.exit(1)
    print_summary(data)
    plot_eda(data)
