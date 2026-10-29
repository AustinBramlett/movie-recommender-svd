
import numpy as np
import pandas as pd


def split_ratings(ratings, test_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(ratings), dtype=bool)

    user_left = ratings.groupby("user").size().to_dict()
    item_left = ratings.groupby("item").size().to_dict()

    for pos in rng.permutation(len(ratings)):
        row = ratings.iloc[pos]
        user = row["user"]
        item = row["item"]

        if (rng.random() < test_frac
                and user_left[user] > 1
                and item_left[item] > 1):
            is_test[pos] = True
            user_left[user] -= 1
            item_left[item] -= 1

    train = ratings.iloc[~is_test].copy()
    test = ratings.iloc[is_test].copy()

    return train, test


def to_matrix(ratings):
    R = np.full((943, 1682), np.nan)

    for _, row in ratings.iterrows():
        user = int(row["user"]) - 1
        item = int(row["item"]) - 1
        R[user][item] = row["rating"]

    return R


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))
