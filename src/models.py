
import numpy as np


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def global_mean_model(train, test):
    mean = train["rating"].mean()
    return np.full(len(test), mean)


def user_mean_model(train, test):
    means = train.groupby("user")["rating"].mean()
    return test["user"].map(means).fillna(train["rating"].mean()).to_numpy()


def item_mean_model(train, test):
    means = train.groupby("item")["rating"].mean()
    return test["item"].map(means).fillna(train["rating"].mean()).to_numpy()
