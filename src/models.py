
import numpy as np


def bias_baseline(R):
    mu = np.nanmean(R)

    b_user = np.nanmean(R - mu, axis=1)
    residual = R - mu - b_user[:, None]

    b_item = np.zeros(R.shape[1])

    for i in range(R.shape[1]):
        b_item[i] = np.nanmean(residual[:, i])

    b_user = np.nan_to_num(b_user)

    return mu, b_user, b_item
