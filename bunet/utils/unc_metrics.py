"""
Implementations of the uncertainty metrics in numpy. For evaluation/testing-only.
"""

import numpy as np


def mi_uncertainty(prd):
    # https://arxiv.org/pdf/1703.02910.pdf
    mcs = np.repeat(np.expand_dims(prd, -1), 2, -1)
    mcs[..., 0] = 1 - mcs[..., 1]  # 10, nb_img, 192, 192, 64, 2

    entropy = -np.sum(np.mean(mcs, 0) * np.log(np.mean(mcs, 0) + 1e-5), -1)
    expected_entropy = -np.mean(np.sum(mcs * np.log(mcs + 1e-5), -1), 0)
    mi = entropy - expected_entropy
    return mi


def entropy(prd_mcs):
    mcs = np.repeat(np.expand_dims(prd_mcs, -1), 2, -1)
    mcs[..., 0] = 1 - mcs[..., 1]  # 10, 50, 192, 192, 64, 2
    return -np.sum(np.mean(mcs, 0) * np.log(np.mean(mcs, 0) + 1e-5), -1)


def prd_variance(log_var_mcs):
    return np.mean(np.exp(np.clip(log_var_mcs, -7, 7)), 0)  # (nb_mc, 192, 192, 64) --> (1, 192, 192, 64)


def prd_uncertainty(mu_mcs, prd_var):
    # = Var(mu_mcs) + prd_var
    return np.mean(np.square(mu_mcs), 0) - np.square(np.mean(mu_mcs, 0)) + prd_var
