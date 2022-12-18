"""
Auxiliary functions for cvxpy formulations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np


def monotonic_trend_constraints(monotonic_trend, c, D, t=None):
    if monotonic_trend in ("ascending", "convex"):
        return D @ c >= 0
    elif monotonic_trend in ("descending", "concave"):
        return D @ c <= 0
    elif monotonic_trend == "valley":
        return [D[:t, :] @ c <= 0, D[t:, :] @ c >= 0]
    elif monotonic_trend == "peak":
        return [D[:t, :] @ c >= 0, D[t:, :] @ c <= 0]


def compute_change_point(x, y, splits, order, monotonic_trend):
    n_splits = len(splits)
    n_bins = n_splits + 1
    indices = np.searchsorted(splits, x, side='right')

    mean = [y[indices == i].mean() for i in range(n_bins)]

    if monotonic_trend == "peak":
        change_point = np.argmax(mean)
    else:
        change_point = np.argmin(mean)

    if change_point >= n_splits:
        change_point = n_splits - 1

    if order > 2:
        change_point = np.searchsorted(x, splits[change_point], side='right')
        cp_idx = change_point + 1
    else:
        cp_idx = np.searchsorted(x, splits[change_point], side='right')

    return change_point + 1, cp_idx - 1


def problem_info(status, size_metrics):
    n_variables = size_metrics.num_scalar_variables
    n_constraints = size_metrics.num_scalar_eq_constr
    n_constraints += size_metrics.num_scalar_leq_constr

    info = {
        "status": status,
        "stats": {"n_variables": n_variables, "n_constraints": n_constraints}
    }

    return info
