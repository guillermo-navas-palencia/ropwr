"""
Auxiliary functions for cvxpy formulations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020


def _monotonic_trend_constraints(monotonic_trend, c, D, order):
    if monotonic_trend == "ascending":
        if order == 2:
            return c[1::order] >= 0
        else:
            return D * c >= 0
    elif monotonic_trend == "descending":
        if order == 2:
            return c[1::order] <= 0
        else:
            return D * c <= 0
    elif monotonic_trend in ("convex", "concave"):
        if monotonic_trend == "convex":
            return D * c >= 0
        elif monotonic_trend == "concave":
            return D * c <= 0


def _problem_info(status, size_metrics):
    n_variables = size_metrics.num_scalar_variables
    n_constraints = size_metrics.num_scalar_eq_constr
    n_constraints += size_metrics.num_scalar_leq_constr

    info = {
        "status": status,
        "stats": {"n_variables": n_variables, "n_constraints": n_constraints}
    }

    return info
