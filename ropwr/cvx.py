"""
Auxiliary functions for cvxpy formulations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020


def _monotonic_trend_constraints(monotonic_trend, c, D):
    if monotonic_trend in ("ascending", "convex"):
        return D * c >= 0
    elif monotonic_trend in ("descending", "concave"):
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
