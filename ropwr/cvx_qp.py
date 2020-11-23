"""
Quadratic programming (QP) formulation for piecewise regression with l2-norm
objective.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import cvxpy as cp
import numpy as np

from .cvx import _monotonic_trend_constraints
from .cvx import _problem_info
from .matrices import matrix_A
from .matrices import matrix_A_D
from .matrices import matrix_CC
from .matrices import matrix_D0
from .matrices import matrix_S
from .matrices import submatrix_A
from .matrices import submatrix_A_D


def qp(x, y, splits, degree, lb, ub, monotonic_trend, verbose):
    # Parameters
    n_bins = len(splits) + 1
    order = degree + 1

    if monotonic_trend in ("ascending", "descending"):
        if order == 1:
            A = matrix_A(x, splits, order)
            D = matrix_D0(splits)
        elif order == 2:
            A = matrix_A(x, splits, order)
            D = None
        else:
            A, D = matrix_A_D(x, splits, order)
    elif monotonic_trend in ("convex", "concave"):
        A = matrix_A(x, splits, order)
        D = matrix_CC(splits)
    else:
        A = matrix_A(x, splits, order)
        D = None

    # Decision variables
    nvar = order * n_bins
    c = cp.Variable(nvar)

    Q = A.T.dot(A)
    p = -2. * A.T.dot(y)

    # Objective function
    obj = cp.Minimize(cp.quad_form(c, Q) + p.T * c)

    # Constraints
    constraints = []
    if n_bins > 1:
        S = matrix_S(x, splits, order)
        constraints.append(S * c == 0)

    if monotonic_trend:
        mono_cons = _monotonic_trend_constraints(monotonic_trend, c, D, order)
        constraints.append(mono_cons)

    if lb is not None:
        constraints.append(A * c >= lb)
    if ub is not None:
        constraints.append(A * c <= ub)

    # Solve
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP, verbose=verbose)

    size_metrics = cp.problems.problem.SizeMetrics(prob)
    status = prob.status
    info = _problem_info(status, size_metrics)

    return c.value.reshape((n_bins, order)), info


def qp_separated(x, y, splits, degree, lb, ub, monotonic_trend, verbose):
    # Parameters
    order = degree + 1
    n_bins = len(splits) + 1

    indices = np.searchsorted(splits, x, side='right')

    c = np.zeros((n_bins, order))

    infos = []
    for i in range(n_bins):
        mask = (indices == i)
        xi = x[mask]
        yi = y[mask]
        ni = len(xi)

        if monotonic_trend in ("ascending", "descending"):
            if order == 1:
                Ai = submatrix_A_D(ni, xi, order)
                Di = matrix_D0(splits)
            elif order == 2:
                Ai = submatrix_A(ni, xi, order)
                Di = None
            else:
                Ai, Di = submatrix_A_D(ni, xi, order)
        elif monotonic_trend in ("convex", "concave"):
            Ai = submatrix_A(ni, xi, order)
            Di = matrix_CC(splits)
        else:
            Ai = submatrix_A(ni, xi, order)
            Di = None

        # Decision variables
        ci = cp.Variable(order)

        # Objective function
        Qi = Ai.T.dot(Ai)
        pi = -2. * Ai.T.dot(yi)

        obj = cp.Minimize(cp.quad_form(ci, Qi) + pi.T * ci)

        # Constraints
        constraints = []
        if monotonic_trend:
            mono_cons = _monotonic_trend_constraints(
                monotonic_trend, ci, Di, order)
            constraints.append(mono_cons)

        if lb is not None:
            constraints.append(Ai * ci >= lb)
        if ub is not None:
            constraints.append(Ai * ci <= ub)

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, verbose=verbose)

        size_metrics = cp.problems.problem.SizeMetrics(prob)
        status = prob.status
        info = _problem_info(status, size_metrics)
        infos.append(info)

        c[i, :] = ci.value

    return c, infos
