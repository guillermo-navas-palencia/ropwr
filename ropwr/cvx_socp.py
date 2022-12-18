"""
General Second-order Cone Programming (SOCP) formulation for piecewise
regression.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import cvxpy as cp
import numpy as np

from .cvx import compute_change_point
from .cvx import monotonic_trend_constraints
from .cvx import problem_info
from .matrices import matrix_A
from .matrices import matrix_A_D
from .matrices import matrix_A_H
from .matrices import matrix_D
from .matrices import matrix_H
from .matrices import matrix_S
from .matrices import submatrix_A
from .matrices import submatrix_A_D
from .matrices import submatrix_D


def _model_objective(A, c, y, objective, regularization, h_epsilon, quantile,
                     reg_l1, reg_l2):
    if objective == "l1":
        obj = cp.norm(A @ c - y, 1)
    elif objective == "l2":
        obj = cp.norm(A @ c - y, 2)
    elif objective == "huber":
        obj = cp.sum(cp.huber(A @ c - y, h_epsilon))
    elif objective == "quantile":
        obj1 = 0.5 * cp.norm(A @ c - y, 1)
        obj2 = (quantile - 0.5) * cp.sum(A @ c - y)
        obj = obj1 + obj2

    if regularization == "l1":
        obj += reg_l1 * cp.norm(c, 1)
    elif regularization == "l2":
        obj += reg_l2 * cp.norm(c, 2)

    return cp.Minimize(obj)


def socp(x, y, splits, degree, continuous, continuous_deriv, lb, ub, objective,
         monotonic_trend, h_epsilon, quantile, regularization, reg_l1, reg_l2,
         solver, max_iter, verbose):

    # Parameters
    n_bins = len(splits) + 1
    order = degree + 1

    t = None
    ti = None
    if monotonic_trend in ("ascending", "descending", "peak", "valley"):
        if order <= 2:
            A = matrix_A(x, splits, order)
            D = matrix_D(x, splits, order)
        else:
            A, D = matrix_A_D(x, splits, order)

        if monotonic_trend in ("peak", "valley"):
            t, ti = compute_change_point(x, y, splits, order, monotonic_trend)
    elif monotonic_trend in ("convex", "concave"):
        if order <= 2:
            A = matrix_A(x, splits, order)
            D = matrix_H(x, splits, order)
        else:
            A, D = matrix_A_H(x, splits, order)
    else:
        A = matrix_A(x, splits, order)
        D = None

    # Decision variables
    nvar = order * n_bins
    c = cp.Variable(nvar)

    # Objective function
    obj = _model_objective(A, c, y, objective, regularization, h_epsilon,
                           quantile, reg_l1, reg_l2)

    # Constraints
    constraints = []
    if n_bins > 1 and continuous:
        S = matrix_S(x, splits, order, continuous_deriv)
        constraints.append(S @ c == 0)

    if monotonic_trend:
        mono_cons = monotonic_trend_constraints(monotonic_trend, c, D, t)
        if isinstance(mono_cons, list):
            constraints.extend(mono_cons)
        else:
            constraints.append(mono_cons)

    if lb is not None:
        if monotonic_trend in ("ascending", "descending"):
            constraints.append(A[[0, -1], :] @ c >= lb)
        elif monotonic_trend in ("peak", "valley"):
            constraints.append(A[[0, ti, -1], :] @ c >= lb)
        else:
            constraints.append(A @ c >= lb)
    if ub is not None:
        if monotonic_trend in ("ascending", "descending"):
            constraints.append(A[[0, -1], :] @ c <= ub)
        elif monotonic_trend in ("peak", "valley"):
            constraints.append(A[[0, ti, -1], :] @ c <= ub)
        else:
            constraints.append(A @ c <= ub)

    # Solve
    prob = cp.Problem(obj, constraints)

    if solver in ("auto", "ecos", "scs"):
        if solver in ("auto", "ecos"):
            solve_options = {'solver': cp.ECOS, 'verbose': verbose}
        else:
            solve_options = {'solver': cp.SCS, 'verbose': verbose}

        if max_iter is not None:
            solve_options['max_iters'] = max_iter
    elif solver == "highs":
        solve_options = {'solver': cp.SCIPY}

    if solver == "highs":
        prob.solve(**solve_options, scipy_options={'method': "highs"})
    else:
        prob.solve(**solve_options)

    size_metrics = cp.problems.problem.SizeMetrics(prob)
    status = prob.status
    info = problem_info(status, size_metrics)

    return c.value.reshape((n_bins, order)), info


def socp_separated(x, y, splits, degree, lb, ub, objective,
                   monotonic_trend, h_epsilon, quantile, solver, max_iter,
                   verbose):

    if solver in ("auto", "ecos", "scs"):
        if solver in ("auto", "ecos"):
            solve_options = {'solver': cp.ECOS, 'verbose': verbose}
        else:
            solve_options = {'solver': cp.SCS, 'verbose': verbose}

        if max_iter is not None:
            solve_options['max_iters'] = max_iter
    elif solver == "highs":
        solve_options = {'solver': cp.SCIPY}

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
            if order == 2:
                Ai = submatrix_A(ni, xi, order)
                Di = submatrix_D(order)
            else:
                Ai, Di = submatrix_A_D(ni, xi, order)
        else:
            Ai = submatrix_A(ni, xi, order)
            Di = None

        # Decision variables
        ci = cp.Variable(order)

        # Objective function
        obj = _model_objective(Ai, ci, yi, objective, None, h_epsilon,
                               quantile, None, None)

        # Constraints
        constraints = []
        if monotonic_trend:
            mono_cons = monotonic_trend_constraints(monotonic_trend, ci, Di)
            constraints.append(mono_cons)

        if lb is not None:
            constraints.append(Ai @ ci >= lb)
        if ub is not None:
            constraints.append(Ai @ ci <= ub)

        prob = cp.Problem(obj, constraints)

        if solver == "highs":
            prob.solve(**solve_options, scipy_options={'method': "highs"})
        else:
            prob.solve(**solve_options)

        size_metrics = cp.problems.problem.SizeMetrics(prob)
        status = prob.status
        info = problem_info(status, size_metrics)
        infos.append(info)

        c[i, :] = ci.value

    return c, infos
