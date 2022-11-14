"""
Quadratic programming (QP) formulation for piecewise regression with l2-norm
objective.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import cvxpy as cp
import numpy as np

from cvxpy.atoms.affine.wraps import psd_wrap

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


def qp(x, y, splits, degree, continuous, continuous_deriv, lb, ub,
       monotonic_trend, max_iter, verbose):
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

    Q = psd_wrap(A.T.dot(A))
    p = -2. * A.T.dot(y)

    # Objective function
    obj = cp.Minimize(cp.quad_form(c, Q) + p.T @ c)

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

    solve_options = {'solver': cp.OSQP, 'verbose': verbose}
    if max_iter is not None:
        solve_options['max_iter'] = max_iter

    prob.solve(**solve_options)

    size_metrics = cp.problems.problem.SizeMetrics(prob)
    status = prob.status
    info = problem_info(status, size_metrics)

    return c.value.reshape((n_bins, order)), info


def qp_separated(x, y, splits, degree, lb, ub, monotonic_trend, max_iter,
                 verbose):

    solve_options = {'solver': cp.OSQP, 'verbose': verbose}
    if max_iter is not None:
        solve_options['max_iter'] = max_iter

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
        Qi = psd_wrap(Ai.T.dot(Ai))
        pi = -2. * Ai.T.dot(yi)

        obj = cp.Minimize(cp.quad_form(ci, Qi) + pi.T @ ci)

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
        prob.solve(**solve_options)

        size_metrics = cp.problems.problem.SizeMetrics(prob)
        status = prob.status
        info = problem_info(status, size_metrics)
        infos.append(info)

        c[i, :] = ci.value

    return c, infos
