import cvxpy as cp
import numpy as np

from .cvx import _monotonic_trend_constraints
from .matrices import matrix_A
from .matrices import matrix_A_D
from .matrices import matrix_CC
from .matrices import matrix_D0
from .matrices import matrix_S
from .matrices import submatrix_A
from .matrices import submatrix_A_D


def _model_objective(A, c, y, objective, h_epsilon, quantile):
    if objective == "l1":
        obj = cp.Minimize(cp.norm(A * c - y, 1))
    elif objective == "l2":
        obj = cp.Minimize(cp.norm(A * c - y, 2))
    elif objective == "huber":
        obj = cp.Minimize(cp.sum(cp.huber(A * c - y, h_epsilon)))
    elif objective == "quantile":
        obj1 = 0.5 * cp.norm(A * c - y, 1)
        obj2 = (quantile - 0.5) * cp.sum(A * c - y)
        obj = cp.Minimize(obj1 + obj2)

    return obj


def socp(x, y, splits, degree, lb, ub, objective, monotonic_trend, h_epsilon,
         quantile, solver, verbose):

    # Parameters
    n_bins = len(splits) + 1
    order = degree + 1

    if monotonic_trend in ("ascending", "descending"):
        if order == 1:
            A = matrix_A(x, splits, order)
            D = matrix_D0(splits)
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

    # Objective function
    obj = _model_objective(A, c, y, objective, h_epsilon, quantile)

    # Constraints
    constraints = []
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

    if solver == "ecos":
        _solver = cp.ECOS
    elif solver == "osqp":
        _solver = cp.OSQP
    else:
        _solver = cp.ECOS

    prob.solve(solver=_solver, verbose=verbose)

    return c.value.reshape((n_bins, order))


def socp_separated(x, y, splits, degree, lb, ub, objective,
                   monotonic_trend, h_epsilon, quantile, solver, verbose):

    if solver == "ecos":
        _solver = cp.ECOS
    elif solver == "osqp":
        _solver = cp.OSQP
    else:
        _solver = cp.ECOS

    order = degree + 1
    n_bins = len(splits) + 1

    indices = np.searchsorted(splits, x, side='right')

    c = np.zeros((n_bins, order))

    for i in range(n_bins):
        mask = (indices == i)
        xi = x[mask]
        yi = y[mask]
        ni = len(xi)

        if monotonic_trend in ("ascending", "descending"):
            Ai, Di = submatrix_A_D(ni, xi, order)
        else:
            Ai = submatrix_A(ni, xi, order)
            Di = None

        # Decision variables
        ci = cp.Variable(order)

        # Objective function
        obj = _model_objective(Ai, ci, yi, objective, h_epsilon, quantile)

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
        prob.solve(solver=_solver, verbose=verbose)

        c[i, :] = ci.value

    return c
