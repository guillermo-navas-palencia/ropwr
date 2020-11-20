import cvxpy as cp

from .cvx import _bound_constraints
from .cvx import _monotonic_trend_constraints
from .matrices import matrix_A
from .matrices import matrix_A_D
from .matrices import matrix_S


def qp(x, y, splits, degree, lb, ub, objective, monotonic_trend, verbose):
    # Parameters
    n_bins = len(splits) + 1
    order = degree + 1

    if monotonic_trend in ("ascending", "descending") and order > 2:
        A, D = matrix_A_D(x, splits, order)
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
    S = matrix_S(x, splits, order)
    constraints.append(S * c == 0)

    if monotonic_trend:
        mono_cons = _monotonic_trend_constraints(
            monotonic_trend, c, D, splits, order)
        constraints.append(mono_cons)

    bound_constraints = _bound_constraints(A, c, lb, ub)
    if bound_constraints:
        constraints.extend(bound_constraints)

    # Solve
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP, verbose=verbose)

    return c.value.reshape((n_bins, order))
