import cvxpy as cp

from .matrices import matrix_A
from .matrices import matrix_A_D
from .matrices import matrix_CC
from .matrices import matrix_S


def model(x, y, splits, degree, lb, ub, objective, continuous, monotonicity,
          h_epsilon, quantile, verbose):
    n_bins = len(splits) + 1
    order = degree + 1

    # decision variables
    nvar = order * n_bins
    c = cp.Variable(nvar)

    if monotonicity in ("ascending", "descending"):
        A, D = matrix_A_D(x, splits, order)
    else:
        A = matrix_A(x, splits, order)

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

    constraints = []

    if continuous:
        S = matrix_S(x, splits, order)
        constraints.append(S * c == 0)

    if monotonicity == "ascending":
        constraints.append(D * c >= 0)
    elif monotonicity == "descending":
        constraints.append(D * c <= 0)
    elif monotonicity in ("convex", "concave"):
        CC = matrix_CC(splits)    
        if monotonicity == "convex":
            constraints.append(CC * x >= 0)
        elif monotonicity == "concave":
            constraints.append(CC * x <= 0)

    if lb is not None:
        constraints.append(A * c >= lb)
    elif ub is not None:
        constraints.append(A * c <= ub)

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.ECOS, verbose=verbose)

    return c.value.reshape((n_bins, order))
