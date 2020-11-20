from .matrices import matrix_CC


def _monotonic_trend_constraints(monotonic_trend, c, D, splits, order):
    if monotonic_trend == "ascending":
        if order <= 2:
            return c[1::order] >= 0
        else:
            return D * c >= 0
    elif monotonic_trend == "descending":
        if order <= 2:
            return c[1::order] <= 0
        else:
            return D * c <= 0
    elif monotonic_trend in ("convex", "concave"):
        CC = matrix_CC(splits)
        if monotonic_trend == "convex":
            return CC * c >= 0
        elif monotonic_trend == "concave":
            return CC * c <= 0


def _bound_constraints(A, c, lb, ub):
    b_constraints = []
    if lb is not None:
        b_constraints.append(A * c >= lb)
    elif ub is not None:
        b_constraints.append(A * c <= ub)

    return b_constraints
