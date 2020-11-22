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
