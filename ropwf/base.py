import numbers

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .direct import lsq_direct
from .direct import lsq_direct_separated
from .cvx_socp import socp
from .cvx_socp import socp_separated
from .cvx_qp import qp


def _check_parameters(objective, degree, continuous, monotonic_trend, solver,
                      h_epsilon, quantile, verbose):

    if objective not in ("l1", "l2", "huber", "quantile"):
        raise ValueError()

    if not isinstance(degree, numbers.Integral) or not 0 <= degree <= 5:
        raise ValueError()

    if not isinstance(continuous, bool):
        raise TypeError()

    if monotonic_trend is not None:
        if monotonic_trend not in ("ascending", "descending", "convex",
                                   "concave"):
            raise ValueError()

    if solver not in ("auto", "ecos", "osqp", "direct"):
        raise ValueError()

    if not isinstance(h_epsilon, numbers.Number) or h_epsilon < 1.0:
        raise ValueError()

    if not isinstance(quantile, numbers.Number) or not 0.0 < quantile < 1.0:
        raise ValueError()

    if not isinstance(verbose, bool):
        raise TypeError()


def _choose_method(objective, degree, continuous, monotonic_trend, solver,
                   bounded):
    if solver == "auto":
        if bounded:
            if objective == "l2":
                return "qp"
            else:
                if continuous:
                    return "socp"
                else:
                    return "socp_separated"
        else:
            if objective == "l2":
                if monotonic_trend is None:
                    if continuous:
                        return "lsq_direct"
                    else:
                        return "lsq_direct_separated"
                else:
                    return "qp"
            else:
                if continuous:
                    return "socp"
                else:
                    return "socp_separated"
    elif solver == "direct":
        if objective != "l2" or monotonic_trend is not None or bounded:
            raise ValueError("")
        else:
            if continuous:
                return "lsq_direct"
            else:
                return "lsq_direct_separated"
    elif solver == "osqp":
        if objective == "l2" and continuous:
            return "qp"
        else:
            return "socp"
    elif solver == "ecos":
        if continuous:
            return "socp"
        else:
            return "socp_separated"


def _check_bounds(lb, ub):
    if not isinstance(lb, numbers.Number):
        raise TypeError("lb must be a number or None; got {}."
                        .format(type(bool)))

    if not isinstance(ub, numbers.Number):
        raise TypeError("ub must be a number or None; got {}."
                        .format(type(bool)))

    if lb is not None and ub is not None:
        if lb > ub:
            raise ValueError("lb must be <= ub.")


class RobustPWRegression(BaseEstimator):
    def __init__(self, objective="l2", degree=1, continuous=True,
                 monotonic_trend=None, solver="auto", h_epsilon=1.35,
                 quantile=0.5, verbose=False):

        self.objective = objective
        self.degree = degree
        self.continuous = continuous
        self.monotonic_trend = monotonic_trend
        self.solver = solver
        self.h_epsilon = h_epsilon
        self.quantile = quantile
        self.verbose = verbose

        self.coef_ = None

        self._is_fitted = False

    def fit(self, x, y, splits, lb=None, ub=None):
        _check_parameters(**self.get_params())

        idx = np.argsort(x)
        xs = x[idx]
        ys = y[idx]

        bounded = (lb is not None or ub is not None)
        _method = _choose_method(self.objective, self.degree, self.continuous,
                                 self.monotonic_trend, self.solver, bounded)

        if self.verbose:
            print("Method: {}".format(_method))

        if _method == "lsq_direct":
            c = lsq_direct(xs, ys, splits, self.degree)
        elif _method == "lsq_direct_separated":
            c = lsq_direct_separated(xs, ys, splits, self.degree)
        elif _method == "socp":
            c = socp(xs, ys, splits, self.degree, lb, ub, self.objective,
                     self.monotonic_trend, self.h_epsilon, self.quantile,
                     self.solver, self.verbose)
        elif _method == "socp_separated":
            c = socp_separated(xs, ys, splits, self.degree, lb, ub,
                               self.objective, self.monotonic_trend,
                               self.h_epsilon, self.quantile,
                               self.solver, self.verbose)
        elif _method == "qp":
            c = qp(xs, ys, splits, self.degree, lb, ub, self.objective,
                   self.monotonic_trend, self.verbose)

        self.coef_ = c
        self._splits = splits
        self._is_fitted = True

        return self

    def predict(self, x, lb=None, ub=None):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

        bounded = (lb is not None or ub is not None)

        if bounded:
            _check_bounds(lb, ub)

        n_bins = len(self._splits) + 1
        indices = np.digitize(x, self._splits, right=False)

        pred = np.zeros(x.shape)
        for i in range(n_bins):
            mask = (indices == i)
            pred[mask] = np.polyval(self.coef_[i, :][::-1], x[mask])

        bounded = (lb is not None or ub is not None)
        if bounded:
            pred = np.clip(pred, lb, ub)

        return pred
