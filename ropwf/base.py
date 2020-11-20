import numbers

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.exceptions import NotFittedError

from .direct import lsq_direct
from .direct import lsq_direct_separated
from .cvx_socp import socp
from .cvx_socp import socp_separated
from .cvx_qp import qp


def _check_parameters(objective, degree, continuous, monotonic_trend, solver,
                      h_epsilon, quantile, verbose):

    if objective not in ("l1", "l2", "huber", "quantile"):
        raise ValueError('Invalid value for objective. Allowed string '
                         'values are "l1", "l2", "huber" and "quantile".')

    if not isinstance(degree, numbers.Integral) or not 0 <= degree <= 5:
        raise ValueError("degree must be an integer in [0, 5].")

    if not isinstance(continuous, bool):
        raise TypeError("continuous must be a boolean; got {}."
                        .format(verbose))

    if monotonic_trend is not None:
        if monotonic_trend not in ("ascending", "descending", "convex",
                                   "concave"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "ascending", "descending", '
                             '"convex" and "concave".')

        if monotonic_trend in ("convex", "concave") and degree > 1:
            raise ValueError('Option monotonic trend "convex" and "concave" '
                             'valid if degree <= 1.')

        if monotonic_trend in ("convex", "concave") and not continuous:
            raise ValueError('Option monotonic trend "convex" and "concave" '
                             'valid if "continuous"=True.')

    if solver not in ("auto", "ecos", "osqp", "direct"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "auto", "ecos", "osqp" and "direct".')

    if not isinstance(h_epsilon, numbers.Number) or h_epsilon < 1.0:
        raise ValueError("h_epsilon must a number >= 1.0; got {}."
                         .format(h_epsilon))

    if not isinstance(quantile, numbers.Number) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be a value in (0, 1); got {}."
                         .format(quantile))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def _choose_method(objective, degree, continuous, monotonic_trend, solver,
                   bounded):

    if solver == "auto":
        if bounded:
            if objective == "l2" and not continuous:
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
                    if continuous:
                        return "qp"
                    else:
                        return "socp_separated"
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


def _check_splits(splits):
    user_splits = check_array(splits, ensure_2d=False, force_all_finite=True)

    if len(set(user_splits)) != len(user_splits):
        raise ValueError("splits are not unique.")

    sorted_idx = np.argsort(user_splits)
    user_splits = user_splits[sorted_idx]

    return user_splits


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

        # Check inputs x and y
        x = check_array(x, ensure_2d=False, force_all_finite=True)
        y = check_array(y, ensure_2d=False, force_all_finite=True)
        check_consistent_length(x, y)

        idx = np.argsort(x)
        xs = x[idx]
        ys = y[idx]

        # Check user splits
        _check_splits(splits)

        # Check bounds
        bounded = (lb is not None or ub is not None)

        if bounded:
            _check_bounds(lb, ub)

        # Choose the most appropiate method/solver given the parameters
        _method = _choose_method(self.objective, self.degree, self.continuous,
                                 self.monotonic_trend, self.solver, bounded)

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

    def fit_predict(self, x, y, splits, lb=None, ub=None):
        return self.fit(x, y, splits, lb, ub).predict(x, lb, ub)

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
