"""
Robust piecewise regression.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

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
from .cvx_qp import qp_separated


def _check_parameters(objective, regularization, degree, continuous,
                      monotonic_trend, solver, h_epsilon, quantile, reg_l1,
                      reg_l2, verbose):

    if objective not in ("l1", "l2", "huber", "quantile"):
        raise ValueError('Invalid value for objective. Allowed string '
                         'values are "l1", "l2", "huber" and "quantile".')

    if regularization is not None:
        if regularization not in ("l1", "l2"):
            raise ValueError('Invalid value for regularization. Allowed '
                             'string values are "l1" and "l2".')

    if not isinstance(degree, numbers.Integral) or not 0 <= degree <= 5:
        raise ValueError("degree must be an integer in [0, 5].")

    if not isinstance(continuous, bool):
        raise TypeError("continuous must be a boolean; got {}."
                        .format(verbose))

    if monotonic_trend is not None:
        if monotonic_trend not in ("ascending", "descending", "convex",
                                   "concave", "peak", "valley"):
            raise ValueError('Invalid value for monotonic trend. Allowed '
                             'string values are "ascending", "descending", '
                             '"convex", "concave", "peak" and "valley".')

        if (monotonic_trend in ("convex", "concave", "peak", "valley")
                and not continuous):
            raise ValueError('Option monotonic trend "convex", "concave", '
                             '"peak" and "valley" valid if continuous=True.')

    if solver not in ("auto", "ecos", "osqp", "direct", "scs", "highs"):
        raise ValueError('Invalid value for solver. Allowed string '
                         'values are "auto", "ecos", "osqp", "direct", '
                         '"scs" and "highs".')

    if not isinstance(h_epsilon, numbers.Number) or h_epsilon < 1.0:
        raise ValueError("h_epsilon must a number >= 1.0; got {}."
                         .format(h_epsilon))

    if not isinstance(quantile, numbers.Number) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be a value in (0, 1); got {}."
                         .format(quantile))

    if not isinstance(reg_l1, numbers.Number) or reg_l1 < 0.0:
        raise ValueError("reg_l1 must be a positive value; got {}."
                         .format(reg_l1))

    if not isinstance(reg_l2, numbers.Number) or reg_l2 < 0.0:
        raise ValueError("reg_l2 must be a positive value; got {}."
                         .format(reg_l2))

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a boolean; got {}.".format(verbose))


def _choose_method(objective, degree, continuous, monotonic_trend, solver,
                   bounded, regularization):

    if solver == "auto":
        if bounded or regularization is not None:
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
                elif degree <= 1:
                    if continuous or degree == 0:
                        return "qp"
                    else:
                        return "qp_separated"
                else:
                    if continuous:
                        return "socp"
                    else:
                        return "socp_separated"
            else:
                if continuous:
                    return "socp"
                else:
                    return "socp_separated"
    elif solver == "direct":
        if (objective != "l2" or monotonic_trend is not None or bounded
                or regularization is not None):
            raise ValueError('solver "direct" only for objective="l2", '
                             'monotonic_trend=None, lb=ub=None, and '
                             'regularization=None.')
        else:
            if continuous:
                return "lsq_direct"
            else:
                return "lsq_direct_separated"
    elif solver == "osqp":
        if objective == "l2" and regularization is None:
            if continuous:
                return "qp"
            else:
                return "qp_separated"
        else:
            raise ValueError('solver="osqp" only for objective="l2" and '
                             'regularization=None.')
    elif solver in ("ecos", "scs"):
        if continuous or degree == 0:
            return "socp"
        else:
            return "socp_separated"
    elif solver == "highs":
        if objective == "l1" and regularization in (None, "l1"):
            if continuous:
                return "socp"
            else:
                return "socp_separated"
        else:
            raise ValueError('solver="highs" only for objective="l1" and '
                             'regularization in (None, "l1").')


def _check_bounds(lb, ub):
    if lb is not None:
        if not isinstance(lb, numbers.Number):
            raise TypeError("lb must be a number or None; got {}."
                            .format(type(bool)))

    if ub is not None:
        if not isinstance(ub, numbers.Number):
            raise TypeError("ub must be a number or None; got {}."
                            .format(type(bool)))

    if lb is not None and ub is not None:
        if lb > ub:
            raise ValueError("lb must be <= ub.")


def _check_splits(splits, monotonic_trend):
    if not isinstance(splits, (list, np.ndarray)):
        raise TypeError("splits must be a list or numpy.ndarray.")

    if not len(splits):
        if monotonic_trend in ("peak", "valley"):
            raise ValueError('monotonic trend "peak" and "valley" require a '
                             'list of splits.')

        return splits
    else:
        user_splits = check_array(splits, ensure_2d=False,
                                  force_all_finite=True)

        if len(set(user_splits)) != len(user_splits):
            raise ValueError("splits are not unique.")

        sorted_idx = np.argsort(user_splits)
        user_splits = user_splits[sorted_idx]

        return user_splits


class RobustPWRegression(BaseEstimator):
    """Robust piecewise regression.

    Parameters
    ----------
    objective : str, optional (default="l2")
        The objective function. Supported objectives are "l2", "l1", "huber"
        and "quantile". Note that "l1", "huber" and "quantile" are robust
        objective functions.

    regularization: str or None (default=None)
        Type of regularization. Supported regularization are "l1" (Lasso) and
        "l2" (Ridge). If None, no regularization is applied.

    degree : int (default=1)
        The degree of the polynomials.

        * degree = 0: piecewise constant functions.
        * degree = 1: piecewise linear functions.
        * degree > 1: piecewise polynomial functions.

    continuous : bool (default=True)
        Whether to fit a continuous or discontinuous piecewise regression.

    monotonic_trend : str or None, optional (default=None)
        The monotonic trend. Supported trends are "ascending", "descending",
        "convex" and "concave". If None, then the monotonic constraint is
        disabled.

    solver : str, optional (default="auto")
        The optimizer to solve the underlying mathematical optimization
        problem. Supported solvers are `"ecos"
        <https://github.com/embotech/ecos>`_, `"osqp"
        <https://github.com/oxfordcontrol/osqp>`_, "direct", to choose the
        direct solver, and "auto", to choose the most appropriate solver for
        the problem. Version 0.3.0 added support to solvers
        `"scs" <https://github.com/cvxgrp/scs>`_ and `"highs"
        <https://github.com/ERGO-Code/HiGHS>`_.

    h_epsilon: float (default=1.35)
        The parameter h_epsilon used when ``objective="huber"``, controls the
        number of samples that should be classified as outliers.

    quantile : float (default=0.5)
        The parameter quantile is the q-th quantile to be used when
        ``objective="quantile"``.

    reg_l1 : float (default=1.0)
        L1 regularization term. Increasing this value will smooth the
        regression model. Only applicable if ``regularization="l1"``.

    reg_l2 : float (default=1.0)
        L2 regularization term. Increasing this value will smooth the
        regression model. Only applicable if ``regularization="l2"``.

    verbose : bool (default=False)
        Enable verbose output.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_splits + 1, degree + 1)
        Coefficients for each bin. Number of bins = n_splits + 1.
    """
    def __init__(self, objective="l2", degree=1, continuous=True,
                 monotonic_trend=None, solver="auto", h_epsilon=1.35,
                 quantile=0.5, regularization=None, reg_l1=1.0, reg_l2=1.0,
                 verbose=False):

        self.objective = objective
        self.degree = degree
        self.continuous = continuous
        self.monotonic_trend = monotonic_trend
        self.solver = solver
        self.h_epsilon = h_epsilon
        self.quantile = quantile
        self.regularization = regularization
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.verbose = verbose

        self.coef_ = None

        self._is_fitted = False

    def fit(self, x, y, splits, lb=None, ub=None):
        """Fit the piecewise regression according to the given training data.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        lb : float or None (default=None)
            Add constraints to avoid values below the lower bound lb.

        ub : float or None (default=None)
            Add constraints to avoid values above the upper bound ub.

        Returns
        -------
        self : object
            Fitted piecewise regression.
        """
        _check_parameters(**self.get_params())

        # Check inputs x and y
        x = check_array(x, ensure_2d=False, force_all_finite=True)
        y = check_array(y, ensure_2d=False, force_all_finite=True)
        check_consistent_length(x, y)

        idx = np.argsort(x)
        xs = x[idx]
        ys = y[idx]

        # Check user splits
        _check_splits(splits, self.monotonic_trend)

        # Check bounds
        bounded = (lb is not None or ub is not None)

        if bounded:
            _check_bounds(lb, ub)

        # Choose the most appropriate method/solver given the parameters
        _method = _choose_method(self.objective, self.degree, self.continuous,
                                 self.monotonic_trend, self.solver, bounded,
                                 self.regularization)

        if _method == "lsq_direct":
            c, info = lsq_direct(xs, ys, splits, self.degree)
        elif _method == "lsq_direct_separated":
            c, info = lsq_direct_separated(xs, ys, splits, self.degree)
        elif _method == "socp":
            c, info = socp(xs, ys, splits, self.degree, self.continuous, lb,
                           ub, self.objective, self.monotonic_trend,
                           self.h_epsilon, self.quantile, self.regularization,
                           self.reg_l1, self.reg_l2, self.solver, self.verbose)
        elif _method == "socp_separated":
            c, info = socp_separated(xs, ys, splits, self.degree, lb, ub,
                                     self.objective, self.monotonic_trend,
                                     self.h_epsilon, self.quantile,
                                     self.solver, self.verbose)
        elif _method == "qp":
            c, info = qp(xs, ys, splits, self.degree, self.continuous, lb, ub,
                         self.monotonic_trend, self.verbose)
        elif _method == "qp_separated":
            c, info = qp_separated(xs, ys, splits, self.degree, lb, ub,
                                   self.monotonic_trend, self.verbose)

        self.coef_ = c

        if self.continuous or self.degree == 0:
            self._status = info["status"]
            self._stats = info["stats"]
        else:
            self._status = [_info["status"] for _info in info]
            self._stats = [_info["stats"] for _info in info]

        self._splits = splits
        self._is_fitted = True

        return self

    def fit_predict(self, x, y, splits, lb=None, ub=None):
        """Fit the piecewise regression according to the given training data,
        then predict.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = (n_samples,)
            Target vector relative to x.

        lb : float or None (default=None)
            Fit impose constraints to satisfy that values are greater or equal
            than lb. In predict, values below the lower bound lb are clipped to
            lb.

        ub : float or None (default=None)
            Fit impose constraints to satisfy that values are less or equal
            than ub. In predict, values above the upper bound ub are clipped to
            ub.

        Returns
        -------
        p : numpy array, shape = (n_samples,)
            Predicted array.
        """
        return self.fit(x, y, splits, lb, ub).predict(x, lb, ub)

    def predict(self, x, lb=None, ub=None):
        """Predict using the piecewise regression.

        Parameters
        ----------
        x : array-like, shape = (n_samples,)
            Training vector, where n_samples is the number of samples.

        lb : float or None (default=None)
            Values below the lower bound lb are clipped to lb.

        ub : float or None (default=None)
            Values above the upper bound ub are clipped to ub.

        Returns
        -------
        p : numpy array, shape = (n_samples,)
            Predicted array.
        """
        self._check_is_fitted()

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

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This {} instance is not fitted yet. Call "
                                 "'fit' with appropriate arguments."
                                 .format(self.__class__.__name__))

    @property
    def status(self):
        """The status of the underlying optimization solver.

        Returns
        -------
        status : str
        """
        self._check_is_fitted()

        return self._status

    @property
    def stats(self):
        """The number of variables and constraints of the underlying
        optimization problem.

        Returns
        -------
        stats : dict
        """
        self._check_is_fitted()

        return self._stats
