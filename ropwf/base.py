import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .socp import model
from .system import solve_l2_system_continuous


class RobustPWRegression(BaseEstimator):
    def __init__(self, objective="l2", degree=1, continuous=True,
                 monotonicity=None, solver="exact", h_epsilon=1.35,
                 quantile=0.5, verbose=False):

        self.objective = objective
        self.degree = degree
        self.continuous = continuous
        self.monotonicity = monotonicity
        self.solver = solver
        self.h_epsilon = h_epsilon
        self.quantile = quantile
        self.verbose = verbose

        self.coef_ = None

        self._is_fitted = False

    def fit(self, x, y, splits, lb=None, ub=None):

        idx = np.argsort(x)

        xs = x[idx]
        ys = y[idx]

        if self.solver == "exact":
            c = solve_l2_system_continuous(xs, ys, splits, self.degree)
        else:
            c = model(xs, ys, splits, self.degree, lb, ub, self.objective,
                      self.continuous, self.monotonicity, self.h_epsilon,
                      self.quantile, self.verbose)

        self.coef_ = c
        self._splits = splits
        self._is_fitted = True

        return self

    def predict(self, x):
        n_bins = len(self._splits) + 1
        indices = np.digitize(x, self._splits, right=False)

        pred = np.zeros(x.shape)
        for i in range(n_bins):
            mask = (indices == i)
            pred[mask] = np.polyval(self.coef_[i, :][::-1], x[mask])

        return pred
