"""
RobustPWRegression testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

from pytest import approx, raises

from ropwr import RobustPWRegression
from sklearn.datasets import load_boston
from sklearn.exceptions import NotFittedError


X, y = load_boston(return_X_y=True)
x = X[:, -1]


def test_params():
    with raises(ValueError):
        pw = RobustPWRegression(objective="l0")
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(regularization="l0")
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(degree=7)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(TypeError):
        pw = RobustPWRegression(continuous=1)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(monotonic_trend="new")
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(monotonic_trend="convex", continuous=False)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(solver=None)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(h_epsilon=0.9)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(quantile=0)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(reg_l1=-0.5)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(reg_l2=-0.5)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(TypeError):
        pw = RobustPWRegression(verbose=1)
        pw.fit(x, y, splits=[5, 10, 15])


def test_splits():
    pass


def test_bounds():
    pass


def test_continuous():
    pass


def test_discontinuous():
    pass


def test_solver_auto():
    pass


def test_solver_direct():
    pass


def test_solver_osqp():
    pass


def test_solver_ecos():
    pass
