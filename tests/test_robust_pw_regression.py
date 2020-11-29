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
    with raises(TypeError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits=(5, 10, 15))

    with raises(ValueError):
        pw = RobustPWRegression(monotonic_trend="peak")
        pw.fit(x, y, splits=[])

    with raises(ValueError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits=[5, 5, 10])


def test_bounds():
    with raises(TypeError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits=[], lb=[1, 2])

    with raises(TypeError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits=[], ub=[1, 2])

    with raises(ValueError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits=[], lb=2, ub=1)


def test_continuous_default():
    splits = [5, 10, 15, 20]
    pw_d = RobustPWRegression(solver="direct")
    pw_d.fit(x, y, splits)

    pw_o = RobustPWRegression(solver="osqp")
    pw_o.fit(x, y, splits)

    pw_e = RobustPWRegression(solver="ecos")
    pw_e.fit(x, y, splits)

    assert pw_d.coef_ == approx(pw_o.coef_, rel=1e-6)
    assert pw_d.coef_ == approx(pw_e.coef_, rel=1e-6)


def test_discontinuous_default():
    splits = [5, 10, 15, 20]
    pw_d = RobustPWRegression(solver="direct", continuous=False)
    pw_d.fit(x, y, splits)

    pw_o = RobustPWRegression(solver="osqp", continuous=False)
    pw_o.fit(x, y, splits)

    pw_e = RobustPWRegression(solver="ecos", continuous=False)
    pw_e.fit(x, y, splits)

    assert pw_d.coef_ == approx(pw_o.coef_, rel=1e-6)
    assert pw_d.coef_ == approx(pw_e.coef_, rel=1e-6)


def test_solver_auto():
    pass


def test_solver_direct():
    pass


def test_solver_osqp():
    pass


def test_solver_ecos():
    pass


def test_predict():
    pw = RobustPWRegression()

    with raises(NotFittedError):
        pw.predict(x)


def test_predict_bounds():
    pass
