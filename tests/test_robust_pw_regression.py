"""
RobustPWRegression testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

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
    splits = [5, 10, 15, 20]

    with raises(ValueError):
        pw = RobustPWRegression(solver="direct", objective="l1")
        pw.fit(x, y, splits)

    with raises(ValueError):
        pw = RobustPWRegression(solver="direct", monotonic_trend="peak")
        pw.fit(x, y, splits)

    with raises(ValueError):
        pw = RobustPWRegression(solver="direct")
        pw.fit(x, y, splits, lb=2)


def test_solver_osqp():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    with raises(ValueError):
        pw = RobustPWRegression(solver="osqp", objective="l1")
        pw.fit(x, y, splits)

    with raises(ValueError):
        pw = RobustPWRegression(solver="osqp", regularization="l1")
        pw.fit(x, y, splits)

    # Monotonic trend: descending
    for degree in (0, 1, 2):
        pw = RobustPWRegression(
            solver="osqp", degree=degree, monotonic_trend="descending")
        pw.fit(x, y, splits)

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3

    # Monotonic trend: descending + continuous=False
    pw = RobustPWRegression(solver="osqp", degree=1,
                            monotonic_trend="descending", continuous=False)
    pw.fit(x, y, splits)
    assert np.all(pw.coef_[:, 1] <= 0)

    # Bounds
    x = X[:, 2]
    for continuous in (True, False):
        for degree in (1, 2):
            pw = RobustPWRegression(solver="osqp", degree=degree,
                                    monotonic_trend="descending",
                                    continuous=continuous)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))

    for monotonic_trend in ("descending", "convex", "peak", "valley"):
        for degree in (1, 2):
            print(monotonic_trend, degree)
            pw = RobustPWRegression(solver="osqp", degree=degree,
                                    monotonic_trend=monotonic_trend)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))


def test_solver_ecos():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    # Monotonic trend: descending
    for degree in (0, 1, 2):
        pw = RobustPWRegression(
            solver="ecos", objective="l1", degree=degree,
            monotonic_trend="descending")
        pw.fit(x, y, splits)

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3

    # Monotonic trend: descending + continuous=False
    pw = RobustPWRegression(solver="ecos", objective="huber", degree=1,
                            monotonic_trend="descending", continuous=False)
    pw.fit(x, y, splits)
    assert np.all(pw.coef_[:, 1] <= 0)

    # Bounds
    x = X[:, 2]
    for continuous in (True, False):
        for degree in (1, 2):
            pw = RobustPWRegression(solver="ecos", objective="quantile",
                                    degree=degree, quantile=0.4,
                                    monotonic_trend="descending",
                                    continuous=continuous)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))

    for monotonic_trend in ("descending", "convex", "peak", "valley"):
        for degree in (1, 2):
            print(monotonic_trend, degree)
            pw = RobustPWRegression(solver="ecos", degree=degree,
                                    monotonic_trend=monotonic_trend)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))

    # No splits
    x = X[:, -1]
    for degree in (1, 2):
        for monotonic_trend in ("convex", "descending"):
            pw = RobustPWRegression(
                solver="ecos", objective="l1", degree=degree,
                monotonic_trend=monotonic_trend)

            pw.fit(x, y, splits=[], lb=5, ub=50)
            pred = pw.predict(x)
            eps = 1e-6
            assert np.all((5 - eps <= pred) & (pred <= 50 + eps))


def test_solver_ecos_regularization():
    pass


def test_predict():
    pw = RobustPWRegression()

    with raises(NotFittedError):
        pw.predict(x)


def test_predict_bounds():
    pass


def test_status():
    pw = RobustPWRegression()
    pw.fit(x, y, splits=[5, 10, 15, 20])

    assert pw.status == "optimal"


def test_stats():
    pw = RobustPWRegression()
    pw.fit(x, y, splits=[5, 10, 15, 20])

    assert pw.stats['n_variables'] == 10
    assert pw.stats['n_constraints'] == 14
