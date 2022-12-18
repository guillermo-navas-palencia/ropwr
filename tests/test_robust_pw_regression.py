"""
RobustPWRegression testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from pytest import approx, raises

from ropwr import RobustPWRegression
from ropwr.base import _choose_method
from sklearn.exceptions import NotFittedError


def load_boston():
    X = np.genfromtxt('tests/datasets/boston.csv', skip_header=1,
                      delimiter=',')
    return X[:, :-1], X[:, -1]


X, y = load_boston()
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

    with raises(TypeError):
        pw = RobustPWRegression(continuous_deriv=1)
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

    with raises(ValueError):
        pw = RobustPWRegression(max_iter=-1)
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(extrapolation="new")
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(TypeError):
        pw = RobustPWRegression(extrapolation_bounds={})
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(extrapolation_bounds=(1,))
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(space="loglinear")
        pw.fit(x, y, splits=[5, 10, 15])

    with raises(ValueError):
        pw = RobustPWRegression(extrapolation="linear", space="log")
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
        pw = RobustPWRegression(monotonic_trend="peak")
        pw.fit(x, y, splits=None)

    with raises(ValueError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits=[5, 5, 10])

    with raises(ValueError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits="new_method")

    with raises(ValueError):
        pw = RobustPWRegression()
        pw.fit(x, y, splits="quantile", n_bins=1)


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
            solver="ecos", objective="l1", continuous_deriv=False,
            degree=degree, monotonic_trend="descending")
        pw.fit(x, y, splits)

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3

    # Monotonic trend: descending + continuous=False
    pw = RobustPWRegression(solver="ecos", objective="huber", degree=1,
                            monotonic_trend="descending", continuous=False,
                            continuous_deriv=False)
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

            pw.fit(x, y, splits=None, lb=5, ub=50)
            pred = pw.predict(x)
            eps = 1e-6
            assert np.all((5 - eps <= pred) & (pred <= 50 + eps))


def test_solver_ecos_regularization():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    pw = RobustPWRegression(solver="ecos", objective="l1",
                            degree=2, monotonic_trend="descending")
    pw.fit(x, y, splits)

    pwrl2 = RobustPWRegression(solver="ecos", objective="l1",
                               regularization="l2", degree=2,
                               monotonic_trend="descending")
    pwrl2.fit(x, y, splits)

    pwrl1 = RobustPWRegression(solver="ecos", objective="l1",
                               regularization="l1", degree=2,
                               monotonic_trend="descending")
    pwrl1.fit(x, y, splits)

    assert np.linalg.norm(pwrl2.coef_, 2) < np.linalg.norm(pw.coef_, 2)
    assert np.linalg.norm(pwrl2.coef_, 1) < np.linalg.norm(pw.coef_, 1)


def test_solver_highs():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    with raises(ValueError):
        pw = RobustPWRegression(solver="highs", objective="l2")
        pw.fit(x, y, splits)

    with raises(ValueError):
        pw = RobustPWRegression(solver="highs", objective="l1",
                                regularization="l2")
        pw.fit(x, y, splits)

    # Monotonic trend: descending
    for degree in (0, 1, 2):
        pw = RobustPWRegression(
            solver="highs", objective="l1", degree=degree,
            monotonic_trend="descending", continuous_deriv=False)
        pw.fit(x, y, splits)

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3

    # Bounds
    x = X[:, 2]
    for continuous in (True, False):
        for degree in (1, 2):
            pw = RobustPWRegression(solver="highs", objective="l1",
                                    degree=degree, quantile=0.4,
                                    monotonic_trend="descending",
                                    regularization="l1",
                                    reg_l1=0.5,
                                    continuous=continuous)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))


def test_solver_scs():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    # Monotonic trend: descending
    for degree in (0, 1, 2):
        pw = RobustPWRegression(
            solver="scs", objective="l1", degree=degree,
            monotonic_trend="descending", continuous_deriv=False)
        pw.fit(x, y, splits)

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3

    # Monotonic trend: descending + continuous=False
    pw = RobustPWRegression(solver="scs", objective="huber", degree=1,
                            monotonic_trend="descending", continuous=False)
    pw.fit(x, y, splits)
    assert np.all(pw.coef_[:, 1] <= 0)

    # Bounds
    x = X[:, 2]
    for continuous in (True, False):
        for degree in (1, 2):
            pw = RobustPWRegression(solver="scs", objective="quantile",
                                    degree=degree, quantile=0.4,
                                    monotonic_trend="descending",
                                    continuous=continuous)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))

    for monotonic_trend in ("descending", "convex", "peak", "valley"):
        for degree in (1, 2):
            print(monotonic_trend, degree)
            pw = RobustPWRegression(solver="scs", degree=degree,
                                    monotonic_trend=monotonic_trend)
            pw.fit(x, y, splits, lb=5, ub=50)
            pred = pw.predict(x)
            assert np.all((5 <= pred) & (pred <= 50))


def test_quantile():
    x = X[:, -1]

    # Monotonic trend: descending
    for degree in (0, 1, 2):
        pw = RobustPWRegression(
            solver="ecos", objective="l2", degree=degree,
            monotonic_trend="descending")
        pw.fit(x, y, splits="quantile")

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3


def test_uniform():
    x = X[:, -1]

    # Monotonic trend: descending
    for degree in (0, 1, 2):
        pw = RobustPWRegression(
            solver="ecos", objective="l2", degree=degree,
            monotonic_trend="descending")
        pw.fit(x, y, splits="uniform", n_bins=3)

        pred = pw.predict(np.sort(x))
        diff = np.max(pred[1:] - pred[:-1])
        assert diff <= 1e-3


def test_predict_not_fitted():
    pw = RobustPWRegression()

    with raises(NotFittedError):
        pw.predict(x)


def test_fit_predict():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    pw = RobustPWRegression()
    pred1 = pw.fit_predict(x, y, splits)

    pw.fit(x, y, splits)
    pred2 = pw.predict(x)

    assert pred1 == approx(pred2, rel=1e-8)


def test_predict_bounds():
    splits = [5, 10, 15, 20]
    x = X[:, -1]

    pw = RobustPWRegression(extrapolation_bounds=(5, 50))
    pw.fit(x, y, splits)

    pred = pw.predict(x)
    np.all((5 <= pred) & (pred <= 50))


def test_interpolation_linear():
    x = [0, 2.,   3.,  3.9, 5.,  7,  10]
    y = [1, 0.92, 0.9, 0.8, 0.7, 0.6, 0.5]

    pw = RobustPWRegression(degree=2, solver="osqp",
                            monotonic_trend="descending",
                            continuous_deriv=True,
                            extrapolation="continue",
                            extrapolation_bounds=(0, 1))

    pw.fit(x, y, splits=x)
    assert pw.predict(np.array([12])) == approx(0.49619792, rel=1e-6)

    pw = RobustPWRegression(degree=2, solver="ecos",
                            monotonic_trend="descending",
                            continuous_deriv=True,
                            extrapolation="linear",
                            extrapolation_bounds=(0, 1))

    pw.fit(x, y, splits=x)
    assert pw.predict(np.array([12])) == approx(0.55558441, rel=1e-6)


def test_interpolation_log():
    x = [0,  2.,   3.,  3.9, 5.,  7,  10]
    y = [-1, 0.92, 0.9, 0.8, 0.7, 0.6, 0.5]

    pw = RobustPWRegression(degree=3, solver="ecos",
                            monotonic_trend="descending",
                            continuous_deriv=True,
                            extrapolation="continue",
                            extrapolation_bounds=(0, 1), space="log")

    with raises(ValueError):
        pw.fit(x, y, splits=x)

    y = [1, 0.92, 0.9, 0.8, 0.7, 0.6, 0.5]
    pw.fit(x, y, splits=x)
    assert pw.predict(np.array([12])) == approx(0.18302127, rel=1e-6)

    pw.fit(x, y, splits=x, lb=1e-4, ub=1.0)
    assert pw.predict(np.array([12])) == approx(0.4165408, rel=1e-6)


def test_extrapolation_none():
    x = [0, 2.,   3.,  3.9, 5.,  7,  10]
    y = [1, 0.92, 0.9, 0.8, 0.7, 0.6, 0.5]

    pw = RobustPWRegression(degree=2, solver="osqp",
                            monotonic_trend="descending",
                            continuous_deriv=True,
                            extrapolation=None)

    pw.fit(x, y, splits=x)

    with raises(ValueError):
        pw.predict(np.array([11, 12]))


def test_status():
    pw = RobustPWRegression()
    pw.fit(x, y, splits=[5, 10, 15, 20])

    assert pw.status == "optimal"


def test_stats():
    pw = RobustPWRegression()
    pw.fit(x, y, splits=[5, 10, 15, 20])

    assert pw.stats['n_variables'] == 10
    assert pw.stats['n_constraints'] == 14


def test_choose_method_auto():
    assert _choose_method(
        "l2", 1, True, None, "auto", False, None) == "lsq_direct"

    assert _choose_method(
        "l2", 1, False, None, "auto", False, None) == "lsq_direct_separated"

    assert _choose_method(
        "l2", 1, True, "descending", "auto", False, None) == "qp"

    assert _choose_method(
        "l2", 1, False, "descending", "auto", False, None) == "qp_separated"

    assert _choose_method(
        "l2", 2, True, "descending", "auto", False, None) == "socp"

    assert _choose_method(
        "l2", 2, False, "descending", "auto", False, None) == "socp_separated"

    assert _choose_method(
        "l1", 2, True, "descending", "auto", False, None) == "socp"

    assert _choose_method(
        "l1", 2, False, "descending", "auto", False, None) == "socp_separated"

    assert _choose_method(
        "l1", 2, True, "descending", "auto", False, "l1") == "socp"

    assert _choose_method(
        "l1", 2, False, "descending", "auto", False, "l2") == "socp_separated"
