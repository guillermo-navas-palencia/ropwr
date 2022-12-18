"""
Direct method for piecewise regression with l2-norm objective and no other
constraints.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from .matrices import matrix_A
from .matrices import matrix_S
from .matrices import submatrix_A


def lsq_direct(x, y, splits, degree, continuous_deriv):
    order = degree + 1
    n_splits = len(splits)
    n_bins = n_splits + 1

    A = matrix_A(x, splits, order)
    S = matrix_S(x, splits, order, continuous_deriv)

    nA = n_bins * order
    nM = nA + S.shape[0]
    M = np.zeros((nM, nM))

    M[:nA, :nA] = 2 * A.T.dot(A)
    M[:nA, nA:] = S.T
    M[nA:, :nA] = S

    d = np.zeros(nM)
    d[:nA] = 2 * A.T.dot(y)

    c = np.linalg.solve(M, d)

    info = {
        "status": "optimal",
        "stats": {"n_variables": nA, "n_constraints": nM}
    }

    return c[:nA].reshape((n_bins, order)), info


def lsq_direct_separated(x, y, splits, degree):
    order = degree + 1
    n_bins = len(splits) + 1

    indices = np.searchsorted(splits, x, side='right')

    c = np.zeros((n_bins, order))

    infos = []
    for i in range(n_bins):
        mask = (indices == i)
        xi = x[mask]
        yi = y[mask]
        ni = len(xi)
        Ai = submatrix_A(ni, xi, order)

        ci, _, _, _ = np.linalg.lstsq(Ai, yi, rcond=None)

        info = {
            "status": "optimal",
            "stats": {"n_variables": order, "n_constraints": 0}
        }
        infos.append(info)

        c[i, :] = ci

    return c, infos
