"""
Auxiliary functions to create model matrices.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np

from numpy.polynomial.polynomial import polyder


def matrix_A(x, splits, order):
    n = len(x)
    n_bins = len(splits) + 1

    indices = np.searchsorted(splits, x, side='right')

    A = np.zeros((n, n_bins * order))

    cn = 0
    for i in range(n_bins):
        xi = x[indices == i]
        ni = len(xi)

        pxi = np.ones(ni)
        for j in range(order * i, order * (i + 1)):
            A[cn: cn + ni, j] = pxi
            pxi *= xi

        cn += ni

    return A


def matrix_S(x, splits, order, continuous_deriv):
    n_splits = len(splits)
    n_bins = n_splits + 1

    S = np.zeros((n_splits, n_bins * order))

    exporder = np.arange(order)
    for i, s in enumerate(splits):
        r = np.power(s, exporder)
        S[i, order * i: order * (i + 1)] = r
        S[i, order * (i + 1): order * (i + 2)] = -r

    if continuous_deriv and order >= 3:
        for m in range(1, order - 1):
            SD = np.zeros((n_splits, n_bins * order))
            d = np.zeros(order)
            d[m:] = polyder(np.ones(order), m)
            for i, s in enumerate(splits):
                rd = np.array([d[k] * s ** (k-m) for k in range(order)])
                SD[i, order * i: order * (i + 1)] = rd
                SD[i, order * (i + 1): order * (i + 2)] = -rd

            S = np.r_[S, SD]

    return S


def matrix_D(x, splits, order):
    n_splits = len(splits)
    n_bins = n_splits + 1

    if order == 1:
        D = np.zeros((n_splits, n_bins))

        for i in range(n_splits):
            D[i, [i, i + 1]] = [-1, 1]

    elif order == 2:
        D = np.zeros((n_bins * order, n_bins * order))

        for i in range(n_bins):
            D[i, i * 2 + 1] = 1

    return D


def matrix_A_D(x, splits, order):
    n = len(x)
    n_bins = len(splits) + 1

    indices = np.searchsorted(splits, x, side='right')

    nA = n_bins * order
    A = np.zeros((n, nA))
    D = np.zeros((n, nA))

    cn = 0
    for i in range(n_bins):
        xi = x[indices == i]
        ni = len(xi)

        pxi = np.ones(ni)
        qxi = np.ones(ni)
        for k, j in enumerate(range(order * i, order * (i + 1))):
            A[cn: cn + ni, j] = pxi
            if k <= 1:
                D[cn: cn + ni, j] = k
            else:
                qxi *= xi
                D[cn: cn + ni, j] = k * qxi
            pxi *= xi

        cn += ni

    return A, D


def matrix_H(x, splits, order):
    n_splits = len(splits)
    n_bins = n_splits + 1

    if order == 2 and n_bins > 1:
        H = np.zeros((n_splits, n_bins * 2))

        for i in range(n_splits):
            H[i, [i * 2 + 1, i * 2 + 3]] = [-1, 1]
    else:
        n = len(x)
        indices = np.searchsorted(splits, x, side='right')
        H = np.zeros((n, n_bins * order))

        cn = 0
        for i in range(n_bins):
            xi = x[indices == i]
            ni = len(xi)

            qxi = np.ones(ni)
            for k, j in enumerate(range(order * i, order * (i + 1))):
                if k <= 2:
                    H[cn: cn + ni, j] = k * (k - 1)
                else:
                    qxi *= xi
                    H[cn: cn + ni, j] = k * (k - 1) * qxi

            cn += ni

    return H


def matrix_A_H(x, splits, order):
    n = len(x)
    n_bins = len(splits) + 1

    indices = np.searchsorted(splits, x, side='right')

    nA = n_bins * order
    A = np.zeros((n, nA))
    H = np.zeros((n, nA))

    cn = 0
    for i in range(n_bins):
        xi = x[indices == i]
        ni = len(xi)

        pxi = np.ones(ni)
        qxi = np.ones(ni)
        for k, j in enumerate(range(order * i, order * (i + 1))):
            A[cn: cn + ni, j] = pxi
            if k <= 2:
                H[cn: cn + ni, j] = k * (k - 1)
            else:
                qxi *= xi
                H[cn: cn + ni, j] = k * (k - 1) * qxi
            pxi *= xi

        cn += ni

    return A, H


def submatrix_A(ni, xi, order):
    Ai = np.zeros((ni, order))
    pxi = np.ones(ni)
    for j in range(order):
        Ai[:, j] = pxi
        pxi *= xi

    return Ai


def submatrix_D(order):
    Di = np.zeros(order)
    Di[1] = 1

    return Di


def submatrix_A_D(ni, xi, order):
    Ai = np.zeros((ni, order))
    Di = np.zeros((ni, order))
    pxi = np.ones(ni)
    qxi = np.ones(ni)
    for j in range(order):
        Ai[:, j] = pxi
        if j <= 1:
            Di[:, j] = j
        else:
            qxi *= xi
            Di[:, j] = j * qxi
        pxi *= xi

    return Ai, Di
