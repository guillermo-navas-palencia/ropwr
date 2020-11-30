"""
Auxiliary functions to create model matrices.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np


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


def matrix_S(x, splits, order):
    n_splits = len(splits)
    n_bins = n_splits + 1

    S = np.zeros((n_splits, n_bins * order))

    exporder = np.arange(order)
    for i, s in enumerate(splits):
        r = np.power(s, exporder)
        S[i, order * i: order * (i + 1)] = r
        S[i, order * (i + 1): order * (i + 2)] = -r

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
        qxi = pxi / xi
        for k, j in enumerate(range(order * i, order * (i + 1))):
            A[cn: cn + ni, j] = pxi
            D[cn: cn + ni, j] = k * qxi
            pxi *= xi
            qxi *= xi

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

            qxi = np.ones(ni) / (xi * xi)
            for k, j in enumerate(range(order * i, order * (i + 1))):
                H[cn: cn + ni, j] = k * (k - 1) * qxi
                qxi *= xi

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
        qxi = pxi / (xi * xi)
        for k, j in enumerate(range(order * i, order * (i + 1))):
            A[cn: cn + ni, j] = pxi
            H[cn: cn + ni, j] = k * (k - 1) * qxi
            pxi *= xi
            qxi *= xi

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
    qxi = pxi / xi
    for j in range(order):
        Ai[:, j] = pxi
        Di[:, j] = j * qxi
        pxi *= xi
        qxi *= xi

    return Ai, Di
