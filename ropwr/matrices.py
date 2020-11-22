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
    n = len(x)
    n_bins = len(splits) + 1
    indices = np.searchsorted(splits, x, side='right')

    D = np.zeros((n, n_bins * order))

    cn = 0
    for i in range(n_bins):
        xi = x[indices == i]
        ni = len(xi)

        qxi = np.ones(ni) / xi
        for k, j in enumerate(range(order * i, order * (i + 1))):
            D[cn: cn + ni, j] = k * qxi
            qxi *= xi

        cn += ni

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


def matrix_CC(splits):
    n_splits = len(splits)
    n_bins = n_splits + 1

    CC = np.zeros((n_splits, n_bins * 2))

    for i in range(n_splits):
        CC[i, [i * 2 + 1, i * 2 + 3]] = [-1, 1]

    return CC


def matrix_D0(splits):
    n_splits = len(splits)
    n_bins = n_splits + 1

    D = np.zeros((n_splits, n_bins))

    for i in range(n_splits):
        D[i, [i, i + 1]] = [-1, 1]

    return D


def submatrix_A(ni, xi, order):
    Ai = np.zeros((ni, order))
    pxi = np.ones(ni)
    for j in range(order):
        Ai[:, j] = pxi
        pxi *= xi

    return Ai


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
