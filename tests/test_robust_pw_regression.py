"""
RobustPWRegression testing.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2020

import numpy as np
import pandas as pd

from pytest import approx, raises

from ropwr import RobustPWRegression
from sklearn.datasets import load_boston
from sklearn.exceptions import NotFittedError


data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)

variable = "LSTAT"
x = df[variable].values
y = data.target


def test_params():
    pass


def test_splits():
    pass


def test_bounds():
    pass


def test_continuous():
    pass


def test_discontinuous():
    pass