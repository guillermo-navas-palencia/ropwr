=====
RoPWR
=====

.. image::  https://github.com/guillermo-navas-palencia/ropwr/workflows/CI/badge.svg
   :target: https://github.com/guillermo-navas-palencia/ropwr/workflows/CI/badge.svg

.. image::  https://img.shields.io/github/license/guillermo-navas-palencia/optbinning
   :target: https://img.shields.io/github/license/guillermo-navas-palencia/optbinning

The **RoPWR** library implements several mathematical programming formulations
to compute the optimal continuous/discontinuous piecewise polynomial regression
given a list of split points. It supports several monotonic constraints, 
objective functions and regularizations. The library is written in Python and
relies on cvxpy (ECOS and OSQP solvers) to solve the underlying optimization
problems. Other formulations are solved using a direct approach.

.. contents:: **Table of Contents**

Installation
============

To install the current release of ropwr from PyPI:

.. code-block:: text

   pip install ropwr

To install from source, download or clone the git repository

.. code-block:: text

   git clone https://github.com/guillermo-navas-palencia/ropwr.git
   cd ropwr
   python setup.py install

Dependencies
------------
RoPWR requires

* cvxpy (>=1.0)
* numpy (>=1.16)
* scikit-learn (>=0.22)

Getting started
===============

Please visit the RoPWR documentation (**current** release) http://gnpalencia.org/ropwr/. You can get started following the `tutorial <http://gnpalencia.org/ropwr/tutorial.html>`_ and checking the API reference.

Examples:
---------

To get us started, let’s load a well-known dataset from the UCI repository and transform the data into a ``pandas.DataFrame``.

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import load_boston

   data = load_boston()
   df = pd.DataFrame(data.data, columns=data.feature_names)

   x = df["NOX"].values
   y = data.target

To devise split points, we use the implementation of the unsupervised technique equal-size or equal-frequency interval implemented in scikit-learn ``KBinsDiscretizer``.

.. code-block:: python

   from sklearn.preprocessing import KBinsDiscretizer
   from ropwr import RobustPWRegression

   est = KBinsDiscretizer(n_bins=10, strategy="quantile")
   est.fit(x.reshape(-1, 1), y)
   splits = est.bin_edges_[0][1:-1]

.. code-block:: python

   pw = RobustPWRegression(objective="l2", degree=1, monotonic_trend=None)
   pw.fit(x, y, splits)

.. image:: doc/source/_images/pw_default.png
   :target: doc/source/_images/pw_default.png


.. code-block:: python

   pw = RobustPWRegression(objective="l1", degree=1, monotonic_trend="convex")
   pw.fit(x, y, splits)

.. image:: doc/source/_images/pw_convex.png
   :target: doc/source/_images/pw_convex.png


A regularization term using :math:`l_1−`norm (Lasso) :math:`l_2−`norm (Ridge) can be added to the objective function.

.. code-block:: python
   
   from sklearn.datasets import fetch_california_housing

   data = fetch_california_housing()

   target = "target"
   variable_names = data.feature_names
   df = pd.DataFrame(data.data, columns=variable_names)
   df[target] = data.target
   x = df["MedInc"].values
   y = df[target].values

   est = KBinsDiscretizer(n_bins=15, strategy="quantile")
   est.fit(x.reshape(-1, 1), y)
   splits = est.bin_edges_[0][1:-1]

   pw = RobustPWRegression(objective="l1", degree=1, monotonic_trend="valley")
   pw.fit(x, y, splits)

.. image:: doc/source/_images/pw_valley.png
   :target: doc/source/_images/pw_valley.png

.. code-block:: python

   pw = RobustPWRegression(objective="huber", monotonic_trend="ascending",
                           degree=2, regularization="l1", verbose=True)
   pw.fit(x, y, splits)

.. code-block:: text

   ECOS 2.0.7 - (C) embotech GmbH, Zurich Switzerland, 2012-15. Web: www.embotech.com/ECOS

   It     pcost       dcost      gap   pres   dres    k/t    mu     step   sigma     IR    |   BT
    0  +0.000e+00  -6.012e+03  +5e+05  8e-01  5e+00  1e+00  6e+00    ---    ---    2  1  - |  -  - 
    1  +7.934e+02  -3.304e+03  +3e+05  7e-01  3e-01  2e+00  4e+00  0.4820  5e-01   2  1  1 |  0  0
    2  +4.004e+03  +2.932e+03  +1e+05  3e-01  5e-02  8e-01  1e+00  0.7396  1e-01   2  1  1 |  0  0
    3  +6.368e+03  +5.536e+03  +9e+04  2e-01  4e-02  7e-01  1e+00  0.5427  6e-01   1  1  1 |  0  0
    4  +9.067e+03  +8.671e+03  +4e+04  1e-01  2e-02  3e-01  5e-01  0.5371  8e-02   1  2  1 |  0  0
    5  +1.043e+04  +1.022e+04  +2e+04  6e-02  2e-02  2e-01  3e-01  0.6971  4e-01   2  2  2 |  0  0
    6  +1.064e+04  +1.048e+04  +2e+04  5e-02  1e-02  1e-01  2e-01  0.9699  7e-01   2  2  1 |  0  0
    7  +1.216e+04  +1.212e+04  +4e+03  1e-02  7e-03  2e-02  5e-02  0.7909  3e-02   2  2  2 |  0  0
    8  +1.230e+04  +1.227e+04  +3e+03  7e-03  6e-03  2e-02  4e-02  0.4845  4e-01   1  2  1 |  0  0
    9  +1.254e+04  +1.253e+04  +8e+02  2e-03  2e-03  5e-03  1e-02  0.8206  1e-01   2  2  2 |  0  0
   10  +1.259e+04  +1.258e+04  +4e+02  1e-03  1e-03  2e-03  5e-03  0.5946  2e-01   2  1  2 |  0  0
   11  +1.262e+04  +1.262e+04  +2e+02  4e-04  5e-04  1e-03  2e-03  0.6943  2e-01   2  2  2 |  0  0
   12  +1.263e+04  +1.263e+04  +7e+01  2e-04  2e-04  3e-04  8e-04  0.9890  4e-01   2  1  1 |  0  0
   13  +1.264e+04  +1.264e+04  +9e+00  2e-05  3e-05  5e-05  1e-04  0.8925  3e-02   2  1  1 |  0  0
   14  +1.264e+04  +1.264e+04  +1e+00  3e-06  3e-06  6e-06  1e-05  0.8787  1e-02   2  1  1 |  0  0
   15  +1.264e+04  +1.264e+04  +3e-01  8e-07  9e-07  2e-06  4e-06  0.9890  3e-01   2  1  1 |  0  0
   16  +1.264e+04  +1.264e+04  +6e-02  1e-07  2e-07  3e-07  7e-07  0.8498  3e-02   2  1  1 |  0  0
   17  +1.264e+04  +1.264e+04  +2e-02  5e-08  6e-08  1e-07  3e-07  0.7942  2e-01   2  1  1 |  0  0
   18  +1.264e+04  +1.264e+04  +9e-03  2e-08  3e-08  5e-08  1e-07  0.7819  2e-01   2  1  1 |  0  0
   19  +1.264e+04  +1.264e+04  +1e-03  3e-09  4e-09  7e-09  2e-08  0.9584  1e-01   2  1  1 |  0  0
   20  +1.264e+04  +1.264e+04  +3e-05  7e-11  7e-11  1e-10  3e-10  0.9824  1e-04   2  1  1 |  0  0

   OPTIMAL (within feastol=7.0e-11, reltol=2.0e-09, abstol=2.6e-05).
   Runtime: 2.713925 seconds.

.. image:: doc/source/_images/pw_huber_reg_l1.png
   :target: doc/source/_images/pw_huber_reg_l1.png
