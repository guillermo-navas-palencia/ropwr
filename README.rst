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

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import load_boston

   data = load_boston()
   df = pd.DataFrame(data.data, columns=data.feature_names)

   x = df["NOX"].values
   y = data.target


.. code-block:: python

   from sklearn.preprocessing import KBinsDiscretizer
   from ropwr import RobustPWRegression

   est = KBinsDiscretizer(n_bins=10, strategy="quantile")
   est.fit(x.reshape(-1, 1), y)
   splits = est.bin_edges_[0][1:-1]

   pw = RobustPWRegression(objective="l2", degree=1, monotonic_trend=None)
   pw.fit(x, y, splits)

.. image:: doc/source/_images/pw_default.png
   :target: doc/source/_images/pw_default.png


.. code-block:: python

   pw = RobustPWRegression(objective="l1", degree=1, monotonic_trend="convex")
   pw.fit(x, y, splits)

.. image:: doc/source/_images/pw_convex.png
   :target: doc/source/_images/pw_convex.png


.. code-block:: python

   pw = RobustPWRegression(objective="l1", degree=1, monotonic_trend="valley")
   pw.fit(x, y, splits)

.. image:: doc/source/_images/pw_valley.png
   :target: doc/source/_images/pw_valley.png
