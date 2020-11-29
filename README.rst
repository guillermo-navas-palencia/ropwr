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
ropwr requires

* cvxpy (>=1.0)
* numpy (>=1.16)
* scikit-learn (>=0.22)