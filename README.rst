=====
RoPWR
=====

**RoPWR** is a library written in Python implementing several mathematical programming formulations to compute the optimal continuous/discontinuous piecewise 
polynomial regression given a list of split points.

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