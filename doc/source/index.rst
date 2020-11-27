.. ropwf documentation master file, created by
   sphinx-quickstart on Sat Nov 21 19:43:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RoPWR: Robust Piecewise Regression
==================================

The **RoPWR** library implements several mathematical programming formulations
to compute the optimal continuous/discontinuous piecewise polynomial regression
given a list of split points. It supports several monotonic constraints, 
objective functions and regularizations. The library is written in Python and
relies on cvxpy (ECOS and OSQP solvers) to solve the underlying optimization
problems. Other formulations are solver using a direct approach.


.. toctree::
   :maxdepth: 1

   installation
   release_notes
   api