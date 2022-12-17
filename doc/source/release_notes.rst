Release Notes
=============

Version 1.0.0 (2022-12-18)
--------------------------

New features:

   - Implement continuous derivative at split points (`Issue 7 <https://github.com/guillermo-navas-palencia/ropwr/issues/7>`_).
   - Implement extrapolation methods.
   - Add parameter ``max_iter`` to control the maximum number of iterations for solvers.
   - Add y-space linear or logarithmic transformation via parameter ``space``.

Improvements

   - Improve formulation for lower and upper bound constraints when monotonicity constraints are actived.
   - Improve computation of problem matrices.


Version 0.4.0 (2022-10-25)
--------------------------

New features:

   - Add support to various split methods.

Bugfixes:

   - Handle solver 'auto' for discontinuous SOCP model
   - Solve OSQP convergence issues using ``psd_wrap`` (`Issue 1424 <https://github.com/cvxpy/cvxpy/issues/1424>`_).


Version 0.3.0 (2022-09-20)
--------------------------

New features:

   - Add support to solver SCS and HIGHS.


Version 0.2.0 (2021-04-25)
--------------------------

Improvements:

   - Use @ operator for CVXPY matrix-vector multiplication.

   - Remove support to Python 3.6.


Version 0.1.0 (2020-12-02)
--------------------------

* First release of RoPWR.
