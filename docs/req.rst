Requirements
============

Hard Requirments
----------------

- pybind11_ (and therefore a C++11 compiler). Must be explicitly
  installed before pygram11.
- NumPy_

Soft Requirements
-----------------

- OpenMP_

You can use pygram11 without OpenMP, but you might want to try
`fast-histogram <https://github.com/astrofrog/fast-histogram>`_ if you
just need to compute fixed bin histograms with OpenMP (see `the
benchmarks <purpose.html#some-benchmarks>`__). If you're here for
variable width histograms or the sum-of-weights-squared first class
citizenry - I think you'll still find pygram11 useful.

.. _pybind11: https://github.com/pybind/pybind11
.. _NumPy: http://www.numpy.org/
.. _OpenMP: https://www.openmp.org/
