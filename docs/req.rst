Installation
============

Requirements
------------

Hard Requirments
^^^^^^^^^^^^^^^^

- NumPy_
- pybind11_ (and therefore a C++11 compiler; if using ``pip``, must be
  explicitly installed before pygram11).

Soft Requirements
^^^^^^^^^^^^^^^^^

- OpenMP_ (will be used if installed from conda-forge).

You can use pygram11 without OpenMP, but you might want to try
`fast-histogram <https://github.com/astrofrog/fast-histogram>`_ if you
just need to compute fixed bin histograms with OpenMP (see `the
benchmarks <purpose.html#some-benchmarks>`__). If you're here for
variable width histograms or the sum-of-weights-squared first class
citizenry - I think you'll still find pygram11 useful.


Install Options
---------------

conda-forge
^^^^^^^^^^^

Installations from conda-forge provide a build that used OpenMP.

.. code-block:: none

   $ conda install -c conda-forge pygram11

PyPI
^^^^

pybind11 must be installed explicitly before pygram11.

.. code-block:: none

   $ pip install pybind11 numpy
   $ pip install pygram11

.. note::

   If installing from PyPI in an Anaconda environment on macOS 10.14
   with Xcode 10.2 command line tools (Apple LLVM version 10.0.1), you
   must specify the deployment target:

.. code-block:: none

   $ MACOSX_DEPLOYMENT_TARGET=10.14.4 pip install pygram11


Source
^^^^^^

pybind11 must be installed explicitly before pygram11.

.. code-block:: none

   $ git clone https://github.com/douglasdavis/pygram11
   $ pip install pybind11 numpy
   $ cd pygram11
   $ pip install .

In macOS Anaconda environments see the above note related to PyPI
installations.

.. _pybind11: https://github.com/pybind/pybind11
.. _NumPy: http://www.numpy.org/
.. _OpenMP: https://www.openmp.org/
