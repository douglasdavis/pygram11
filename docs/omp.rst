OpenMP Support
==============

The ``setup.py`` script tests to see of OpenMP is available during
installation. Unit tests executed by continuous integration currently
do not test OpenMP.

Three methods are manually tested for OpenMP support before releases:

- Arch Linux: system python (3.7.2) with ``extra/openmp`` installed (GCC
  8.2).
- macOS 10.14: Homebrew python3 (3.7.2) with ``libomp`` installed from
  Homebrew (Apple LLVM version 10.0.0).
- macOS 10.14: Anaconda python3 (3.6.8 and 3.7.2) and python2 (2.7.15)
  distributions with ``libomp`` installed from Homebrew (you'll likely
  need to remove the extra ``libiomp5.dylib`` from the Anaconda
  environment ``lib`` folder or ``conda install nomkl``, see
  `here <https://github.com/dmlc/xgboost/issues/1715>`_).

Ubuntu 18.04 (with GCC 7.3 and system python 3.6.7) has also been
reported to successfully build pygram11 with OpenMP, but this is not
tested.

To check if OpenMP was detected and used while compiling the extension
module ``pygram11._core``, try the following:

.. code-block:: python

   >>> import pygram11
   >>> pygram11.OPENMP
   True

Needless to say, if you see ``False`` OpenMP acceleration isn't
available.

The histogramming functions use a named argument for requesting OpenMP
usage. If ``pygram11.OPENMP`` is ``False`` then the argument is ignored by
the C++ code.
