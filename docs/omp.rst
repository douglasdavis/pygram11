OpenMP Support
==============

If installing pygram11 from conda-forge, OpenMP should work. If using
PyPI or a source release, the ``setup.py`` script tests to see of
OpenMP is available during installation.

Three platforms have been manually tested for OpenMP support (to not
worry about this, just use conda-forge):

- Arch Linux: system python (3.7.2) and GCC (8.2).
- Debian Buster: system python3 (3.7.2) with GCC 8.3 and ``libgomp1``
  installed.
- macOS 10.14: Homebrew python3 (3.7.2) with ``libomp`` installed from
  Homebrew (Apple LLVM version 10.0.0). Also (**not conda-forge**)
  Anaconda python3 (3.6.8 and 3.7.2) and python2 (2.7.15)
  distributions with ``libomp`` installed from Homebrew (you'll likely
  need to remove the extra ``libiomp5.dylib`` from the Anaconda
  environment ``lib`` folder or ``conda install nomkl``, see `here
  <https://github.com/dmlc/xgboost/issues/1715>`_).

To check if OpenMP was detected and used while compiling the extension
module ``pygram11._core``, try the following:

.. code-block:: python

   >>> import pygram11
   >>> pygram11.OPENMP
   True

Needless to say, if you see ``False`` OpenMP acceleration isn't
available.

The histogramming functions use a named argument for requesting OpenMP
usage. If ``pygram11.OPENMP`` is ``False`` then the argument is
ignored by the C++ code.
