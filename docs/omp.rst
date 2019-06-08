OpenMP Support
==============

If installing pygram11 from conda-forge, OpenMP acceleration is
available. If using PyPI or a source release, the ``setup.py`` script
tests to see of OpenMP is available during installation. If you don't
want to use conda-forge, keep reading.

You can look at the `continuous integration configuration files
<https://github.com/drdavis/pygram11/tree/master/.builds>`_ to see how
builds are defined (Linux only) to ensure OpenMP acceleration is
available. For macOS 10.14, manual tests have shown these setups are
OpenMP accelerated:

- Homebrew or `pyenv <https://github.com/pyenv/pyenv>`_ Python3
  (3.7.*) with ``libomp`` installed from Homebrew (Apple LLVM version
  10.0.*). This is probably simplest non-conda-forge setup.
- Default (not conda-forge) Anaconda Python3 (3.6.8 and 3.7.3) and
  Python2 (2.7.16) distributions with ``libomp`` installed from
  Homebrew (you'll likely need to remove the extra ``libiomp5.dylib``
  from the Anaconda environment ``lib`` folder or ``conda install
  nomkl``, see `here <https://github.com/dmlc/xgboost/issues/1715>`_).

To check if OpenMP was detected and used while compiling the extension
module ``pygram11._core``, try the following:

.. code-block:: python

   >>> import pygram11
   >>> pygram11.OPENMP
   True

Needless to say, if you see ``False`` OpenMP acceleration isn't
available.

The histogramming functions use a named argument (``omp``) for
requesting OpenMP usage. If ``pygram11.OPENMP`` is ``False`` the
``omp`` function argument is ignored.
