Installation
============

Requirements
------------

The only requirement for pygram11 is NumPy_. All installation methods
will ensure that ``numpy`` is installed.

Extras for Source Builds
^^^^^^^^^^^^^^^^^^^^^^^^

When building from source, all you need is a C++ compiler with C++11
support. The ``setup.py`` script will test to see if OpenMP is
available during compilation and linking of the backend extenstion
module. Most Linux distributions with relatively modern GCC versions
should provide OpenMP automatically (search the web to see how to
install OpenMP from your distribution's package manager). On macOS you
you'll want to install ``libomp`` from Homebrew to use OpenMP with the
Clang compiler shipped with macOS. If you install binaries from
conda-forge or PyPI, OpenMP acceleration is available.

You can use pygram11 without OpenMP, but you might want to try
`fast-histogram <https://github.com/astrofrog/fast-histogram>`_ if you
just need to compute fixed bin histograms (see `the benchmarks
<purpose.html#some-benchmarks>`__). If you're here for variable width
histograms or the sum-of-weights-squared first class citizenry or the
multiple weight variation histograms - I think you'll still find
pygram11 useful.


Install Options
---------------

PyPI
^^^^

.. code-block:: none

   $ pip install pygram11

conda-forge
^^^^^^^^^^^

Installations from conda-forge provide a build that used OpenMP.

.. code-block:: none

   $ conda install pygram11 -c conda-forge

.. note::

   On macOS the OpenMP libraries from LLVM (``libomp``) and Intel
   (``libiomp``) can clash if your ``conda`` environment includes the
   Intel Math Kernel Library (MKL) package distributed by
   Anaconda. You may need to install the ``nomkl`` package to prevent
   the clash (Intel MKL accelerates many linear algebra operations,
   but does not impact pygram11):

Source
^^^^^^

.. code-block:: none

   $ pip install git+https://github.com/douglasdavis/pygram11.git@master

.. note::

   If installing from source on macOS 10.14 you might have to prepend
   the ``pip`` command with ``MACOSX_DEPLOYMENT_TARGET=10.14``. (This
   may be required for Anaconda environments, because the Python
   distribution from Anaconda for macOS is built with an older macOS
   SDK).

.. code-block:: none

   $ MACOSX_DEPLOYMENT_TARGET=10.14 pip install ...


.. _pybind11: https://github.com/pybind/pybind11
.. _NumPy: http://www.numpy.org/
.. _OpenMP: https://www.openmp.org/
