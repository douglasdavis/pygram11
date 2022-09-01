Installation
============

Requirements
------------

The only requirement to use pygram11 is NumPy_. If you install
binaries from conda-forge or PyPI, NumPy will be installed as a
required dependency.

Extras for Source Builds
^^^^^^^^^^^^^^^^^^^^^^^^

When building from source, all you need is a C++ compiler with C++14
support. The ``setup.py`` script will test to see if OpenMP is
available. If it's not, then the installation will abort. Most Linux
distributions with modern GCC versions should provide OpenMP
automatically (search the web to see how to install OpenMP from your
distribution's package manager). On macOS you'll want to install
``libomp`` from Homebrew to use OpenMP with the Clang compiler shipped
by Apple.

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

Source
^^^^^^

.. code-block:: none

   $ pip install git+https://github.com/douglasdavis/pygram11.git@main

.. _pybind11: https://github.com/pybind/pybind11
.. _NumPy: http://www.numpy.org/
.. _OpenMP: https://www.openmp.org/
