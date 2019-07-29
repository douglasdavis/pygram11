Installation
============

Requirements
------------

Hard Requirments
^^^^^^^^^^^^^^^^

- NumPy_

Soft Requirements
^^^^^^^^^^^^^^^^^

- OpenMP_

If you install binaries from conda-forge or PyPI, OpenMP acceleration
should be available.

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

   $ conda install pygram11 -c conda-forge

PyPI
^^^^

.. code-block:: none

   $ pip install pygram11


Source
^^^^^^

.. code-block:: none

   $ pip install git+https://github.com/douglasdavis/pygram11.git@master

.. note::

   If installing from source on macOS 10.14 you might have to prepend
   the ``pip`` command with ``MACOS_DEPLOYMENT_TARGET=10.14``.

.. code-block:: none

   $ MACOSX_DEPLOYMENT_TARGET=10.14 pip install ...


.. _pybind11: https://github.com/pybind/pybind11
.. _NumPy: http://www.numpy.org/
.. _OpenMP: https://www.openmp.org/
