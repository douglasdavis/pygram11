Overview
========

Jumping In
----------

The main purpose of pygram11 is to be a faster `near` drop-in
replacement of :py:func:`numpy.histogram` and
:py:func:`numpy.histogram2d` with support for uncertainties. The NumPy
functions always return the bin counts and the bin edges, while
pygram11 functions return the bin counts and the standard error on the
bin counts (if weights are not used, the second return type from
pygram11 functions will be ``None``). Therefore, if one only cares
about the bin counts, the libraries are completely interchangable.

These two funcion calls will provide the same result::

  import numpy as np
  import pygram11 as pg
  rng = np.random.default_rng(123)
  x = rng.standard_normal(10000)
  counts1, __ = np.histogram(x, bins=20, range=(-3, 3))
  counts2, __ = pg.histogram(x, bins=20, range=(-3, 3))
  np.testing.assert_allclose(counts1, counts2)

If one cares about the statistical uncertainty on the bin counts, or
the ability to retain under- and over-flow counts, then pygram11 is a
great replacement. Checkout a `blog post
<https://ddavis.io/posts/numpy-histograms/>`_ which describes how to
recreate this behavior in pure NumPy, while pygram11 is as simple as::

  data = rng.standard_normal(10000)
  weights = rng.uniform(0.1, 0.9, x.shape[0])
  counts, err = pg.histogram(data, bins=10, range=(-3, 3), weights=weights, flow=True)

The :py:func:`pygram11.histogram` and :py:func:`pygram11.histogram2d`
functions in the pygram11 API are meant to provide an easy transition
from NumPy to pygram11. The next couple of sections summarize the
structure of the pygram11 API.

Core pygram11 Functions
-----------------------

pygram11 provides a simple set of functions for calculating histograms:

.. autosummary::

   pygram11.fix1d
   pygram11.fix1dmw
   pygram11.var1d
   pygram11.var1dmw
   pygram11.fix2d
   pygram11.var2d

You'll see that the API specific to pygram11 is a bit more specialized
than the NumPy histogramming API (shown below).

Histogramming a normal distribution:

.. code-block:: python

   >>> rng = np.random.default_rng(123)
   >>> h, __ = pygram11.fix1d(rng.standard_normal(10000), bins=25, range=(-3, 3))

See the API reference for more examples.

NumPy-like Functions
--------------------

For convenience a NumPy-like API is also provided (**not one-to-one**,
see the API reference).

.. autosummary::

   pygram11.histogram
   pygram11.histogram2d

Supported Types
---------------

Conversions between NumPy array types can take some time when
calculating histograms.

.. code-block:: ipython

   In [1]: import numpy as np

   In [2]: import pygram11 as pg

   In [3]: rng = np.random.default_rng(123)

   In [4]: x = rng.standard_normal(2_000_000)

   In [5]: %timeit pg.histogram(x, bins=30, range=(-4, 4))
   1.95 ms ± 138 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

   In [6]: %timeit pg.histogram(x.astype(np.float32), bins=30, range=(-4, 4))
   2.33 ms ± 170 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

You can see the type conversion increases this calculation time by
about 20%. The back-end C++ functions prohibit type conversions of the
input data. If an array with an unsupported :py:class:`numpy.dtype` is
passed to pygram11, a :py:class:`TypeError` will be rasied. Supported
:py:class:`numpy.dtype`'s for data are:

- :py:class:`numpy.float64` (a C/C++ ``double``)
- :py:class:`numpy.int64` (a C/C++ ``int64_t``)
- :py:class:`numpy.uint64` (a C/C++ ``uint64_t``)
- :py:class:`numpy.float32` (a C/C++ ``float``)
- :py:class:`numpy.int32` (a C/C++ ``int32_t``
- :py:class:`numpy.uint32` (a C/C++ ``uint32_t``)

and for weights:

- :py:class:`numpy.float64`
- :py:class:`numpy.float32`

OpenMP Configuration
--------------------

For small datasets OpenMP acceleration introduces unncessary overhead.
Or, if you're using the pygram11 API in cluster workflows (like with
Dask_), you have your threads committed to higher level abstractions.

By default, the C++ back-end utilizes OpenMP parallel loops if the
data size is above a threshold for a respective histogramming
situation. These thresholds are 10,000 for fixed width histograms and
5,000 for variable width histograms. The thresholds can be configured
in a granular way with the ``pygram11.config`` module.

The parameters are:

- ``"thresholds.fix1d"``
- ``"thresholds.fix1dmw"``
- ``"thresholds.fix2d"``
- ``"thresholds.var1d"``
- ``"thresholds.var1dmw"``
- ``"thresholds.var2d"``

Low level reading/writing is handled through two functions:

.. autosummary::

   pygram11.config.get
   pygram11.config.set

If you have specific thresholds in mind,
:py:func:`pygram11.config.set` is the recommended interface.

The recommended entry points for controlling OpenMP acceleration in an
on/off switch way are through the provided context managers and
decorators (if we want to force OpenMP acceleration, we set the
thresholds to zero; if we want to disable OpenMP acceleration, we set
the thresholds to `sys.maxsize`).

.. autosummary::

   pygram11.omp_disabled
   pygram11.omp_forced
   pygram11.without_omp
   pygram11.with_omp

The context manager and decorator APIs provide an interface that
executes *temporary* adjustments to the thresholds that live during
specific code blocks or for entire function calls. For example, we can
disable a specific threshold during a :py:func:`pygram11.histogram`
call with the :py:func:`pygram11.omp_disabled` context manager:

.. code-block:: python

   import pygram11
   import numpy as np

   rng = np.random.default_rng(123)
   x = rng.standard_normal(50_000)
   with omp_disabled(key="thresholds.fix1d"):
       result = pygram11.histogram(x, bins=50, range=(-3, 3))

or we can decorate a function to disable OpenMP during its use:

.. code-block:: python

   import pygram11
   import numpy as np

   @pygram11.without_omp
   def hist():
       rng = np.random.default_rng(123)
       x = rng.standard_normal(50_000)
       return pygram11.histogram(x, bins=50, range=(-3, 3))


If the `key` argument is not provided, all thresholds will be
temporarily modified.

An example of threshold modification via the granular interface:

.. code-block:: python

   >>> import pygram11
   >>> import pygram11.config
   >>> import numpy as np
   >>> rng = np.random.default_rng(123)
   >>> x = rng.standard_uniform(6000)
   >>> bins = np.array([-3.1, -2.5, -2.0, 0.1, 0.2, 2.1, 3.0])
   >>> result = pygram11.histogram(x, bins=bins)  # will use OpenMP
   >>> pygram11.config.set("thresholds.var1d", 7500)
   >>> result = pygram11.histogram(x, bins=bins)  # now will _not_ use OpenMP

Some shortcuts exist to completely disable or enable OpenMP, along
with returning to the defaults:

- :py:func:`pygram11.disable_omp`: maximizes all thresholds so OpenMP
  will never be used.
- :py:func:`pygram11.force_omp`: zeros all thresholds so OpenMP will
  always be used.
- :py:func:`pygram11.default_omp`: return to default thresholds.


.. _Dask: https://dask.org
