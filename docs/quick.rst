Quick Start
===========

Jumping In
----------

The main purpose of pygram11 is to be a faster `near` drop-in
replacement of :py:func:`numpy.histogram` and
:py:func:`numpy.histogram2d`. The NumPy functions always return the
bin counts and the bin edges, while pygram11 functions return the bin
counts and the standard error on the bin counts (if weights are not
used, the second return type from pygram11 functions will be
``None``). Therefore, if one only cares about the bin counts, the
libraries are completely interchangable.

These two funcion calls will provide the same result::

  import numpy as np
  import pygram11 as pg
  rng = np.random.default_rng(123)
  x = rng.standard_normal(10000)
  counts, __ = np.histogram(x, bins=20, range=(-3, 3))
  counts, __ = pg.histogram(x, bins=20, range=(-3, 3))

If one cares about the standard error on the bin counts, or the
ability to retain under- and over-flow counts, then pygram11 is a
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

OpenMP Configuration
--------------------

For small datasets OpenMP acceleration introduced unncessary overhead.
The C++ backend utilizes OpenMP parallel loops if the data size is
above a threshold for a respective histogramming situation. By default
these thresholds are 10,000 for fixed width histograms and 5,000 for
variable width histograms. The thresholds can be configured with
dynamic variables in the ``pygram11`` module:

- ``FIXED_WIDTH_PARALLEL_THRESHOLD_1D``
- ``FIXED_WIDTH_PARALLEL_THRESHOLD_2D``
- ``FIXED_WIDTH_MW_PARALLEL_THRESHOLD_1D``
- ``VARIABLE_WIDTH_PARALLEL_THRESHOLD_1D``
- ``VARIABLE_WIDTH_PARALLEL_THRESHOLD_2D``
- ``VARIABLE_WIDTH_MW_PARALLEL_THRESHOLD_1D``

An example changing the threshold:

.. code-block:: python

   >>> import pygram11
   >>> import numpy as np
   >>> rng = np.random.default_rng(123)
   >>> x = rng.standard_uniform(6000)
   >>> bins = np.array([-3.1, -2.5, -2.0, 0.1, 0.2, 2.1, 3.0])
   >>> result = pygram11.histogram(x, bins=bins)  # will use OpenMP
   >>> pygram11.VARIABLE_WIDTH_PARALLEL_THRESHOLD_1D = 7500
   >>> result = pygram11.histogram(x, bins=bins)  # now will _not_ use OpenMP

Some shortcut functions exist to blanket enable/disable OpenMP:

- :py:func:`pygram11.disable_omp`: maximizes all thresholds so OpenMP
  will never be used.
- :py:func:`pygram11.force_omp`: zeros all thresholds so OpenMP will
  always be used.
