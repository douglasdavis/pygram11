Quick Start
===========

Jumping In
----------

The main purpose of pygram11 is to be a faster `near` drop-in
replacement of :py:func:`numpy.histogram` and
:py:func:`numpy.histogram2d`. The NumPy functions always return the bin
counts and the bin edges, while pygram11 functions return the bin
counts and the standard error on the bin counts. Therefore, if one
only cares about the bin counts, the libraries are completely
interchangable.

These two funcion calls will provide the same result::

  import numpy as np
  import pygram11 as pg
  counts, __ = np.histogram(np.random.randn(1000), bins=20, range=(-3, 3))
  counts, __ = pg.histogram(np.random.randn(1000), bins=20, range=(-3, 3))

If one cares about the standard error on the bin counts, or the
ability to retain under- and over-flow counts, then pygram11 is a
great replacement. Checkout a `blog post
<https://ddavis.io/posts/numpy-histograms/>`_ which describes how to
recreate this behavior in pure NumPy, while pygram11 is as simple as::

  data = np.random.randn(1000)
  weights = np.random.uniform(0.5, 0.8, x.shape[0])
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

   >>> h, err = pygram11.fix1d(np.random.randn(10000), bins=25, range=(-3, 3))

See the API reference for more examples.


NumPy-like Functions
--------------------

For convenience a NumPy-like API is also provided (**not one-to-one**,
see the API reference).

.. autosummary::

   pygram11.histogram
   pygram11.histogram2d
