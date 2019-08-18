Quick Intro
-----------

Core pygram11 Functions
^^^^^^^^^^^^^^^^^^^^^^^

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

   >>> h, __ = pygram11.fix1d(np.random.randn(10000), bins=25, range=(-3, 3))

See the API reference for more examples.


NumPy-like Functions
^^^^^^^^^^^^^^^^^^^^

For convenience a NumPy-like API is also provided (**not one-to-one**,
see the API reference).

.. autosummary::
   pygram11.histogram
   pygram11.histogram2d
