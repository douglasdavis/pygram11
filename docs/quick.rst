Quick Intro
-----------

pygram11 currently provides four simple functions for histogramming.

.. autosummary::
   pygram11.fix1d
   pygram11.var1d
   pygram11.fix2d
   pygram11.var2d

You'll see there are a number of types of histograms supported by
pygram11. For each of the four generic types, one can build a weighted
or unweighted histogram. The weighted histograms always return the sum
of weights squared in each bin.

Histogramming a normal distribution:

.. code-block:: python

   >>> h = pygram11.fix1d(np.random.randn(10000), bins=25, range=(-3, 3))

See the API reference for more examples.
