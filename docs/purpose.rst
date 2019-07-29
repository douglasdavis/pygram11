Purpose
=======

Motivation
----------

There are more than enough histogramming packages for Python out in
the wild, and some have *a lot* of functionality beyond dropping data
into bins.

- This library aims to provide a way to drop data into bins quickly
  with a simple (and hopefully easy to install) code base.
- A property of histograms lacking from other options is the ability
  to retrieve the sum of weights squared in each bin (it's possible in
  NumPy, but not directly from the histogramming functions). This
  calculation is a "first class citizen" in pygram11.
- pygram11 also provided an API for histogramming the same data with
  multiple weight variations in a single function call.
- Finally, I thought it would be fun to learn how to write software
  with OpenMP and pybind11 because I had not used either piece of
  software before. Hopefully having them in my tool belt will be
  useful for a more complex future project. I thought about trying
  this out while I was sitting waiting for :math:`\mathcal{O}(1000)`
  histograms to be generated (one of many steps in the workflow for my
  thesis analysis).

Some of the other options that you might find useful:

- `numpy.histogram
  <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html>`_:
  versatile but slow; doesn't handle sum of weights squared
- `fast-histogram <https://github.com/astrofrog/fast-histogram>`_:
  leverages NumPy's C API. Very fast (fixed bin only) histogramming
  and easy to install; no OpenMP support or sum of weights squared.
- `physt <https://github.com/janpipek/physt>`_: *way* more than just
  sorting data into bins.

Some Benchmarks
---------------

Here are a couple of benchmarks testing ``pygram11`` (labeled `pg11`)
against ``numpy.histogram`` (labeled `np`) and ``fast-histogram``
(labeled `fh`). These were performed on a MacBook Pro with a 2.6GHz 12
Core Intel i7 (6 hyperthreaded cores) processor.

The :math:`y`-axis is the ratio of the times to complete the
calculation from two different packages; the higher the ratio, the
faster the denominator.

The first plot shows use of pygram11 without OpenMP acceleration. For
the second plot, we turn on OpenMP acceleration.

.. image:: /_static/compare.png
   :width: 70%
   :align: center

.. image:: /_static/compare_omp.png
   :width: 70%
   :align: center

Without OpenMP, fast-histogram outperforms pygram11 across the
board. With OpenMP, pygram11 starts to outperform fast-histogram when
the array size exceeds 10,000 entries. At 1,000,000 entries, pygram11
appears to be up to 3x faster than fast-histogram, and 100x faster
than numpy. For very small arrays, the overhead to spin up the
parallel loops via OpenMP is observable.

For variable width binning we just compare pygram11 to NumPy:

.. image:: /_static/compare_var.png
   :width: 70%
   :align: center

.. image:: /_static/compare_var_omp.png
   :width: 70%
   :align: center

Here we see pygram11 is always useful, and OpenMP becomes helpful for
arrays exceeding about 1,000 entries. For large arrays we approach
speeds 40x(10x) faster than NumPy in one(two)-dimension(s) with OpenMP enabled.
