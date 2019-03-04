.. pygram11 documentation master file, created by
   sphinx-quickstart on Wed Jan 30 23:40:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pygram11
========

.. image:: https://builds.sr.ht/~ddavis/pygram11.svg
   :target: https://builds.sr.ht/~ddavis/pygram11?
   :alt: builds.sr.ht Status
.. image:: https://readthedocs.org/projects/pygram11/badge/?version=stable
   :target: https://pygram11.readthedocs.io/en/stable/?badge=stable
   :alt: Documentation Status
.. image:: https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat
   :target: https://pypi.org/project/pygram11/
   :alt: PyPI
.. image:: https://img.shields.io/pypi/pyversions/pygram11.svg?colorB=blue&style=flat
   :alt: Python Versions
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
   :alt: Code style: black


pygram11 is a small Python library for creating simple histograms
quickly. The backend is written in C++11 (with help from pybind11_)
and (optionally) accelerated with OpenMP_.


.. _OpenMP: https://www.openmp.org/
.. _pybind11: https://github.com/pybind/pybind11


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   req.rst
   quick.rst
   purpose.rst
   omp.rst
   api_reference.rst
