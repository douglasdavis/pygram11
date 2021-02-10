.. pygram11 documentation master file; you can adapt this file
   completely to your liking, but it should at least contain the root
   `toctree` directive.

pygram11
========

pygram11 is a small Python library for creating simple histograms
quickly. The backend is written in C++14 with some help from pybind11_
and accelerated with OpenMP_.

.. image:: https://img.shields.io/conda/vn/conda-forge/pygram11.svg?colorB=486b87&style=flat
   :target: https://anaconda.org/conda-forge/pygram11
   :alt: conda-forge
.. image:: https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat
   :target: https://pypi.org/project/pygram11/
   :alt: PyPI
.. image:: https://img.shields.io/pypi/pyversions/pygram11
   :target: https://pypi.org/project/pygram11
   :alt: PyPI - Python Version
.. image:: https://img.shields.io/github/stars/douglasdavis/pygram11?style=social
   :target: https://github.com/douglasdavis/pygram11
   :alt: GitHub stars

.. _OpenMP: https://www.openmp.org/
.. _pybind11: https://github.com/pybind/pybind11

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   req.rst
   quick.rst
   bench.rst
   api.rst
