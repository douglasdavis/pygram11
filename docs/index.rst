.. pygram11 documentation master file, created by
   sphinx-quickstart on Wed Jan 30 23:40:24 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pygram11
========

pygram11 is a small Python library for creating simple histograms
quickly. The backend is written in C++11 (with help from pybind11_)
and (optionally) accelerated with OpenMP_.


.. image:: https://img.shields.io/conda/vn/conda-forge/pygram11.svg?colorB=486b87&style=flat
   :target: https://anaconda.org/conda-forge/pygram11
   :alt: conda-forge
.. image:: https://img.shields.io/pypi/v/pygram11.svg?colorB=486b87&style=flat
   :target: https://pypi.org/project/pygram11/
   :alt: PyPI
.. image:: https://zenodo.org/badge/168767581.svg
   :target: https://zenodo.org/badge/latestdoi/168767581
   :alt: DOI
.. image:: https://img.shields.io/github/stars/douglasdavis/pygram11?style=social
   :target: https://github.com/douglasdavis/pygram11
   :alt: GitHub stars


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
