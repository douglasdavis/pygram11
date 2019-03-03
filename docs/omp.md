# OpenMP Support

The `setup.py` script tests to see of OpenMP is available during
installation. Unit tests executed by continuous integration currently
do not test OpenMP.

Three methods are manually tested before releases:

- Arch Linux: System python with `extra/openmp` installed (GCC 8.2).
- macOS 10.14: Homebrew python3 with `libomp` installed from Homebrew
  (Apple LLVM version 10.0.0).
- macOS 10.14: Anaconda python3 and python2 distributions with
  `libomp` installed from Homebrew (you probably need to remove the
  extra `libiomp5.dylib` from the Anaconda environment `lib` folder or
  `conda install nomkl`, see
  [here](https://github.com/dmlc/xgboost/issues/1715)).

To check if OpenMP was detected and used, try the following:

```python
>>> import pygram11
>>> pygram11.OPENMP
True
```

Needless to say, if you see `False` OpenMP acceleration isn't
available.


The histogramming functions use a named argument for requesting OpenMP
usage. If `pygram11.OPENMP` is `False` then the argument is ignored by
the C++ code.
