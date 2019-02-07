# OpenMP Support

The `setup.py` script tests to see of OpenMP is available during
installation. The logic is not incredibly robust at the moment.

Three methods have been tested:

- Arch Linux: System python with `extra/openmp` installed.
- macOS: Homebrew python3 with `libomp` installed from Homebrew
- macOS: Anaconda python3 and python2 distributions with `libomp`
  installed from Homebrew (you probably need to remove the extra
  `libiomp5.dylib` from the Anaconda environment `lib` folder or
  `conda install nomkl`, see
  [here](https://github.com/dmlc/xgboost/issues/1715)).

```python
>>> import pygram11
>>> pygram11.OPENMP
True
```

I plan to make the flag toggle-able in the future to turn off OpenMP
usage even if the package was installed with OpenMP support.
