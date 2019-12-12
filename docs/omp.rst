OpenMP Support
==============

If installing pygram11 from conda-forge or via a binary wheel from
PyPI, OpenMP acceleration should be available. If you don't want to
use conda-forge or you can't use a binary wheel, keep reading.

If you are building from source, the ``setup.py`` script tests to see
if OpenMP is available during installation. You can look at the
`builds.sr.ht continuous integration configuration files
<https://github.com/douglasdavis/pygram11/tree/master/.builds>`_ to
see how builds are defined (Linux only) to ensure OpenMP acceleration
is available. For macOS, take a look at the `GitHub actions
configuration files
<https://github.com/douglasdavis/pygram11/blob/master/.github/workflows/ci.yml>`_. We
rely on ``libomp`` from Homebrew.

To check if OpenMP was detected and used while compiling the extension
module ``pygram11._core``, try the following:

.. code-block:: python

   >>> import pygram11
   >>> pygram11.omp_available()
   True

Needless to say, if you see ``False`` OpenMP acceleration isn't
available.

The histogramming functions use a named argument (``omp``) for
requesting OpenMP usage. If ``pygram11.omp_available()`` is ``False``
the ``omp`` function argument is ignored.

You can use ``pygram11.omp_max_threads()`` to check the number of
threads that OpenMP has determined are available on your CPU.
