from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import tempfile
import os
import sys
import setuptools


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "pygram11._core",
        [os.path.join("pygram11", "_core.cpp")],
        include_dirs=[get_pybind_include()],
        language="c++",
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def has_omp(compiler):
    """Check if omp available"""
    if sys.platform == "darwin":
        xpa = ["-Xpreprocessor", "-fopenmp"]
    else:
        xpa = ["-fopenmp"]
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("#include <omp.h>")
        f.write("int main() (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_preargs=xpa, extra_postargs=["-lomp"])
        except setuptools.distutils.errors.CompileError:
            return false
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.10"]

    def build_extensions(self):
        use_omp = has_omp(self.compiler)

        if use_omp:
            if sys.platform == "darwin":
                self.c_opts["unix"].append("-Xpreprocessor")
            self.c_opts["unix"].append("-fopenmp")
            self.c_opts["unix"].append("-DPYGRAMUSEOMP")

        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            if use_omp:
                ext.extra_link_args = ["-lomp"]
                if sys.platform == "darwin":
                    ext.extra_link_args.append("-mmacosx-version-min=10.10")
        build_ext.build_extensions(self)


def get_version():
    g = {}
    exec(open(os.path.join("pygram11", "version.py")).read(), g)
    return g["__version__"]


setup(
    name="pygram11",
    version=get_version(),
    author="Doug Davis",
    author_email="ddavis@ddavis.io",
    url="https://github.com/drdavis/pygram11",
    description="Fast histogramming in python built on pybind11",
    long_description="Fast histogramming in python built on pybind11",
    packages=["pygram11"],
    ext_modules=ext_modules,
    install_requires=["numpy>=1.12", "pybind11>=2.2"],
    cmdclass={"build_ext": BuildExt},
    test_suite="tests",
    tests_require=["pytest>=3.0"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
    ],
)
