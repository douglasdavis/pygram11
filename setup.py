from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import glob
import tempfile
import os
import sys
import setuptools
import subprocess
from distutils import log
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler, get_config_var
from distutils.errors import CompileError, LinkError


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
        include_dirs=["/usr/local/include", get_pybind_include(), get_pybind_include(user=True)],
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


CCODE = """
#include <omp.h>
#include <stdio.h>
int main(void) {
  #pragma omp parallel
  printf("nthreads=%d\\n", omp_get_num_threads());
 return 0;
}
"""


def has_omp():
    """Check if omp available"""
    haveit = False

    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    if sys.platform == "darwin":
        compflags = ["-Xpreprocessor", "-fopenmp"]
        linkflags = ["-Xpreprocessor", "-fopenmp", "-lomp"]
    else:
        compflags = ["-fopenmp"]
        linkflags = ["-fopenmp", "-lomp"]

    tmp_dir = tempfile.mkdtemp()
    start_dir = os.path.abspath(".")

    try:
        os.chdir(tmp_dir)
        with open("test_openmp.c", "w") as f:
            f.write(CCODE)

        os.mkdir("objects")
        ccompiler.compile(
            ["test_openmp.c"], output_dir="objects",
            extra_preargs=["-I/usr/local/include"] if sys.platform == "darwin" else None,
            extra_postargs=compflags
        )
        objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))
        ccompiler.link_executable(objects, "test_openmp", extra_postargs=linkflags)

        output = subprocess.check_output("./test_openmp")
        output = output.decode(sys.stdout.encoding or "utf-8").splitlines()
        if "nthreads=" in output[0]:
            nthreads = int(output[0].strip().split("=")[1])
            if len(output) == nthreads:
                haveit = True
            else:
                log.warn(
                    "Unexpected number of lines from output of test OpenMP "
                    "program (output was {0})".format(output)
                )
                haveit = False
        else:
            log.warn(
                "Unexpected output from test OpenMP "
                "program (output was {0})".format(output)
            )
            haveit = False

        haveit = True

    except (CompileError, LinkError):
        haveit = False

    finally:
        os.chdir(start_dir)

    return haveit


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
        use_omp = has_omp()

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
                if sys.platform == "darwin":
                    ext.extra_link_args = ["-mmacosx-version-min=10.10"]
                ext.extra_link_args.append("-lomp")
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
