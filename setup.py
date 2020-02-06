# MIT License

# Copyright (c) 2019 Douglas Davis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import codecs
import glob
import os
import re
import pathlib
import subprocess
import sys
import tempfile

import setuptools
from setuptools import setup
from setuptools.extension import Extension


def has_flag(compiler, flag):
    """check if compiler has compatibility with the flag"""
    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char** argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flag])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def get_cpp_std_flag():
    compiler = setuptools.distutils.ccompiler.new_compiler()
    setuptools.distutils.sysconfig.customize_compiler(compiler)
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError("C++11 (or later) compatible compiler required")


def get_compile_flags(is_cpp=False):
    """get the compile flags"""
    if is_cpp:
        cpp_std = get_cpp_std_flag()
    cflags = ["-Wall", "-Wextra"]
    debug_env = os.getenv("PYGRAM11_DEBUG")
    if debug_env is None:
        cflags += ["-g0"]
    else:
        cflags += ["-g"]
    if sys.platform.startswith("darwin"):
        if is_cpp:
            cflags += ["-fvisibility=hidden", "-stdlib=libc++", cpp_std]
        cflags += ["-Xpreprocessor", "-fopenmp"]
    else:
        if is_cpp:
            cflags += ["-fvisibility=hidden", cpp_std]
        cflags += ["-fopenmp"]
    return cflags


def get_link_flags(is_cpp=False):
    envPREFIX = os.getenv("PREFIX")
    lflags = []
    if sys.platform.startswith("darwin"):
        if envPREFIX is not None:
            lflags += ["-Wl,-rpath,{}/lib".format(envPREFIX)]
        lflags += ["-lomp"]
    else:
        lflags += ["-lgomp"]
    return lflags


def has_openmp():
    test_code = """
    #include <omp.h>
    #include <stdio.h>
    int main() {
      #pragma omp parallel
      printf("nthreads=%d\\n", omp_get_num_threads());
      return 0;
    }
    """
    has_omp = False
    compiler = setuptools.distutils.ccompiler.new_compiler()
    setuptools.distutils.sysconfig.customize_compiler(compiler)
    cflags = get_compile_flags()
    lflags = get_link_flags()
    tmp_dir = tempfile.mkdtemp()
    start_dir = pathlib.PosixPath.cwd()
    try:
        os.chdir(tmp_dir)
        with open("test_openmp.c", "w") as f:
            f.write(test_code)
        os.mkdir("obj")
        compiler.compile(["test_openmp.c"], output_dir="obj", extra_postargs=cflags)
        objs = glob.glob(os.path.join("obj", "*{}".format(compiler.obj_extension)))
        compiler.link_executable(objs, "test_openmp", extra_postargs=lflags)
        output = subprocess.check_output("./test_openmp")
        output = output.decode(sys.stdout.encoding or "utf-8").splitlines()
        if "nthreads=" in output[0]:
            nthreads = int(output[0].strip().split("=")[1])
            if len(output) == nthreads:
                has_omp = True
            else:
                has_omp = False
        else:
            has_omp = False
    except (
        setuptools.distutils.errors.CompileError,
        setuptools.distutils.errors.LinkError,
    ):
        has_omp = False
    finally:
        os.chdir(start_dir)

    return has_omp


def get_extensions():
    c_cflags = get_compile_flags()
    c_lflags = get_link_flags()
    cpp_cflags = get_compile_flags(is_cpp=True)
    cpp_lflags = get_link_flags(is_cpp=True)
    extenmods = []
    extenmods += [
        Extension(
            "pygram11._CPP",
            [os.path.join("pygram11", "_backend.cpp")],
            language="c++",
            include_dirs=[numpy.get_include()],
            extra_compile_args=cpp_cflags,
            extra_link_args=cpp_lflags,
        ),
        Extension(
            "pygram11._CPP_PB",
            [os.path.join("pygram11", "_backend_pb.cpp")],
            language="c++",
            include_dirs=["extern/pybind11/include"],
            extra_compile_args=cpp_cflags,
            extra_link_args=cpp_lflags,
        ),
        Extension(
            "pygram11._CPP_CPP_OBJ",
            [os.path.join("pygram11", "_backend_pb_obj.cpp")],
            language="c++",
            include_dirs=["extern/pybind11/include"],
            extra_compile_args=cpp_cflags,
            extra_link_args=cpp_lflags,
        ),
        Extension(
            "pygram11._CPP_PB_2D",
            [os.path.join("pygram11", "_backend_pb_2d.cpp")],
            language="c++",
            include_dirs=["extern/pybind11/include"],
            extra_compile_args=cpp_cflags,
            extra_link_args=cpp_lflags,
        ),
    ]
    return extenmods


def read_files(*parts):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def get_version(*file_paths):
    version_file = read_files(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_readme():
    project_root = pathlib.PosixPath(__file__).parent
    with (project_root / "README.md").open("rb") as f:
        return f.read().decode("utf-8")


def get_requirements():
    project_root = pathlib.PosixPath(__file__).parent
    with (project_root / "requirements.txt").open("r") as f:
        requirements = f.read().splitlines()


try:
    import numpy
except ImportError:
    sys.exit(
        "\n"
        "*******************************************************\n"
        "* NumPy is required to use pygram11's setup.py script *\n"
        "*******************************************************"
    )

if not has_openmp():
    sys.exit(
        "\n"
        "****************************************************\n"
        "* OpenMP not available, aborting installation.     *\n"
        "* On macOS you can install `libomp` with Homebrew. *\n"
        "* On Linux check your GCC installation.            *\n"
        "****************************************************"
    )

setup(
    name="pygram11",
    version=get_version("pygram11", "__init__.py"),
    author="Doug Davis",
    author_email="ddavis@ddavis.io",
    url="https://github.com/douglasdavis/pygram11",
    description="Fast histogramming in python built on pybind11 and OpenMP.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    packages=["pygram11"],
    ext_modules=get_extensions(),
    install_requires=get_requirements(),
    python_requires=">=3.6",
    test_suite="tests",
    tests_require=["pytest>=4.0"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: C++",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
    ],
)
