# MIT License
#
# Copyright (c) 2021 Douglas Davis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
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
import platform
import subprocess
import sys
import tempfile

import setuptools
from setuptools import setup
from setuptools.extension import Extension


CONDA_BUILD = "CONDA_BUILD" in os.environ


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
    else:
        raise RuntimeError("C++14 (or later) compatible compiler required")


def conda_darwin_flags(flavor="inc"):
    if not CONDA_BUILD:
        return []
    pref = os.environ["PREFIX"]
    if flavor == "inc":
        return [f"-I{pref}/include"]
    elif flavor == "lib":
        return [f"-Wl,-rpath,{pref}/lib", f"-L{pref}/lib"]
    else:
        return []


def is_apple_silicon():
    return sys.platform.startswith("darwin") and "arm" in platform.processor()


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
        if is_apple_silicon() and not CONDA_BUILD:
            cflags += ["-I/opt/homebrew/include"]
        cflags += ["-Xpreprocessor", "-fopenmp"]
        cflags += conda_darwin_flags("inc")
    else:
        if is_cpp:
            cflags += ["-fvisibility=hidden", cpp_std]
        cflags += ["-fopenmp"]
    return cflags


def get_link_flags(is_cpp=False):
    lflags = []
    if sys.platform.startswith("darwin"):
        if is_apple_silicon() and not CONDA_BUILD:
            lflags += ["-L/opt/homebrew/lib"]
        lflags += conda_darwin_flags("lib")
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
    cpp_cflags = get_compile_flags(is_cpp=True)
    cpp_lflags = get_link_flags(is_cpp=True)
    return [
        Extension(
            "pygram11._backend",
            [os.path.join("src", "_backend.cpp")],
            language="c++",
            include_dirs=["extern/pybind11/include", "extern/mp11/include"],
            extra_compile_args=cpp_cflags,
            extra_link_args=cpp_lflags,
        ),
    ]


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
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    ext_modules=get_extensions(),
    zip_safe=False,
)
