#!/bin/bash

## This script will build manylinux1 x86_64 wheels; it should be
## executed from the root project directory while inside a
## skhep/manylinuxgcc-x86_64 docker container.

set -e -x

CC=/usr/local/gcc-9.1.0/bin/gcc-9.1.0
CXX=/usr/local/gcc-9.1.0/bin/g++-9.1.0
LD_LIBRARY_PATH=/usr/local/gcc-9.1.0/lib64

pys=(
    /opt/python/cp27-cp27m/bin/python
    /opt/python/cp36-cp36m/bin/python
    /opt/python/cp37-cp37m/bin/python
)

for py in "${pys[@]}"; do
    $py -m pip install pip -U
    $py -m pip wheel .
done

whls=(pygram11*.whl)

/opt/python/cp37-cp37m/bin/python -m pip install twine auditwheel

for whl in "${whls[@]}"; do
    /opt/python/cp37-cp37m/bin/python -m auditwheel repair $whl
done

rm -rf *.whl
