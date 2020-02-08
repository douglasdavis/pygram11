#!/bin/bash

## This script will build manylinux2010 x86_64 wheels; it should be
## executed from the root project directory while inside a
## manylinux{2010,2014} docker container

set -e -x

pys=(
    /opt/python/cp36-cp36m/bin/python
    /opt/python/cp37-cp37m/bin/python
    /opt/python/cp38-cp38/bin/python
)

for py in "${pys[@]}"; do
    $py -m pip install pip setuptools numpy -U
    $py -m pip wheel .
done

whls=(pygram11*linux*.whl)

/opt/python/cp37-cp37m/bin/python -m pip install twine auditwheel

for whl in "${whls[@]}"; do
    /opt/python/cp37-cp37m/bin/python -m auditwheel repair $whl
done
