#!/bin/bash

## This script will build manylinux2010 x86_64 wheels; it should be
## executed from the root project directory while inside a
## manylinux{1,2010} docker container
## Example shell command:
## $ docker run --rm -it -w $PWD -v $PWD:$PWD quay.io/pypa/manylinux2010_x86_64 \
##     /bin/bash utils/manylinux-wheels.sh

set -e -x

pys=(
    /opt/python/cp36-cp36m/bin/python
    /opt/python/cp37-cp37m/bin/python
    /opt/python/cp38-cp38/bin/python
)

for py in "${pys[@]}"; do
    $py -m pip install pip pep517 setuptools -U
    $py -m pep517.build --binary . --out-dir .
done

whls=(pygram11*.whl)

/opt/python/cp37-cp37m/bin/python -m pip install twine auditwheel

for whl in "${whls[@]}"; do
    /opt/python/cp37-cp37m/bin/python -m auditwheel repair $whl
done
