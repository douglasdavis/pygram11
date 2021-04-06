"""pygram11 configuration."""

# MIT License
#
# Copyright (c) 2021 Douglas Davis
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Iterator


config: Dict[str, int] = {
    "thresholds.fixed1d": 10_000,
    "thresholds.fixed1dmw": 10_000,
    "thresholds.fixed2d": 10_000,
    "thresholds.variable1d": 5_000,
    "thresholds.variable1dmw": 5_000,
    "thresholds.variable2d": 5_000,
}


def get(key: str) -> int:
    """Retrieve a configuration value given a key.

    Parameters
    ----------
    key : str
        Desired configuration parameter key.

    Returns
    -------
    int
        Value of the key.

    """
    global config
    return config[key]


def set(key: str, value: int) -> None:
    """Set a configuration key's value.

    Parameters
    ----------
    key : str
        Desired configuration parameter key to modify.
    value : int
        New value for the given key.

    """
    global config
    if key not in config:
        raise ValueError(f"{key} not in config")
    config[key] = value


def threshold_keys() -> Iterator[str]:
    """All available keys in the configuration dictionary."""
    global config
    for k in config.keys():
        yield k
