"""pygram11 configuration."""


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
