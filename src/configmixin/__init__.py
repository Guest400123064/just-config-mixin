"""
YACM - Yet Another Config Manager

A simple, standalone configuration management package inspired by the ConfigMixin class
from the diffusers library.
"""

from ._core import ConfigMixin, FrozenDict, register_to_config

__version__ = "0.1.0"

__all__ = [
    "ConfigMixin",
    "FrozenDict",
    "register_to_config",
]
