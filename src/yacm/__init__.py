"""
YACM - Yet Another Config Manager

A simple, standalone configuration management package inspired by the ConfigMixin class 
from the diffusers library.
"""

from ._cli import (
    add_argparse_arguments,
    config_from_args,
    create_config_subparser,
    parse_config_from_args,
)
from ._core import ConfigMixin, FrozenDict, register_to_config

__version__ = "0.1.0"

__all__ = [
    "ConfigMixin",
    "FrozenDict", 
    "register_to_config",
    "add_argparse_arguments",
    "config_from_args",
    "parse_config_from_args",
    "create_config_subparser",
]
