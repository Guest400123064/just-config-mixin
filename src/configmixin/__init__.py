"""
.. include:: ../../README.md
"""

from ._core import ConfigMixin, register_to_config
from ._json import default, option

__version__ = "0.1.1rc1"

__all__ = [
    "ConfigMixin",
    "register_to_config",
    "default",
    "option",
]
