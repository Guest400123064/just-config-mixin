"""
.. include:: ../../README.md
"""

from importlib.metadata import PackageNotFoundError, version

from ._core import ConfigMixin, register_to_config
from ._json import default, option

try:
    __version__ = version("just-config-mixin")
except PackageNotFoundError:
    # Fallback for source trees where the distribution metadata is unavailable.
    __version__ = "0.0.0+unknown"

__all__ = [
    "ConfigMixin",
    "register_to_config",
    "default",
    "option",
]
