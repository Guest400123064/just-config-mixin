#!/usr/bin/env python3

"""Shared fixtures and utilities for the ConfigMixin test suite.

This module provides common test fixtures and utilities used across
multiple test files to reduce duplication and ensure consistency.
"""

import pathlib
import tempfile
from typing import Any, Dict

import pytest

from configmixin import ConfigMixin, register_to_config


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files.

    This fixture is used across multiple test files for creating
    temporary config files during testing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


class MockComplexObject:
    """Mock complex object for testing runtime_kwargs across test files."""

    def __init__(self, name: str, data: Any):
        self.name = name
        self.data = data

    def __eq__(self, other):
        return (
            isinstance(other, MockComplexObject)
            and self.name == other.name
            and self.data == other.data
        )


class SimpleConfig(ConfigMixin):
    """Simple configuration class for basic testing."""

    config_name = "simple_config.json"

    @register_to_config
    def __init__(self, param1: int = 10, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2


class ConfigWithIgnoredParams(ConfigMixin):
    """Configuration class with ignored parameters for cross-test usage."""

    config_name = "config_with_ignored.json"
    ignore_for_config = ["runtime_param", "ignored_param"]

    @register_to_config
    def __init__(
        self,
        tracked_param: int = 100,
        runtime_param: Any = None,
        ignored_param: str = "ignored",
    ):
        self.tracked_param = tracked_param
        self.runtime_param = runtime_param
        self.ignored_param = ignored_param


def custom_serializer(obj):
    """Custom serializer that handles objects with to_dict method.

    This is used across tests that need custom serialization behavior.
    """
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def assert_config_roundtrip(
    config_instance: ConfigMixin, runtime_kwargs: Dict[str, Any] = None
) -> None:
    """Assert that a config instance can be saved and loaded correctly.

    This utility function tests the complete save/load cycle and verifies
    that the configuration roundtrips correctly.

    Parameters
    ----------
    config_instance : ConfigMixin
        The configuration instance to test.
    runtime_kwargs : dict[str, Any], optional
        Runtime kwargs to pass during loading (for ignored/private parameters).

    Raises
    ------
    AssertionError
        If the roundtrip fails or produces different results.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save config
        config_instance.save_config(temp_dir)

        # Load config
        if runtime_kwargs:
            loaded_instance = config_instance.__class__.from_config(
                save_directory=temp_dir, runtime_kwargs=runtime_kwargs
            )
        else:
            loaded_instance = config_instance.__class__.from_config(
                save_directory=temp_dir
            )

        # Compare main config parameters (excluding metadata)
        original_config = {
            k: v for k, v in config_instance.config.items() if not k.startswith("__")
        }
        loaded_config = {
            k: v for k, v in loaded_instance.config.items() if not k.startswith("__")
        }

        assert loaded_config == original_config

        # Verify tracked parameters match
        for param_name in original_config:
            assert getattr(loaded_instance, param_name) == getattr(
                config_instance, param_name
            )

        # Verify runtime parameters match if provided
        if runtime_kwargs:
            for param_name, expected_value in runtime_kwargs.items():
                assert getattr(loaded_instance, param_name) == expected_value
