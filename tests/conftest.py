#!/usr/bin/env python3

r"""Shared fixtures and utilities for the ConfigMixin test suite.

This module provides common test fixtures, base classes, and utilities
used across the test suite to reduce duplication and ensure consistency.
"""

import json
import pathlib
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from configmixin import ConfigMixin, register_to_config

# Test Fixtures

@pytest.fixture
def temp_directory():
    r"""Provide a temporary directory for test files.

    Returns
    -------
    pathlib.Path
        Path to a temporary directory that will be cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def sample_config_data():
    r"""Provide sample configuration data for testing.

    Returns
    -------
    dict[str, Any]
        Sample configuration dictionary with all required metadata.
    """
    return {
        "_class_name": "SampleConfig",
        "param1": 42,
        "param2": "test_value",
        "param3": [1, 2, 3],
        "param4": {"nested": "data"},
        "_use_default_values": ["param2"],
        "_var_positional": (),
        "_var_keyword": {},
    }


@pytest.fixture
def sample_config_file(temp_directory, sample_config_data):
    r"""Create a sample config file for testing.

    Parameters
    ----------
    temp_directory : pathlib.Path
        Temporary directory fixture.
    sample_config_data : dict[str, Any]
        Sample configuration data fixture.

    Returns
    -------
    pathlib.Path
        Path to the created config file.
    """
    config_file = temp_directory / "sample_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(sample_config_data, f, indent=2)
    return config_file


@pytest.fixture
def mock_model():
    r"""Provide a mock model object for runtime testing.

    Returns
    -------
    Mock
        Mock object with configurable attributes for testing runtime parameters.
    """
    model = Mock()
    model.name = "test_model"
    model.version = "1.0"
    return model


# Base Test Classes

class BaseConfig(ConfigMixin):
    r"""Base configuration class for testing core functionality."""

    config_name = "base_config.json"

    @register_to_config
    def __init__(self, param1: int = 10, param2: str = "default", param3: List[int] = None):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3 or [1, 2, 3]


class ConfigWithIgnored(ConfigMixin):
    r"""Configuration class with ignored parameters for testing."""

    config_name = "config_ignored.json"
    ignore_for_config = ["ignored_param", "runtime_param"]

    @register_to_config
    def __init__(
        self,
        tracked_param: int = 5,
        ignored_param: str = "ignored",
        runtime_param: Any = None,
        another_tracked: float = 1.0,
    ):
        self.tracked_param = tracked_param
        self.ignored_param = ignored_param
        self.runtime_param = runtime_param
        self.another_tracked = another_tracked


class ConfigWithPrivate(ConfigMixin):
    r"""Configuration class with private parameters for testing."""

    config_name = "config_private.json"

    @register_to_config
    def __init__(
        self,
        public_param: int = 10,
        _private_param: str = "private",
        normal_param: float = 2.5,
    ):
        self.public_param = public_param
        self._private_param = _private_param
        self.normal_param = normal_param


class ConfigWithVarArgs(ConfigMixin):
    r"""Configuration class with var args for testing."""

    config_name = "config_var_args.json"

    @register_to_config
    def __init__(self, base_param: int = 10, *args, extra_param: str = "default"):
        self.base_param = base_param
        self.args = args
        self.extra_param = extra_param


class ConfigWithVarKwargs(ConfigMixin):
    r"""Configuration class with var kwargs for testing."""

    config_name = "config_var_kwargs.json"

    @register_to_config
    def __init__(self, base_param: int = 20, **kwargs):
        self.base_param = base_param
        self.kwargs = kwargs


class ConfigWithBothVarArgs(ConfigMixin):
    r"""Configuration class with both *args and **kwargs for testing."""

    config_name = "config_both_var_args.json"

    @register_to_config
    def __init__(
        self, base_param: int = 30, *args, named_param: str = "named", **kwargs
    ):
        self.base_param = base_param
        self.args = args
        self.named_param = named_param
        self.kwargs = kwargs


class ConfigWithRuntimeArgs(ConfigMixin):
    r"""Configuration class with runtime arguments for testing."""

    config_name = "config_runtime.json"
    ignore_for_config = ["model", "device", "optimizer"]

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        learning_rate: float = 0.001,
        model: Any = None,
        device: str = "cpu",
        optimizer: Any = None,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.model = model
        self.device = device
        self.optimizer = optimizer


class ConfigWithComplexTypes(ConfigMixin):
    r"""Configuration class with complex types for testing."""

    config_name = "config_complex.json"

    @register_to_config
    def __init__(
        self,
        list_param: List[int] = None,
        dict_param: Dict[str, Any] = None,
        optional_param: Optional[str] = None,
        path_param: pathlib.Path = None,
    ):
        self.list_param = list_param or [1, 2, 3]
        self.dict_param = dict_param or {"key": "value"}
        self.optional_param = optional_param
        self.path_param = path_param or pathlib.Path("/default/path")


# Utility Classes for Multiple Inheritance Testing

class BaseModel:
    r"""Abstract base model class for mixin testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_initialized = True

    def get_info(self) -> str:
        return f"{self.__class__.__name__} model"


class TrainableMixin:
    r"""Mixin for trainable objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trainable = True
        self.training_steps = 0

    def train_step(self) -> str:
        self.training_steps += 1
        return f"Training step {self.training_steps}"


class SerializableMixin:
    r"""Mixin for serializable objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serialization_format = "json"

    def to_bytes(self) -> bytes:
        return json.dumps({"class": self.__class__.__name__}).encode()


# Test Utility Functions

def assert_config_roundtrip(config_instance: ConfigMixin) -> None:
    r"""Assert that a config instance can be saved and loaded correctly.

    Parameters
    ----------
    config_instance : ConfigMixin
        The configuration instance to test.

    Raises
    ------
    AssertionError
        If the roundtrip fails or produces different results.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save config
        config_instance.save_config(temp_dir)

        # Load config
        loaded_instance = config_instance.__class__.from_config(temp_dir)

        # Compare configs
        assert loaded_instance.config == config_instance.config


def assert_config_json_valid(config_instance: ConfigMixin) -> None:
    r"""Assert that a config instance produces valid JSON.

    Parameters
    ----------
    config_instance : ConfigMixin
        The configuration instance to test.

    Raises
    ------
    AssertionError
        If the JSON is invalid or missing required fields.
    """
    json_str = config_instance.get_config_json()
    config_dict = json.loads(json_str)

    # Check required metadata fields
    required_fields = ["_class_name", "_use_default_values"]
    for field in required_fields:
        assert field in config_dict, f"Missing required field: {field}"

    # Check class name matches
    assert config_dict["_class_name"] == config_instance.__class__.__name__


def create_config_dict(
    class_name: str,
    **kwargs: Any
) -> Dict[str, Any]:
    r"""Create a properly formatted config dictionary for testing.

    Parameters
    ----------
    class_name : str
        The name of the configuration class.
    **kwargs : Any
        Configuration parameters to include.

    Returns
    -------
    dict[str, Any]
        Properly formatted configuration dictionary with metadata.
    """
    config_dict = {
        "_class_name": class_name,
        "_use_default_values": [],
        "_var_positional": (),
        "_var_keyword": {},
    }
    config_dict.update(kwargs)
    return config_dict


def parametrize_config_classes(*config_classes):
    r"""Decorator to parametrize tests with multiple config classes.

    Parameters
    ----------
    *config_classes : type[ConfigMixin]
        Configuration classes to test.

    Returns
    -------
    Callable
        Pytest parametrize decorator.
    """
    return pytest.mark.parametrize(
        "config_class",
        config_classes,
        ids=[cls.__name__ for cls in config_classes]
    )


# Error Testing Utilities

class ConfigWithoutName(ConfigMixin):
    r"""Configuration class without config_name for error testing."""

    # config_name is None by default

    @register_to_config
    def __init__(self, param: int = 1):
        self.param = param


class NonConfigMixinClass:
    r"""Class that doesn't inherit from ConfigMixin for error testing."""

    @register_to_config
    def __init__(self, param: int = 1):
        self.param = param


class MockSerializableObject:
    r"""Mock object with to_dict method for serialization testing."""

    def __init__(self, value: Any):
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"mock_value": self.value, "type": "MockSerializableObject"}

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MockSerializableObject)
            and self.value == other.value
        )


class MockNonSerializableObject:
    r"""Mock object that cannot be serialized for error testing."""

    def __init__(self, value: Any):
        self.value = value
        self.circular_ref = self  # Create circular reference

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, MockNonSerializableObject)
            and self.value == other.value
        )
