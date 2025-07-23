#!/usr/bin/env python3

"""
Shared fixtures and utilities for the YACM test suite.
"""

import json
import pathlib
import tempfile
import pytest
from typing import Any, Dict, List

from yacm import ConfigMixin, register_to_config


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def sample_config_data():
    """Provide sample configuration data for testing."""
    return {
        "_class_name": "SampleConfig",
        "param1": 42,
        "param2": "test_value",
        "param3": [1, 2, 3],
        "param4": {"nested": "data"},
        "_use_default_values": ["param2"]
    }


@pytest.fixture
def sample_config_file(temp_directory, sample_config_data):
    """Create a sample config file for testing."""
    config_file = temp_directory / "sample_config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_config_data, f, indent=2)
    return config_file


class SampleConfigClass(ConfigMixin):
    """Sample configuration class for testing."""

    config_name = "sample_config.json"
    ignore_for_config = ["ignored_param"]

    @register_to_config
    def __init__(self,
                 param1: int = 10,
                 param2: str = "default",
                 param3: List[int] = None,
                 param4: Dict[str, Any] = None,
                 ignored_param: bool = False):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3 or []
        self.param4 = param4 or {}
        self.ignored_param = ignored_param


@pytest.fixture
def sample_config_class():
    """Provide the sample config class for testing."""
    return SampleConfigClass


@pytest.fixture
def sample_config_instance():
    """Provide a sample config instance for testing."""
    return SampleConfigClass(
        param1=100,
        param2="test",
        param3=[4, 5, 6],
        param4={"key": "value"},
        ignored_param=True
    )


class ComplexConfigClass(ConfigMixin):
    """Complex configuration class for advanced testing."""

    config_name = "complex_config.json"
    ignore_for_config = ["debug", "_internal"]

    @register_to_config
    def __init__(self,
                 # Basic types
                 name: str = "default_model",
                 version: int = 1,
                 enabled: bool = True,
                 threshold: float = 0.5,

                 # Complex types
                 tags: List[str] = None,
                 metadata: Dict[str, Any] = None,
                 paths: List[pathlib.Path] = None,

                 # Optional types
                 description: str = None,

                 # Ignored parameters
                 debug: bool = False,
                 _internal: str = "internal_value"):

        self.name = name
        self.version = version
        self.enabled = enabled
        self.threshold = threshold

        self.tags = tags or ["default"]
        self.metadata = metadata or {"created_by": "test"}
        self.paths = paths or [pathlib.Path("/tmp")]

        self.description = description
        self.debug = debug
        self._internal = _internal


@pytest.fixture
def complex_config_class():
    """Provide the complex config class for testing."""
    return ComplexConfigClass


@pytest.fixture
def complex_config_instance():
    """Provide a complex config instance for testing."""
    return ComplexConfigClass(
        name="test_model",
        version=2,
        threshold=0.75,
        tags=["test", "integration"],
        metadata={"author": "pytest", "date": "2024"},
        paths=[pathlib.Path("/data"), pathlib.Path("/models")],
        description="Test configuration",
        debug=True
    )


# Utility functions for tests

def assert_config_equals(config1: ConfigMixin, config2: ConfigMixin):
    """Assert that two config instances have equivalent configurations."""
    assert config1.__class__ == config2.__class__

    # Compare config dictionaries
    config1_dict = dict(config1.config)
    config2_dict = dict(config2.config)

    # Remove metadata that might differ
    for key in ["_use_default_values"]:
        config1_dict.pop(key, None)
        config2_dict.pop(key, None)

    assert config1_dict == config2_dict


def create_test_config_file(directory: pathlib.Path,
                           config_name: str,
                           config_data: Dict[str, Any]) -> pathlib.Path:
    """Create a test configuration file with given data."""
    config_file = directory / config_name
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2, sort_keys=True)
    return config_file


def load_config_file(config_file: pathlib.Path) -> Dict[str, Any]:
    """Load configuration data from a file."""
    with open(config_file) as f:
        return json.load(f)


# Test data generators

def generate_config_variations():
    """Generate various configuration test data."""
    return [
        # Basic configuration
        {
            "param1": 1,
            "param2": "basic",
            "param3": [],
            "param4": {}
        },
        # Configuration with complex types
        {
            "param1": 999,
            "param2": "complex",
            "param3": [1, 2, 3, 4, 5],
            "param4": {
                "nested": {"deep": "value"},
                "list": [1, 2, 3],
                "mixed": {"int": 42, "str": "text"}
            }
        },
        # Configuration with None values
        {
            "param1": 0,
            "param2": "none_test",
            "param3": None,
            "param4": None
        }
    ]


# Pytest markers for organizing tests

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )


# Test collection helpers

class TestConfigFactory:
    """Factory for creating test configuration classes."""

    @staticmethod
    def create_simple_config(config_name: str = "test_config.json"):
        """Create a simple test configuration class."""

        class SimpleTestConfig(ConfigMixin):

            @register_to_config
            def __init__(self, value: int = 42):
                self.value = value

        SimpleTestConfig.config_name = config_name
        return SimpleTestConfig

    @staticmethod
    def create_config_with_ignored(config_name: str = "ignored_config.json",
                                  ignored_params: List[str] = None):
        """Create a config class with ignored parameters."""

        class IgnoredTestConfig(ConfigMixin):

            @register_to_config
            def __init__(self, tracked: int = 1, ignored: str = "ignore_me"):
                self.tracked = tracked
                self.ignored = ignored

        IgnoredTestConfig.config_name = config_name
        IgnoredTestConfig.ignore_for_config = ignored_params or ["ignored"]
        return IgnoredTestConfig

    @staticmethod
    def create_config_with_private(config_name: str = "private_config.json"):
        """Create a config class with private parameters."""

        class PrivateTestConfig(ConfigMixin):

            @register_to_config
            def __init__(self, public: int = 1, _private: str = "private"):
                self.public = public
                self._private = _private

        PrivateTestConfig.config_name = config_name
        return PrivateTestConfig


@pytest.fixture
def test_config_factory():
    """Provide the test config factory."""
    return TestConfigFactory


# Performance testing utilities

def measure_config_performance(config_class, iterations: int = 1000):
    """Measure performance of config operations."""
    import time

    # Measure creation time
    start_time = time.time()
    for _ in range(iterations):
        config = config_class()
    creation_time = time.time() - start_time

    # Measure serialization time
    config = config_class()
    start_time = time.time()
    for _ in range(iterations):
        config.get_config_json()
    serialization_time = time.time() - start_time

    return {
        "creation_time": creation_time,
        "serialization_time": serialization_time,
        "iterations": iterations
    }


# Mock objects for testing

class MockConfigMixin(ConfigMixin):
    """Mock ConfigMixin for testing without decorator."""

    config_name = "mock_config.json"

    def __init__(self, param1: int = 1, param2: str = "mock"):
        self.param1 = param1
        self.param2 = param2
        # Manually register config
        self.register_to_config(param1=param1, param2=param2)


@pytest.fixture
def mock_config():
    """Provide a mock config instance."""
    return MockConfigMixin(param1=123, param2="mocked")
