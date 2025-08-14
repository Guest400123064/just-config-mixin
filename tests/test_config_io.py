#!/usr/bin/env python3

"""Tests for configuration saving and loading functionality.

This module tests the core configuration I/O operations including:
- Basic save_config and from_config functionality
- runtime_kwargs for complex objects
- apply_param_hooks for post-processing
- Edge cases with ignored parameters, private parameters, and variable arguments
"""

import json
from dataclasses import dataclass
from typing import Any, Dict

import pytest

from configmixin import ConfigMixin, register_to_config


class SampleCustomClass:
    """Sample custom class for testing apply_param_hooks."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "_type": "SampleCustomClass"}

    def __eq__(self, other):
        return (
            isinstance(other, SampleCustomClass)
            and self.name == other.name
            and self.value == other.value
        )


class MockComplexObject:
    """Mock complex object for testing runtime_kwargs."""

    def __init__(self, name: str, data: Any):
        self.name = name
        self.data = data

    def __eq__(self, other):
        return (
            isinstance(other, MockComplexObject)
            and self.name == other.name
            and self.data == other.data
        )


class BasicConfig(ConfigMixin):
    """Basic configuration class for testing core functionality."""

    config_name = "basic_config.json"

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 768,
        learning_rate: float = 0.001,
        name: str = "default",
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.name = name


class ConfigWithIgnored(ConfigMixin):
    """Configuration class with ignored parameters."""

    config_name = "config_ignored.json"
    ignore_for_config = ["model", "device"]

    @register_to_config
    def __init__(self, hidden_size: int = 512, model: Any = None, device: str = "cpu"):
        self.hidden_size = hidden_size
        self.model = model
        self.device = device


class ConfigWithPrivate(ConfigMixin):
    """Configuration class with private parameters (leading underscore)."""

    config_name = "config_private.json"

    @register_to_config
    def __init__(self, public_param: int = 10, _private_param: str = "private"):
        self.public_param = public_param
        self._private_param = _private_param


class ConfigWithVarArgs(ConfigMixin):
    """Configuration class with variable arguments."""

    config_name = "config_var_args.json"

    @register_to_config
    def __init__(
        self, base_param: int = 20, *args, extra_param: str = "extra", **kwargs
    ):
        self.base_param = base_param
        self.args = args
        self.extra_param = extra_param
        self.kwargs = kwargs


def custom_serializer(obj):
    """Custom serializer that handles objects with to_dict method."""
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ConfigWithCustomClass(ConfigMixin):
    """Configuration class that uses apply_param_hooks for custom class conversion."""

    config_name = "config_custom.json"

    @register_to_config
    def __init__(self, regular_param: int = 100, data_obj: SampleCustomClass = None):
        self.regular_param = regular_param
        self.data_obj = data_obj or SampleCustomClass("default", 42)

    @classmethod
    def apply_param_hooks(cls, jdict: dict[str, Any]) -> dict[str, Any]:
        """Convert dict back to SampleCustomClass if needed."""
        if "data_obj" in jdict and isinstance(jdict["data_obj"], dict):
            if jdict["data_obj"].get("_type") == "SampleCustomClass":
                jdict["data_obj"] = SampleCustomClass(
                    name=jdict["data_obj"]["name"], value=jdict["data_obj"]["value"]
                )
        return jdict


class TestBasicConfigIO:
    """Test basic save_config and from_config functionality."""

    def test_save_and_load_basic_config(self, temp_dir):
        """Test basic save and load with default values."""
        config = BasicConfig()

        # Save config
        config.save_config(temp_dir)

        # Verify file was created
        config_file = temp_dir / "basic_config.json"
        assert config_file.exists()

        # Load config
        loaded_config = BasicConfig.from_config(save_directory=temp_dir)

        # Verify values match
        assert loaded_config.hidden_size == config.hidden_size
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.name == config.name

    def test_save_and_load_custom_values(self, temp_dir):
        """Test save and load with custom values."""
        config = BasicConfig(hidden_size=1024, learning_rate=0.01, name="custom")

        config.save_config(temp_dir)
        loaded_config = BasicConfig.from_config(save_directory=temp_dir)

        assert loaded_config.hidden_size == 1024
        assert loaded_config.learning_rate == 0.01
        assert loaded_config.name == "custom"

    def test_config_schema_structure(self, temp_dir):
        """Test that saved config has correct __notes__ schema."""
        config = BasicConfig(hidden_size=512)  # learning_rate and name use defaults
        config.save_config(temp_dir)

        # Read and verify file structure
        with open(temp_dir / "basic_config.json") as f:
            saved_data = json.load(f)

        # Check __notes__ structure
        assert "__notes__" in saved_data
        notes = saved_data["__notes__"]
        assert "class_name" in notes
        assert "using_default_values" in notes
        assert "args" in notes
        assert "kwargs" in notes

        # Check values
        assert notes["class_name"] == "BasicConfig"
        assert "learning_rate" in notes["using_default_values"]
        assert "name" in notes["using_default_values"]
        assert saved_data["hidden_size"] == 512


class TestRuntimeKwargs:
    """Test runtime_kwargs functionality for complex objects."""

    def test_runtime_kwargs_basic(self, temp_dir):
        """Test basic runtime_kwargs functionality."""
        mock_model = MockComplexObject("test_model", [1, 2, 3])
        config = ConfigWithIgnored(hidden_size=256, model=mock_model, device="gpu")

        config.save_config(temp_dir)

        # Load with runtime_kwargs
        runtime_kwargs = {"model": mock_model, "device": "gpu"}
        loaded_config = ConfigWithIgnored.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_config.hidden_size == 256
        assert loaded_config.model == mock_model
        assert loaded_config.device == "gpu"

    def test_private_params_via_runtime_kwargs(self, temp_dir):
        """Test that private parameters must be passed via runtime_kwargs."""
        config = ConfigWithPrivate(public_param=50, _private_param="secret")

        config.save_config(temp_dir)

        # Should fail without runtime_kwargs for private param
        with pytest.raises(KeyError, match="missing required parameter"):
            ConfigWithPrivate.from_config(save_directory=temp_dir)

        # Should succeed with runtime_kwargs
        runtime_kwargs = {"_private_param": "secret"}
        loaded_config = ConfigWithPrivate.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_config.public_param == 50
        assert loaded_config._private_param == "secret"


class TestApplyParamHooks:
    """Test apply_param_hooks for post-processing loaded JSON."""

    def test_custom_class_conversion(self, temp_dir):
        """Test converting dict back to custom class via apply_param_hooks."""
        data_obj = SampleCustomClass("test_name", 123)
        config = ConfigWithCustomClass(regular_param=200, data_obj=data_obj)

        # Save config with custom serializer
        config.save_config(temp_dir, default=custom_serializer)

        # Load config (should auto-convert dict back to custom class)
        loaded_config = ConfigWithCustomClass.from_config(save_directory=temp_dir)

        assert loaded_config.regular_param == 200
        assert isinstance(loaded_config.data_obj, SampleCustomClass)
        assert loaded_config.data_obj.name == "test_name"
        assert loaded_config.data_obj.value == 123

    def test_dataclass_realistic_usage(self, temp_dir):
        """Test realistic dataclass usage where apply_param_hooks reconstructs from dict."""

        @dataclass
        class ModelConfig:
            layers: int
            activation: str

        class ConfigWithDataclass(ConfigMixin):
            config_name = "realistic_config.json"

            @register_to_config
            def __init__(
                self, model_config: ModelConfig = None, learning_rate: float = 0.001
            ):
                self.model_config = model_config or ModelConfig(
                    layers=3, activation="relu"
                )
                self.learning_rate = learning_rate

            @classmethod
            def apply_param_hooks(cls, jdict: dict[str, Any]) -> dict[str, Any]:
                """Convert model_config dict back to ModelConfig dataclass."""
                if "model_config" in jdict and isinstance(jdict["model_config"], dict):
                    jdict["model_config"] = ModelConfig(**jdict["model_config"])
                return jdict

        # Test with custom dataclass
        model_config = ModelConfig(layers=5, activation="tanh")
        config = ConfigWithDataclass(model_config=model_config, learning_rate=0.01)

        config.save_config(temp_dir)
        loaded_config = ConfigWithDataclass.from_config(save_directory=temp_dir)

        assert loaded_config.learning_rate == 0.01
        assert isinstance(loaded_config.model_config, ModelConfig)
        assert loaded_config.model_config.layers == 5
        assert loaded_config.model_config.activation == "tanh"


class TestVariableArguments:
    """Test handling of *args and **kwargs."""

    def test_var_args_handling(self, temp_dir):
        """Test variable arguments are stored in __notes__."""
        config = ConfigWithVarArgs(
            30, "arg1", "arg2", extra_param="modified", kwarg1="value1", kwarg2=42
        )

        config.save_config(temp_dir)

        # Check saved structure
        with open(temp_dir / "config_var_args.json") as f:
            saved_data = json.load(f)

        notes = saved_data["__notes__"]
        assert notes["args"] == ["arg1", "arg2"]
        assert notes["kwargs"] == {"kwarg1": "value1", "kwarg2": 42}

        # Test loading
        loaded_config = ConfigWithVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 30
        assert loaded_config.extra_param == "modified"
        assert loaded_config.args == ("arg1", "arg2")
        assert loaded_config.kwargs == {"kwarg1": "value1", "kwarg2": 42}


class TestErrorCases:
    """Test error handling and edge cases."""

    def test_missing_config_name(self):
        """Test error when config_name is not defined."""

        class InvalidConfig(ConfigMixin):
            # Missing config_name
            @register_to_config
            def __init__(self, param: int = 1):
                self.param = param

        with pytest.raises(NotImplementedError, match="config_name"):
            config = InvalidConfig()

    def test_class_name_mismatch(self, temp_dir):
        """Test error when loading config with wrong class name."""
        config = BasicConfig()
        config.save_config(temp_dir)

        # Manually modify saved file to have wrong class name
        config_file = temp_dir / "basic_config.json"
        with open(config_file) as f:
            data = json.load(f)

        data["__notes__"]["class_name"] = "WrongClassName"

        with open(config_file, "w") as f:
            json.dump(data, f)

        # Should raise error
        with pytest.raises(ValueError, match="not a config for BasicConfig"):
            BasicConfig.from_config(save_directory=temp_dir)

    def test_file_not_found(self, temp_dir):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            BasicConfig.from_config(save_directory=temp_dir)

    def test_overwrite_protection(self, temp_dir):
        """Test that overwrite=False prevents file overwriting."""
        config = BasicConfig()
        config.save_config(temp_dir)

        # Should fail without overwrite=True
        with pytest.raises(FileExistsError):
            config.save_config(temp_dir, overwrite=False)

        # Should succeed with overwrite=True
        config.save_config(temp_dir, overwrite=True)


class TestConfigProperties:
    """Test config property and related functionality."""

    def test_config_property_returns_mappingproxy(self):
        """Test that config property returns MappingProxyType."""
        config = BasicConfig()

        from types import MappingProxyType

        assert isinstance(config.config, MappingProxyType)

    def test_config_immutability(self):
        """Test that config cannot be directly modified."""
        config = BasicConfig()

        with pytest.raises((TypeError, AttributeError)):
            config.config["new_key"] = "new_value"

    def test_config_dumps_produces_valid_json(self):
        """Test that config_dumps produces valid JSON."""
        config = BasicConfig(hidden_size=128)

        json_bytes = config.config_dumps()
        parsed = json.loads(json_bytes.decode())

        assert "__notes__" in parsed
        assert "hidden_size" in parsed
        assert parsed["hidden_size"] == 128
