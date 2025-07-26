#!/usr/bin/env python3

r"""Test suite for core ConfigMixin functionality.

This module tests the fundamental features of ConfigMixin including:
- Basic decorator functionality
- Configuration registration and access
- JSON serialization and deserialization
- Default value tracking
- Private parameter handling
- Ignored parameter handling
"""

import json
import pathlib
import tempfile

import pytest

from configmixin import ConfigMixin, FrozenDict

from .conftest import (
    BaseConfig,
    ConfigWithComplexTypes,
    ConfigWithIgnored,
    ConfigWithoutName,
    ConfigWithPrivate,
    NonConfigMixinClass,
    assert_config_json_valid,
    assert_config_roundtrip,
    create_config_dict,
    parametrize_config_classes,
)


class TestConfigMixinBasics:
    r"""Test basic ConfigMixin functionality."""

    def test_config_registration_with_decorator(self):
        r"""Test that @register_to_config properly registers parameters."""
        config = BaseConfig(param1=42, param2="test", param3=[4, 5, 6])

        assert config.config["param1"] == 42
        assert config.config["param2"] == "test"
        assert config.config["param3"] == [4, 5, 6]
        assert config.config["_class_name"] == "BaseConfig"
        assert isinstance(config.config, FrozenDict)

    def test_config_registration_without_decorator(self):
        r"""Test manual config registration without decorator."""

        class ManualConfig(ConfigMixin):
            config_name = "manual_config.json"

            def __init__(self, value: int = 10):
                self.value = value
                self.register_to_config(_class_name="ManualConfig", value=value)

        config = ManualConfig(value=99)
        assert config.config["value"] == 99
        assert config.config["_class_name"] == "ManualConfig"

    def test_config_attribute_access_shortcut(self):
        r"""Test accessing config values as instance attributes."""
        config = BaseConfig(param1=100, param2="shortcut")

        # Should access instance attributes directly
        assert config.param1 == 100
        assert config.param2 == "shortcut"

        # Should access config-only values via __getattr__
        assert config._class_name == "BaseConfig"
        assert config._use_default_values == ["param3"]

    def test_config_immutability(self):
        r"""Test that config is immutable after creation."""
        config = BaseConfig(param1=50)

        # Should not be able to modify config
        with pytest.raises(Exception):
            config.config["param1"] = 999

        with pytest.raises(Exception):
            config.config.param1 = 999

        with pytest.raises(Exception):
            del config.config["param1"]

    def test_config_repr(self):
        r"""Test string representation of config objects."""
        config = BaseConfig(param1=25, param2="repr_test")
        repr_str = repr(config)

        assert "BaseConfig" in repr_str
        assert "25" in repr_str
        assert "repr_test" in repr_str


class TestDefaultValueTracking:
    r"""Test tracking of default values in configuration."""

    def test_all_explicit_values(self):
        r"""Test when all parameters are explicitly provided."""
        config = BaseConfig(param1=100, param2="explicit", param3=[7, 8, 9])
        assert config.config["_use_default_values"] == []

    def test_some_default_values(self):
        r"""Test when some parameters use defaults."""
        config = BaseConfig(param1=200, param2="partial")
        # param3 should use default
        assert "param3" in config.config["_use_default_values"]
        assert config.param3 == [1, 2, 3]

    def test_all_default_values(self):
        r"""Test when all parameters use defaults."""
        config = BaseConfig()
        expected_defaults = ["param1", "param2", "param3"]
        assert set(config.config["_use_default_values"]) == set(expected_defaults)

        assert config.param1 == 10
        assert config.param2 == "default"
        assert config.param3 == [1, 2, 3]

    def test_keyword_argument_overrides_default(self):
        r"""Test that keyword arguments properly override defaults."""
        config = BaseConfig(param3=[99])

        # Only param1 and param2 should use defaults
        expected_defaults = ["param1", "param2"]
        assert set(config.config["_use_default_values"]) == set(expected_defaults)
        assert config.param3 == [99]


class TestIgnoredParameters:
    r"""Test handling of ignored parameters."""

    def test_ignored_params_not_in_config(self):
        r"""Test that ignored parameters are excluded from config."""
        config = ConfigWithIgnored(
            tracked_param=100,
            ignored_param="should_not_appear",
            runtime_param="also_ignored",
            another_tracked=2.5,
        )

        # Tracked params should be in config
        assert config.config["tracked_param"] == 100
        assert config.config["another_tracked"] == 2.5

        # Ignored params should not be in config
        assert "ignored_param" not in config.config
        assert "runtime_param" not in config.config

        # But should still be accessible as attributes
        assert config.ignored_param == "should_not_appear"
        assert config.runtime_param == "also_ignored"

    def test_ignored_params_with_defaults(self):
        r"""Test ignored parameters that have default values."""
        config = ConfigWithIgnored(tracked_param=50)

        # Ignored params should use their defaults but not appear in config
        assert config.ignored_param == "ignored"
        assert config.runtime_param is None
        assert "ignored_param" not in config.config
        assert "runtime_param" not in config.config


class TestPrivateParameters:
    r"""Test handling of private parameters (starting with underscore)."""

    def test_private_params_excluded_from_config(self):
        r"""Test that private parameters are excluded from config."""
        config = ConfigWithPrivate(
            public_param=200, _private_param="secret", normal_param=3.14
        )

        # Public params should be in config
        assert config.config["public_param"] == 200
        assert config.config["normal_param"] == 3.14

        # Private param should not be in config
        assert "_private_param" not in config.config

        # But should still be accessible as attribute
        assert config._private_param == "secret"

    def test_private_params_with_defaults(self):
        r"""Test private parameters with default values."""
        config = ConfigWithPrivate(public_param=300)

        # Private param should use default but not appear in config
        assert config._private_param == "private"
        assert "_private_param" not in config.config


class TestComplexTypes:
    r"""Test configuration with complex parameter types."""

    def test_list_parameters(self):
        r"""Test list parameters in configuration."""
        config = ConfigWithComplexTypes(list_param=[10, 20, 30])
        assert config.config["list_param"] == [10, 20, 30]

    def test_dict_parameters(self):
        r"""Test dictionary parameters in configuration."""
        config = ConfigWithComplexTypes(dict_param={"test": "value", "number": 42})
        assert config.config["dict_param"] == {"test": "value", "number": 42}

    def test_optional_parameters(self):
        r"""Test optional parameters with None values."""
        config = ConfigWithComplexTypes(optional_param=None)
        assert config.config["optional_param"] is None

    def test_pathlib_parameters(self):
        r"""Test pathlib.Path parameters in configuration."""
        test_path = pathlib.Path("/test/path")
        config = ConfigWithComplexTypes(path_param=test_path)
        assert config.config["path_param"] == test_path

    def test_default_complex_types(self):
        r"""Test default values for complex types."""
        config = ConfigWithComplexTypes()

        assert config.list_param == [1, 2, 3]
        assert config.dict_param == {"key": "value"}
        assert config.optional_param is None
        assert config.path_param == pathlib.Path("/default/path")


class TestJSONSerialization:
    r"""Test JSON serialization functionality."""

    def test_basic_json_serialization(self):
        r"""Test basic JSON serialization of config."""
        config = BaseConfig(param1=123, param2="json_test")
        json_str = config.get_config_json()

        # Should be valid JSON
        config_dict = json.loads(json_str)
        assert config_dict["param1"] == 123
        assert config_dict["param2"] == "json_test"
        assert config_dict["_class_name"] == "BaseConfig"

    def test_json_serialization_with_complex_types(self):
        r"""Test JSON serialization with complex types."""
        config = ConfigWithComplexTypes(
            list_param=[1, 2, 3],
            dict_param={"nested": {"deep": "value"}},
            optional_param="not_none",
        )

        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        assert config_dict["list_param"] == [1, 2, 3]
        assert config_dict["dict_param"] == {"nested": {"deep": "value"}}
        assert config_dict["optional_param"] == "not_none"

    def test_json_serialization_pathlib_conversion(self):
        r"""Test that pathlib.Path objects are converted to strings."""
        config = ConfigWithComplexTypes(path_param=pathlib.Path("/json/path"))
        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        # Path should be converted to POSIX string
        assert config_dict["path_param"] == "/json/path"

    def test_json_sorted_keys(self):
        r"""Test that JSON output has sorted keys."""
        config = BaseConfig(param1=1, param2="test", param3=[1])
        json_str = config.get_config_json()

        # Parse and check key order
        config_dict = json.loads(json_str)
        keys = list(config_dict.keys())
        assert keys == sorted(keys)


class TestConfigSaveAndLoad:
    r"""Test saving and loading configuration files."""

    def test_save_config_basic(self, temp_directory):
        r"""Test basic config saving functionality."""
        config = BaseConfig(param1=999, param2="save_test")
        config.save_config(temp_directory)

        # Check file was created
        config_file = temp_directory / "base_config.json"
        assert config_file.exists()

        # Check file contents
        with open(config_file, encoding="utf-8") as f:
            config_data = json.load(f)

        assert config_data["param1"] == 999
        assert config_data["param2"] == "save_test"
        assert config_data["_class_name"] == "BaseConfig"

    def test_save_config_overwrite_protection(self, temp_directory):
        r"""Test that save_config protects against overwriting by default."""
        config = BaseConfig(param1=100)

        # Save first time
        config.save_config(temp_directory)

        # Try to save again without overwrite flag
        with pytest.raises(FileExistsError) as exc_info:
            config.save_config(temp_directory)

        assert "overwrite=True" in str(exc_info.value)

    def test_save_config_with_overwrite(self, temp_directory):
        r"""Test saving with overwrite=True."""
        config1 = BaseConfig(param1=100)
        config2 = BaseConfig(param1=200)

        # Save first config
        config1.save_config(temp_directory)

        # Overwrite with second config
        config2.save_config(temp_directory, overwrite=True)

        # Verify second config was saved
        config_file = temp_directory / "base_config.json"
        with open(config_file, encoding="utf-8") as f:
            config_data = json.load(f)

        assert config_data["param1"] == 200

    def test_from_config_basic(self, temp_directory):
        r"""Test basic config loading functionality."""
        original = BaseConfig(param1=555, param2="load_test", param3=[9, 8, 7])
        original.save_config(temp_directory)

        # Load config
        loaded = BaseConfig.from_config(save_directory=temp_directory)

        # Should have same configuration
        assert loaded.param1 == 555
        assert loaded.param2 == "load_test"
        assert loaded.param3 == [9, 8, 7]

    @parametrize_config_classes(BaseConfig, ConfigWithIgnored, ConfigWithComplexTypes)
    def test_config_roundtrip(self, config_class):
        r"""Test that configs can be saved and loaded correctly."""
        # Create instance with non-default values
        if config_class == BaseConfig:
            config = config_class(param1=777, param2="roundtrip", param3=[1, 2])
        elif config_class == ConfigWithIgnored:
            config = config_class(tracked_param=888, another_tracked=3.14)
        else:  # ConfigWithComplexTypes
            config = config_class(
                list_param=[5, 6, 7],
                dict_param={"test": "roundtrip"},
                optional_param="not_none",
            )

        assert_config_roundtrip(config)


class TestFromConfigEnhancements:
    r"""Test enhanced from_config functionality."""

    def test_from_config_with_dict(self):
        r"""Test loading config from dictionary."""
        config_dict = create_config_dict(
            "BaseConfig", param1=444, param2="dict_test", param3=[4, 4, 4]
        )

        instance = BaseConfig.from_config(config_dict)
        assert instance.param1 == 444
        assert instance.param2 == "dict_test"
        assert instance.param3 == [4, 4, 4]

    def test_from_config_with_runtime_kwargs(self):
        r"""Test loading config with runtime kwargs."""
        config_dict = create_config_dict(
            "ConfigWithIgnored", tracked_param=333, another_tracked=1.23
        )

        runtime_kwargs = {
            "ignored_param": "runtime_value",
            "runtime_param": "also_runtime",
        }

        instance = ConfigWithIgnored.from_config(
            config_dict, runtime_kwargs=runtime_kwargs
        )

        # Config params should come from dict
        assert instance.tracked_param == 333
        assert instance.another_tracked == 1.23

        # Runtime params should come from runtime_kwargs
        assert instance.ignored_param == "runtime_value"
        assert instance.runtime_param == "also_runtime"


class TestErrorHandling:
    r"""Test error handling in core functionality."""

    def test_decorator_without_configmixin_inheritance(self):
        r"""Test error when decorator is used without ConfigMixin inheritance."""
        with pytest.raises(RuntimeError) as exc_info:
            NonConfigMixinClass(param=5)

        assert "does not inherit from `ConfigMixin`" in str(exc_info.value)

    def test_double_config_registration(self):
        r"""Test error when register_to_config is called twice."""

        class DoubleRegisterConfig(ConfigMixin):
            config_name = "double.json"

            def __init__(self, param: int = 1):
                self.register_to_config(param=param)
                # This should raise an error
                self.register_to_config(param=param)

        with pytest.raises(RuntimeError) as exc_info:
            DoubleRegisterConfig(param=5)

        assert "_internal_dict` is already set" in str(exc_info.value)

    def test_from_config_missing_both_args(self):
        r"""Test error when from_config called without required arguments."""
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.from_config()

        assert "Either `save_directory` or `config` must be provided" in str(
            exc_info.value
        )

    def test_from_config_wrong_class_name(self):
        r"""Test error when config has wrong class name."""
        config_dict = create_config_dict("WrongClassName", param1=1)

        with pytest.raises(ValueError) as exc_info:
            BaseConfig.from_config(config_dict)

        assert "is not a config for BaseConfig" in str(exc_info.value)

    def test_from_config_file_not_found(self):
        r"""Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            BaseConfig.from_config(save_directory="/nonexistent/directory")

        assert "does not contain a file named" in str(exc_info.value)

    def test_save_config_to_file_path(self):
        r"""Test error when save_directory points to a file."""
        config = BaseConfig(param1=1)

        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(AssertionError) as exc_info:
                config.save_config(temp_file.name)

            assert "should be a directory, not a file" in str(exc_info.value)

    def test_from_config_directory_is_file(self):
        r"""Test error when save_directory points to a file in from_config."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(AssertionError) as exc_info:
                BaseConfig.from_config(save_directory=temp_file.name)

            assert "should be a directory, not a file" in str(exc_info.value)

    def test_config_access_before_registration(self):
        r"""Test error when accessing config before registration."""

        class UnregisteredConfig(ConfigMixin):
            config_name = "unregistered.json"

            def __init__(self, param: int = 1):
                # Don't call register_to_config
                self.param = param

        instance = UnregisteredConfig(param=5)

        with pytest.raises(AttributeError):
            _ = instance.config

    def test_attribute_error_for_nonexistent(self):
        r"""Test AttributeError for non-existent attributes."""
        config = BaseConfig()

        with pytest.raises(AttributeError) as exc_info:
            _ = config.nonexistent_attribute

        assert "has no attribute `nonexistent_attribute`" in str(exc_info.value)


class TestMetadataValidation:
    r"""Test validation of configuration metadata."""

    def test_required_metadata_fields(self):
        r"""Test that all required metadata fields are present."""
        config = BaseConfig(param1=1)
        assert_config_json_valid(config)

    def test_class_name_metadata(self):
        r"""Test that _class_name metadata is correct."""
        config = ConfigWithComplexTypes()
        assert config.config["_class_name"] == "ConfigWithComplexTypes"

    def test_use_default_values_metadata(self):
        r"""Test that _use_default_values is properly tracked."""
        config = BaseConfig(param1=100)  # Only param1 specified

        # param2 and param3 should be marked as using defaults
        defaults = config.config["_use_default_values"]
        assert "param2" in defaults
        assert "param3" in defaults
        assert "param1" not in defaults
