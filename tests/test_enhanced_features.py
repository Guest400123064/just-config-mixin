#!/usr/bin/env python3

r"""Test suite for enhanced ConfigMixin features.

This module tests the enhanced functionality including:
- Loading config from dictionary instead of file
- Runtime kwargs for non-serializable parameters
- Integration of enhanced features with var args
- Round-trip serialization/deserialization
- Error handling for enhanced features
"""

import json
import pathlib
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest

from configmixin import ConfigMixin, register_to_config

from .conftest import (
    BaseConfig,
    ConfigWithBothVarArgs,
    ConfigWithRuntimeArgs,
    assert_config_roundtrip,
    create_config_dict,
    mock_model,
)


class ConfigWithVarArgsAndRuntime(ConfigMixin):
    r"""Configuration class with both var args and runtime arguments for testing."""

    config_name = "var_runtime_config.json"
    ignore_for_config = ["runtime_param"]

    @register_to_config
    def __init__(
        self,
        base_param: int = 100,
        *args,
        named_param: str = "named",
        runtime_param: Any = None,
        **kwargs,
    ):
        self.base_param = base_param
        self.args = args
        self.named_param = named_param
        self.runtime_param = runtime_param
        self.kwargs = kwargs


class TestFromConfigWithDict:
    r"""Test loading config from dictionary instead of file."""

    def test_from_config_with_dict_basic(self):
        r"""Test basic loading from config dict."""
        config_dict = create_config_dict(
            "BaseConfig",
            param1=42,
            param2="test_value",
            param3=[4, 5, 6]
        )

        instance = BaseConfig.from_config(config=config_dict)

        assert instance.param1 == 42
        assert instance.param2 == "test_value"
        assert instance.param3 == [4, 5, 6]

    def test_from_config_with_dict_missing_optional(self):
        r"""Test loading from dict with missing optional parameters."""
        config_dict = create_config_dict(
            "BaseConfig",
            param1=100,
            param2="partial",
            _use_default_values=["param3"]
        )

        instance = BaseConfig.from_config(config=config_dict)

        assert instance.param1 == 100
        assert instance.param2 == "partial"
        assert instance.param3 == [1, 2, 3]  # Should use default

    def test_from_config_with_dict_all_defaults(self):
        r"""Test loading from dict using all default values."""
        config_dict = create_config_dict(
            "BaseConfig",
            param1=10,
            param2="default",
            param3=None,
            _use_default_values=["param1", "param2", "param3"]
        )

        instance = BaseConfig.from_config(config=config_dict)

        assert instance.param1 == 10
        assert instance.param2 == "default"
        assert instance.param3 == [1, 2, 3]  # None gets processed to default

    def test_from_config_dict_vs_file_equivalence(self, temp_directory):
        r"""Test that loading from dict vs file produces equivalent results."""
        original = BaseConfig(param1=999, param2="equivalence", param3=[9, 8, 7])

        # Save to file
        original.save_config(temp_directory)

        # Get config as dict
        config_dict = json.loads(original.get_config_json())

        # Load from dict and file
        from_dict = BaseConfig.from_config(config=config_dict)
        from_file = BaseConfig.from_config(temp_directory)

        # Compare
        assert from_dict.param1 == from_file.param1 == original.param1
        assert from_dict.param2 == from_file.param2 == original.param2
        assert from_dict.param3 == from_file.param3 == original.param3
        assert from_dict.config == from_file.config == original.config


class TestFromConfigWithRuntimeKwargs:
    r"""Test from_config with runtime_kwargs parameter."""

    def test_runtime_kwargs_basic(self, mock_model):
        r"""Test basic runtime kwargs functionality."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            hidden_size=512,
            num_layers=6,
            learning_rate=0.01
        )

        runtime_kwargs = {
            "model": mock_model,
            "device": "cuda",
            "optimizer": "Adam"
        }

        instance = ConfigWithRuntimeArgs.from_config(
            config=config_dict, runtime_kwargs=runtime_kwargs
        )

        # Config params should be loaded from dict
        assert instance.hidden_size == 512
        assert instance.num_layers == 6
        assert instance.learning_rate == 0.01

        # Runtime params should come from runtime_kwargs
        assert instance.model is mock_model
        assert instance.device == "cuda"
        assert instance.optimizer == "Adam"

    def test_runtime_kwargs_override_defaults(self):
        r"""Test that runtime kwargs can override default values."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            learning_rate=0.005,
            _use_default_values=["hidden_size", "num_layers"]
        )

        runtime_kwargs = {
            "device": "gpu",  # Override default
            "model": Mock(name="custom_model"),
        }

        instance = ConfigWithRuntimeArgs.from_config(
            config=config_dict, runtime_kwargs=runtime_kwargs
        )

        assert instance.learning_rate == 0.005
        assert instance.hidden_size == 768  # Default
        assert instance.num_layers == 12  # Default
        assert instance.device == "gpu"  # From runtime_kwargs
        assert instance.model.name == "custom_model"

    def test_runtime_kwargs_empty(self):
        r"""Test from_config with empty runtime_kwargs."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            hidden_size=256,
            num_layers=8,
            learning_rate=0.002
        )

        instance = ConfigWithRuntimeArgs.from_config(
            config=config_dict, runtime_kwargs={}
        )

        assert instance.hidden_size == 256
        assert instance.model is None  # Default
        assert instance.device == "cpu"  # Default

    def test_runtime_kwargs_none(self):
        r"""Test from_config with runtime_kwargs=None."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            learning_rate=0.003,
            _use_default_values=["hidden_size", "num_layers"]
        )

        instance = ConfigWithRuntimeArgs.from_config(
            config=config_dict, runtime_kwargs=None
        )

        assert instance.learning_rate == 0.003
        assert instance.model is None
        assert instance.device == "cpu"


class TestFromConfigCombined:
    r"""Test from_config with both dict and runtime_kwargs."""

    def test_dict_and_runtime_kwargs_together(self):
        r"""Test using both config dict and runtime kwargs."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            hidden_size=1024,
            num_layers=24,
            learning_rate=0.0001
        )

        mock_model = Mock(name="combined_model")
        runtime_kwargs = {
            "model": mock_model,
            "device": "tpu",
            "optimizer": "RMSprop"
        }

        instance = ConfigWithRuntimeArgs.from_config(
            config=config_dict, runtime_kwargs=runtime_kwargs
        )

        # Config values
        assert instance.hidden_size == 1024
        assert instance.num_layers == 24
        assert instance.learning_rate == 0.0001

        # Runtime values
        assert instance.model is mock_model
        assert instance.device == "tpu"
        assert instance.optimizer == "RMSprop"

    def test_file_and_runtime_kwargs(self, temp_directory):
        r"""Test loading from file with runtime kwargs."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            hidden_size=512,
            num_layers=16,
            learning_rate=0.001
        )

        config_file = temp_directory / "config_runtime.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

        mock_model = Mock(name="file_model")
        runtime_kwargs = {
            "model": mock_model,
            "device": "multi_gpu",
        }

        instance = ConfigWithRuntimeArgs.from_config(
            save_directory=temp_directory, runtime_kwargs=runtime_kwargs
        )

        assert instance.hidden_size == 512
        assert instance.num_layers == 16
        assert instance.device == "multi_gpu"
        assert instance.model is mock_model


class TestFromConfigWithVarArgs:
    r"""Test from_config with var args and runtime kwargs."""

    def test_var_args_with_runtime_kwargs(self):
        r"""Test loading config with var args and runtime kwargs."""
        config_dict = create_config_dict(
            "ConfigWithVarArgsAndRuntime",
            base_param=200,
            named_param="var_test",
            _var_positional=["arg1", "arg2", 42],
            _var_keyword={"extra1": "value1", "extra2": [1, 2, 3]}
        )

        mock_runtime = Mock(name="var_runtime")
        runtime_kwargs = {"runtime_param": mock_runtime}

        instance = ConfigWithVarArgsAndRuntime.from_config(
            config=config_dict, runtime_kwargs=runtime_kwargs
        )

        # Regular params
        assert instance.base_param == 200
        assert instance.named_param == "var_test"

        # Var args should be restored
        assert instance.args == ("arg1", "arg2", 42)
        assert instance.kwargs == {"extra1": "value1", "extra2": [1, 2, 3]}

        # Runtime param
        assert instance.runtime_param is mock_runtime

    def test_var_args_missing_metadata(self):
        r"""Test from_config when var args metadata is missing (backward compatibility)."""
        # Config without var args metadata (old format)
        config_dict = create_config_dict(
            "ConfigWithBothVarArgs",
            base_param=50,
            named_param="old_format"
        )

        # Remove var args metadata to simulate old config
        config_dict.pop("_var_positional", None)
        config_dict.pop("_var_keyword", None)

        instance = ConfigWithBothVarArgs.from_config(config=config_dict)

        # Should work with empty var args
        assert instance.base_param == 50
        assert instance.named_param == "old_format"
        assert instance.args == ()
        assert instance.kwargs == {}

    def test_var_args_with_none_values(self):
        r"""Test from_config when var args contain None."""
        config_dict = create_config_dict(
            "ConfigWithBothVarArgs",
            base_param=75,
            named_param="none_test",
            _var_positional=[None, "not_none", None],
            _var_keyword={"none_key": None, "value_key": "value"}
        )

        instance = ConfigWithBothVarArgs.from_config(config=config_dict)

        assert instance.base_param == 75
        assert instance.args == (None, "not_none", None)
        assert instance.kwargs == {"none_key": None, "value_key": "value"}


class TestFromConfigErrorHandling:
    r"""Test error handling in enhanced from_config."""

    def test_missing_both_directory_and_config(self):
        r"""Test error when neither save_directory nor config is provided."""
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.from_config()

        assert "Either `save_directory` or `config` must be provided" in str(exc_info.value)

    def test_wrong_class_name_in_config(self):
        r"""Test error when config has wrong class name."""
        config_dict = create_config_dict("WrongClassName", param1=42)

        with pytest.raises(ValueError) as exc_info:
            BaseConfig.from_config(config=config_dict)

        assert "is not a config for BaseConfig" in str(exc_info.value)

    def test_missing_required_param_in_config(self):
        r"""Test error when config is missing required parameters."""
        class RequiredParamConfig(ConfigMixin):
            config_name = "required.json"

            @register_to_config
            def __init__(self, required_param: str, optional_param: int = 10):
                self.required_param = required_param
                self.optional_param = optional_param

        config_dict = create_config_dict(
            "RequiredParamConfig",
            optional_param=20
        )

        with pytest.raises(ValueError) as exc_info:
            RequiredParamConfig.from_config(config=config_dict)

        assert "missing required parameter(s)" in str(exc_info.value)

    def test_unexpected_param_in_config(self):
        r"""Test error when config contains unexpected parameters."""
        config_dict = create_config_dict(
            "BaseConfig",
            param1=42,
            param2="test",
            param3=[1, 2, 3],
            unexpected_param="should_cause_error"
        )

        with pytest.raises(ValueError) as exc_info:
            BaseConfig.from_config(config=config_dict)

        assert "unexpected parameter: unexpected_param" in str(exc_info.value)

    def test_runtime_kwargs_dont_interfere_with_var_kwargs(self):
        r"""Test that runtime_kwargs don't interfere with var_kwargs."""
        config_dict = create_config_dict(
            "ConfigWithVarArgsAndRuntime",
            base_param=75,
            named_param="runtime_test",
            _var_positional=["arg1"],
            _var_keyword={"config_kwarg": "from_config"}
        )

        runtime_kwargs = {"runtime_param": "runtime_value"}

        instance = ConfigWithVarArgsAndRuntime.from_config(
            config=config_dict,
            runtime_kwargs=runtime_kwargs
        )

        # Config var_kwargs should be preserved
        assert instance.kwargs == {"config_kwarg": "from_config"}
        assert instance.args == ("arg1",)

        # Runtime kwargs should set ignored attributes
        assert instance.runtime_param == "runtime_value"


class TestRoundTripSerialization:
    r"""Test round-trip serialization/deserialization with enhancements."""

    def test_round_trip_with_runtime_kwargs(self):
        r"""Test that config can be saved and loaded with runtime kwargs."""
        # Create instance with runtime args
        original_runtime = Mock(name="original")
        original = ConfigWithRuntimeArgs(
            hidden_size=2048,
            num_layers=32,
            learning_rate=0.0005,
            model=original_runtime,
            device="original_device",
        )

        # Save to get config dict
        config_dict = json.loads(original.get_config_json())

        # Load with different runtime kwargs
        new_runtime = Mock(name="reloaded")
        loaded = ConfigWithRuntimeArgs.from_config(
            config=config_dict,
            runtime_kwargs={
                "model": new_runtime,
                "device": "new_device",
            },
        )

        # Config params should match
        assert loaded.hidden_size == original.hidden_size
        assert loaded.num_layers == original.num_layers
        assert loaded.learning_rate == original.learning_rate

        # Runtime params should be from runtime_kwargs
        assert loaded.model is new_runtime
        assert loaded.device == "new_device"

    def test_round_trip_var_args_with_runtime(self):
        r"""Test round trip with var args and runtime kwargs."""
        original_runtime = Mock(name="var_original")
        original = ConfigWithVarArgsAndRuntime(
            100,
            "arg1",
            "arg2",
            named_param="round_trip_var",
            runtime_param=original_runtime,
            extra1="value1",
            extra2=42,
        )

        config_dict = json.loads(original.get_config_json())

        new_runtime = Mock(name="var_reloaded")
        loaded = ConfigWithVarArgsAndRuntime.from_config(
            config=config_dict, runtime_kwargs={"runtime_param": new_runtime}
        )

        # Check regular params
        assert loaded.base_param == 100
        assert loaded.named_param == "round_trip_var"

        # Check var args restoration
        assert loaded.args == ("arg1", "arg2")
        assert loaded.kwargs == {"extra1": "value1", "extra2": 42}

        # Check runtime param
        assert loaded.runtime_param is new_runtime

    def test_config_preservation_across_save_load(self, temp_directory):
        r"""Test that all config aspects are preserved during save/load."""
        config = ConfigWithRuntimeArgs(
            hidden_size=1536,
            num_layers=18,
            learning_rate=0.0003,
            model=Mock(name="preserved"),
            device="preserved_device"
        )

        # Test round trip
        assert_config_roundtrip(config)

    def test_enhanced_features_compatibility(self, temp_directory):
        r"""Test that enhanced features work with traditional save/load."""
        # Create config with var args
        original = ConfigWithBothVarArgs(
            50, "compat1", "compat2",
            named_param="compatibility",
            extra_kw1="compat_value1",
            extra_kw2="compat_value2"
        )

        # Save traditionally
        original.save_config(temp_directory)

        # Load with enhanced method (dict + runtime_kwargs)
        config_dict = json.loads(original.get_config_json())
        loaded = ConfigWithBothVarArgs.from_config(
            config=config_dict,
            runtime_kwargs={}  # Empty runtime kwargs
        )

        # Should be identical
        assert loaded.base_param == original.base_param
        assert loaded.named_param == original.named_param
        assert loaded.args == original.args
        assert loaded.kwargs == original.kwargs
        assert loaded.config == original.config


class TestEnhancedMetadataHandling:
    r"""Test enhanced metadata handling in new features."""

    def test_var_args_metadata_in_dict_loading(self):
        r"""Test that var args metadata is properly handled when loading from dict."""
        config_dict = create_config_dict(
            "ConfigWithBothVarArgs",
            base_param=123,
            named_param="metadata_test",
            _var_positional=["meta1", "meta2"],
            _var_keyword={"meta_kw": "meta_value"}
        )

        instance = ConfigWithBothVarArgs.from_config(config=config_dict)

        # Check metadata is properly restored
        assert instance.config["_var_positional"] == ("meta1", "meta2")
        assert instance.config["_var_keyword"] == {"meta_kw": "meta_value"}
        assert instance.config["_class_name"] == "ConfigWithBothVarArgs"

    def test_use_default_values_with_runtime_kwargs(self):
        r"""Test that _use_default_values works correctly with runtime kwargs."""
        config_dict = create_config_dict(
            "ConfigWithRuntimeArgs",
            learning_rate=0.007,
            _use_default_values=["hidden_size", "num_layers"]
        )

        instance = ConfigWithRuntimeArgs.from_config(
            config=config_dict,
            runtime_kwargs={"device": "runtime_device"}
        )

        # Default tracking should be preserved
        defaults = instance.config["_use_default_values"]
        assert "hidden_size" in defaults
        assert "num_layers" in defaults
        assert "learning_rate" not in defaults

        # Values should be correct
        assert instance.hidden_size == 768  # Default
        assert instance.num_layers == 12   # Default
        assert instance.learning_rate == 0.007  # From config
        assert instance.device == "runtime_device"  # From runtime
