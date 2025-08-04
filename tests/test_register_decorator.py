#!/usr/bin/env python3

r"""Test suite for @register_to_config decorator functionality.

This module specifically tests the decorator behavior and integration
with new API features. Core functionality is tested in test_core_functionality.py.
"""

import json

from configmixin import ConfigMixin, register_to_config

from .conftest import (
    BaseConfig,
    ConfigWithBothVarArgs,
    ConfigWithVarArgs,
    create_config_dict,
)


class DecoratorWithVarArgsOnly(ConfigMixin):
    r"""Test class specifically for decorator with var args."""

    config_name = "decorator_var_args.json"

    @register_to_config
    def __init__(self, base: int = 10, *args):
        self.base = base
        self.args = args


class DecoratorWithKwargsOnly(ConfigMixin):
    r"""Test class specifically for decorator with kwargs."""

    config_name = "decorator_kwargs.json"

    @register_to_config
    def __init__(self, base: int = 20, **kwargs):
        self.base = base
        self.kwargs = kwargs


class TestDecoratorWithNewAPIFeatures:
    r"""Test decorator behavior with new API features."""

    def test_decorator_with_var_positional_only(self):
        r"""Test decorator with only var positional args."""
        instance = DecoratorWithVarArgsOnly(100, "arg1", "arg2", 42)

        assert instance.config["base"] == 100
        assert instance.config["_var_positional"] == ("arg1", "arg2", 42)
        assert instance.config["_var_keyword"] == {}
        assert instance.args == ("arg1", "arg2", 42)

    def test_decorator_with_var_keyword_only(self):
        r"""Test decorator with only var keyword args."""
        instance = DecoratorWithKwargsOnly(200, extra1="value1", extra2=42)

        assert instance.config["base"] == 200
        assert instance.config["_var_positional"] == ()
        assert instance.config["_var_keyword"] == {"extra1": "value1", "extra2": 42}
        assert instance.kwargs == {"extra1": "value1", "extra2": 42}

    def test_decorator_with_both_var_args(self):
        r"""Test decorator with both var positional and keyword args."""
        instance = ConfigWithBothVarArgs(
            300, "arg1", "arg2", named_param="test", kw1="val1", kw2="val2"
        )

        assert instance.config["base_param"] == 300
        assert instance.config["named_param"] == "test"
        assert instance.config["_var_positional"] == ("arg1", "arg2")
        assert instance.config["_var_keyword"] == {"kw1": "val1", "kw2": "val2"}

    def test_decorator_metadata_with_var_args(self):
        r"""Test that decorator properly sets metadata with var args."""
        instance = ConfigWithBothVarArgs(400, "meta_arg", meta_kw="meta_value")

        # Check all required metadata fields
        assert instance.config["_class_name"] == "ConfigWithBothVarArgs"
        assert "_use_default_values" in instance.config
        assert "_var_positional" in instance.config
        assert "_var_keyword" in instance.config

        # Check metadata values
        assert instance.config["_var_positional"] == ("meta_arg",)
        assert instance.config["_var_keyword"] == {"meta_kw": "meta_value"}


class TestNewAPICompatibility:
    r"""Test new API compatibility with decorator."""

    def test_decorator_creates_required_metadata(self):
        r"""Test that decorator creates all required metadata fields."""
        instance = ConfigWithVarArgs(50, "test_arg", extra_param="test_extra")

        # All new metadata fields should be present
        required_fields = [
            "_class_name",
            "_use_default_values",
            "_var_positional",
            "_var_keyword",
        ]
        for field in required_fields:
            assert field in instance.config, f"Missing required field: {field}"

        # Values should be correct
        assert instance.config["_class_name"] == "ConfigWithVarArgs"
        assert instance.config["_var_positional"] == ("test_arg",)
        assert instance.config["_var_keyword"] == {}

    def test_decorator_with_from_config_dict(self):
        r"""Test that decorator works with from_config dict parameter."""
        config_dict = create_config_dict(
            "ConfigWithVarArgs",
            base_param=99,
            extra_param="from_dict",
            _var_positional=["dict_arg1", "dict_arg2"],
            _var_keyword={},
        )

        instance = ConfigWithVarArgs.from_config(config=config_dict)

        assert instance.base_param == 99
        assert instance.extra_param == "from_dict"
        assert instance.args == ("dict_arg1", "dict_arg2")

    def test_decorator_preserves_backward_compatibility(self):
        r"""Test that decorator maintains backward compatibility."""
        # Test that old-style configs (without var args metadata) still work
        instance = BaseConfig(param1=123, param2="backward_compat")

        # Should have new metadata fields with empty var args
        assert instance.config["_var_positional"] == ()
        assert instance.config["_var_keyword"] == {}
        assert instance.config["_class_name"] == "BaseConfig"

    def test_decorator_json_serialization_with_var_args(self):
        r"""Test that decorator properly handles JSON serialization with var args."""
        instance = ConfigWithBothVarArgs(
            999,
            "json_arg1",
            "json_arg2",
            named_param="json_test",
            json_kw1="json_value1",
            json_kw2=42,
        )

        json_str = instance.config_dumps()
        config_dict = json.loads(json_str)

        # Should properly serialize var args
        assert config_dict["_var_positional"] == ["json_arg1", "json_arg2"]
        assert config_dict["_var_keyword"] == {
            "json_kw1": "json_value1",
            "json_kw2": 42,
        }
        assert config_dict["base_param"] == 999
        assert config_dict["named_param"] == "json_test"


class TestDecoratorBehaviorSpecifics:
    r"""Test specific decorator behavior and mechanics."""

    def test_decorator_parameter_capture_order(self):
        r"""Test that decorator captures parameters in correct order."""
        instance = ConfigWithBothVarArgs(
            100, "first", "second", named_param="middle", last_kw="last"
        )

        # Positional args should maintain order
        assert instance.config["_var_positional"] == ("first", "second")

        # Regular params should be captured
        assert instance.config["base_param"] == 100
        assert instance.config["named_param"] == "middle"

        # Keyword args should be captured
        assert instance.config["_var_keyword"] == {"last_kw": "last"}

    def test_decorator_default_value_tracking_with_var_args(self):
        r"""Test default value tracking works correctly with var args."""
        # Use some defaults
        instance = ConfigWithBothVarArgs("var_arg", named_param="custom")

        # base_param should NOT be marked as using default (it was explicitly provided as "var_arg")
        assert "base_param" not in instance.config["_use_default_values"]
        assert "named_param" not in instance.config["_use_default_values"]

        # Var args should be empty (no extra positional args provided after filling base_param)
        assert instance.config["_var_positional"] == ()

    def test_decorator_signature_inspection_with_var_args(self):
        r"""Test that decorator correctly inspects signatures with var args."""
        # This tests the internal signature inspection logic
        instance = DecoratorWithVarArgsOnly(500, "inspect1", "inspect2")

        assert instance.config["base"] == 500
        assert instance.config["_var_positional"] == ("inspect1", "inspect2")
        assert instance.config["_var_keyword"] == {}

        # base should not be in use_default_values since it was provided
        assert "base" not in instance.config["_use_default_values"]

    def test_decorator_with_empty_var_args(self):
        r"""Test decorator behavior when var args are empty."""
        instance = DecoratorWithVarArgsOnly(600)

        assert instance.config["base"] == 600
        assert instance.config["_var_positional"] == ()
        assert instance.config["_var_keyword"] == {}

        # Only base was provided
        assert instance.config["_use_default_values"] == []
