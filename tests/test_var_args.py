#!/usr/bin/env python3

r"""Test suite for var positional and var keyword argument tracking in ConfigMixin.

This module tests the new var args functionality including:
- Tracking of *args in _var_positional
- Tracking of **kwargs in _var_keyword
- Integration with ignored parameters
- JSON serialization of var args
- Edge cases and complex types
"""

import json
import pathlib

from configmixin import ConfigMixin, register_to_config

from .conftest import (
    ConfigWithBothVarArgs,
    ConfigWithVarArgs,
    ConfigWithVarKwargs,
    MockSerializableObject,
)


class ConfigWithIgnoredVarArgs(ConfigMixin):
    r"""Configuration class with ignored var args for testing."""

    config_name = "ignored_var_args_config.json"
    ignore_for_config = ["extra_param"]

    @register_to_config
    def __init__(
        self, base_param: int = 40, *args, extra_param: str = "ignore", **kwargs
    ):
        self.base_param = base_param
        self.args = args
        self.extra_param = extra_param
        self.kwargs = kwargs


class ConfigWithPrivateVarArgs(ConfigMixin):
    r"""Configuration class with private var args for testing."""

    config_name = "private_var_args.json"

    @register_to_config
    def __init__(
        self,
        public: int = 1,
        *_private_args,
        _private_kw: str = "private",
        **_private_kwargs,
    ):
        self.public = public
        self._private_args = _private_args
        self._private_kw = _private_kw
        self._private_kwargs = _private_kwargs


class TestVarPositionalArgs:
    r"""Test var positional argument tracking."""

    def test_no_var_positional_args(self):
        r"""Test when no extra positional args are provided."""
        config = ConfigWithVarArgs(base_param=100, extra_param="test")

        assert config.config["base_param"] == 100
        assert config.config["extra_param"] == "test"
        assert config.config["_var_positional"] == ()
        assert config.args == ()

    def test_with_var_positional_args(self):
        r"""Test when extra positional args are provided."""
        config = ConfigWithVarArgs(100, "arg1", "arg2", 42, extra_param="test")

        assert config.config["base_param"] == 100
        assert config.config["extra_param"] == "test"
        assert config.config["_var_positional"] == ("arg1", "arg2", 42)
        assert config.args == ("arg1", "arg2", 42)

    def test_mixed_positional_and_keyword_with_varargs(self):
        r"""Test mixing positional and keyword args with *args."""
        config = ConfigWithVarArgs(200, "extra1", "extra2", extra_param="mixed")

        assert config.config["base_param"] == 200
        assert config.config["extra_param"] == "mixed"
        assert config.config["_var_positional"] == ("extra1", "extra2")
        assert config.args == ("extra1", "extra2")

    def test_complex_var_positional_types(self):
        r"""Test var positional args with complex types."""
        complex_data = [1, 2, 3]
        dict_data = {"nested": "value"}
        config = ConfigWithVarArgs(300, complex_data, dict_data, extra_param="complex")

        assert config.config["_var_positional"] == ([1, 2, 3], {"nested": "value"})
        assert config.args == ([1, 2, 3], {"nested": "value"})


class TestVarKeywordArgs:
    r"""Test var keyword argument tracking."""

    def test_no_var_keyword_args(self):
        r"""Test when no extra keyword args are provided."""
        config = ConfigWithVarKwargs(base_param=100)

        assert config.config["base_param"] == 100
        assert config.config["_var_keyword"] == {}
        assert config.kwargs == {}

    def test_with_var_keyword_args(self):
        r"""Test when extra keyword args are provided."""
        config = ConfigWithVarKwargs(
            base_param=200, extra1="value1", extra2=42, extra3=True
        )

        assert config.config["base_param"] == 200
        assert config.config["_var_keyword"] == {
            "extra1": "value1",
            "extra2": 42,
            "extra3": True,
        }
        assert config.kwargs == {"extra1": "value1", "extra2": 42, "extra3": True}

    def test_complex_var_keyword_types(self):
        r"""Test var keyword args with complex types."""
        config = ConfigWithVarKwargs(
            base_param=300,
            list_data=[1, 2, 3],
            dict_data={"nested": "structure"},
            none_data=None,
        )

        expected_var_kwargs = {
            "list_data": [1, 2, 3],
            "dict_data": {"nested": "structure"},
            "none_data": None,
        }
        assert config.config["_var_keyword"] == expected_var_kwargs
        assert config.kwargs == expected_var_kwargs


class TestBothVarArgs:
    r"""Test both *args and **kwargs together."""

    def test_both_var_args_empty(self):
        r"""Test with no extra args or kwargs."""
        config = ConfigWithBothVarArgs(base_param=100, named_param="test")

        assert config.config["base_param"] == 100
        assert config.config["named_param"] == "test"
        assert config.config["_var_positional"] == ()
        assert config.config["_var_keyword"] == {}

    def test_both_var_args_populated(self):
        r"""Test with both extra args and kwargs."""
        config = ConfigWithBothVarArgs(
            100, "arg1", "arg2", named_param="test", extra_kw1="value1", extra_kw2=42
        )

        assert config.config["base_param"] == 100
        assert config.config["named_param"] == "test"
        assert config.config["_var_positional"] == ("arg1", "arg2")
        assert config.config["_var_keyword"] == {"extra_kw1": "value1", "extra_kw2": 42}
        assert config.args == ("arg1", "arg2")
        assert config.kwargs == {"extra_kw1": "value1", "extra_kw2": 42}

    def test_only_var_positional(self):
        r"""Test with only var positional args."""
        config = ConfigWithBothVarArgs(200, "arg1", "arg2", named_param="only_pos")

        assert config.config["_var_positional"] == ("arg1", "arg2")
        assert config.config["_var_keyword"] == {}

    def test_only_var_keyword(self):
        r"""Test with only var keyword args."""
        config = ConfigWithBothVarArgs(
            base_param=300, named_param="only_kw", extra1="value1", extra2="value2"
        )

        assert config.config["_var_positional"] == ()
        assert config.config["_var_keyword"] == {"extra1": "value1", "extra2": "value2"}


class TestVarArgsWithIgnored:
    r"""Test var args with ignored parameters."""

    def test_ignored_params_not_in_var_kwargs(self):
        r"""Test that ignored params don't appear in _var_keyword."""
        config = ConfigWithIgnoredVarArgs(
            100,
            "arg1",
            "arg2",
            extra_param="should_be_ignored",
            extra_kw="should_appear",
        )

        assert config.config["base_param"] == 100
        assert "extra_param" not in config.config
        assert config.config["_var_positional"] == ("arg1", "arg2")
        assert config.config["_var_keyword"] == {"extra_kw": "should_appear"}

        # But the attribute should still be set
        assert config.extra_param == "should_be_ignored"
        assert config.kwargs == {"extra_kw": "should_appear"}


class TestPrivateVarArgs:
    r"""Test private var args functionality."""

    def test_private_var_args_ignored(self):
        r"""Test that private var args are completely ignored."""
        config = ConfigWithPrivateVarArgs(
            5,
            "private_arg1",
            "private_arg2",
            _private_kw="private_value",
            _private_kwarg1="private1",
            public_kwarg="public",
        )

        # Only public param should be in config
        assert config.config["public"] == 5
        assert "_private_args" not in config.config
        assert "_private_kw" not in config.config
        assert "_private_kwargs" not in config.config

        # Var args should be empty since they were private
        assert dict(config.config["_var_keyword"]) == {"public_kwarg": "public"}

        # But private attributes should still be set
        assert config._private_args == ("private_arg1", "private_arg2")
        assert config._private_kw == "private_value"
        assert config._private_kwargs == {"_private_kwarg1": "private1", "public_kwarg": "public"}


class TestVarArgsMetadata:
    r"""Test metadata handling for var args."""

    def test_class_name_in_config(self):
        r"""Test that _class_name is properly set with var args."""
        config = ConfigWithBothVarArgs(100, "arg", extra_kw="value")

        assert config.config["_class_name"] == "ConfigWithBothVarArgs"

    def test_use_default_values_tracking(self):
        r"""Test default value tracking with var args."""
        # Use all defaults except var args
        config = ConfigWithBothVarArgs("arg1", extra_kw="value")

        expected_defaults = ["named_param"]
        assert set(config.config["_use_default_values"]) == set(expected_defaults)

        # Override some defaults
        config2 = ConfigWithBothVarArgs(
            500, "arg1", named_param="custom", extra_kw="value"
        )

        assert config2.config["_use_default_values"] == []


class TestVarArgsJsonSerialization:
    r"""Test JSON serialization with var args."""

    def test_var_args_in_json(self):
        r"""Test that var args are properly serialized to JSON."""
        config = ConfigWithBothVarArgs(
            100,
            "arg1",
            {"nested": "data"},
            named_param="test",
            extra_list=[1, 2, 3],
            extra_dict={"key": "value"},
        )

        json_str = config.config_dumps()
        config_dict = json.loads(json_str)

        assert config_dict["_class_name"] == "ConfigWithBothVarArgs"
        assert config_dict["base_param"] == 100
        assert config_dict["named_param"] == "test"
        assert config_dict["_var_positional"] == ["arg1", {"nested": "data"}]
        assert config_dict["_var_keyword"] == {
            "extra_list": [1, 2, 3],
            "extra_dict": {"key": "value"},
        }

    def test_empty_var_args_in_json(self):
        r"""Test serialization when var args are empty."""
        config = ConfigWithBothVarArgs(base_param=200, named_param="empty")
        json_str = config.config_dumps()
        config_dict = json.loads(json_str)

        assert config_dict["_var_positional"] == []
        assert config_dict["_var_keyword"] == {}

    def test_var_args_with_pathlib_objects(self):
        r"""Test var args containing pathlib objects."""
        path1 = pathlib.Path("/data/input")
        path2 = pathlib.Path("/data/output")

        config = ConfigWithBothVarArgs(
            100,
            path1,
            path2,
            named_param="pathlib_test",
            output_path=pathlib.Path("/results"),
            config_path=pathlib.Path("/config"),
        )

        # Paths should be serialized as POSIX strings
        json_str = config.config_dumps()
        config_dict = json.loads(json_str)

        assert config_dict["_var_positional"] == ["/data/input", "/data/output"]
        assert config_dict["_var_keyword"]["output_path"] == "/results"
        assert config_dict["_var_keyword"]["config_path"] == "/config"

    def test_var_args_with_serializable_objects(self):
        r"""Test var args with objects that have to_dict method."""
        obj1 = MockSerializableObject("test1")
        obj2 = MockSerializableObject("test2")

        config = ConfigWithBothVarArgs(
            100, obj1, named_param="serializable", obj_kwarg=obj2
        )

        json_str = config.config_dumps()
        config_dict = json.loads(json_str)

        # Objects with to_dict should be serialized using that method
        expected_obj1 = {"mock_value": "test1", "type": "MockSerializableObject"}
        expected_obj2 = {"mock_value": "test2", "type": "MockSerializableObject"}

        assert config_dict["_var_positional"][0] == expected_obj1
        assert config_dict["_var_keyword"]["obj_kwarg"] == expected_obj2


class TestVarArgsEdgeCases:
    r"""Test edge cases for var args."""

    def test_var_args_with_none_values(self):
        r"""Test var args containing None values."""
        config = ConfigWithBothVarArgs(
            100, None, "arg2", named_param="none_test", none_kw=None
        )

        assert config.config["_var_positional"] == (None, "arg2")
        assert config.config["_var_keyword"] == {"none_kw": None}

    def test_var_args_order_preservation(self):
        r"""Test that var args preserve order."""
        config = ConfigWithVarArgs(100, "first", "second", "third", extra_param="order")

        assert config.config["_var_positional"] == ("first", "second", "third")
        assert config.args == ("first", "second", "third")

    def test_var_kwargs_order_preservation(self):
        r"""Test that var kwargs preserve order in FrozenDict."""
        config = ConfigWithVarKwargs(
            base_param=100, z_last="z", a_first="a", m_middle="m"
        )

        # Order should be preserved as provided
        var_kwargs = config.config["_var_keyword"]
        assert list(var_kwargs.keys()) == ["z_last", "a_first", "m_middle"]

    def test_complex_types_in_var_args(self):
        r"""Test complex types in var args."""
        complex_list = [{"nested": "dict"}, [1, 2, 3]]
        complex_dict = {"path": pathlib.Path("/tmp"), "nested": {"deep": "value"}}

        config = ConfigWithBothVarArgs(
            100,
            complex_list,
            complex_dict,
            named_param="complex",
            list_kwarg=[pathlib.Path("/data"), {"more": "data"}],
            none_kwarg=None,
            bool_kwarg=True,
        )

        assert config.config["_var_positional"] == (complex_list, complex_dict)
        expected_kwargs = {
            "list_kwarg": [pathlib.Path("/data"), {"more": "data"}],
            "none_kwarg": None,
            "bool_kwarg": True,
        }
        assert config.config["_var_keyword"] == expected_kwargs
