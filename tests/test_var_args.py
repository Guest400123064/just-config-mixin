#!/usr/bin/env python3

"""Tests for variable arguments (*args and **kwargs) handling in ConfigMixin.

This module tests edge cases and complex scenarios for variable positional
and keyword arguments, including their interaction with other ConfigMixin
features like ignored parameters, private parameters, and runtime_kwargs.
"""

import json
from typing import Any

import pytest

from configmixin import ConfigMixin, register_to_config


class ConfigOnlyVarArgs(ConfigMixin):
    """Configuration with only *args (no **kwargs)."""

    config_name = "config_only_args.json"

    @register_to_config
    def __init__(self, base_param: int = 10, *args):
        self.base_param = base_param
        self.args = args


class ConfigOnlyVarKwargs(ConfigMixin):
    """Configuration with only **kwargs (no *args)."""

    config_name = "config_only_kwargs.json"

    @register_to_config
    def __init__(self, base_param: int = 20, **kwargs):
        self.base_param = base_param
        self.kwargs = kwargs


class ConfigBothVarArgs(ConfigMixin):
    """Configuration with both *args and **kwargs."""

    config_name = "config_both_args.json"

    @register_to_config
    def __init__(self, base_param: int = 30, *args, **kwargs):
        self.base_param = base_param
        self.args = args
        self.kwargs = kwargs


class ConfigVarArgsWithKeywordOnly(ConfigMixin):
    """Configuration with *args and keyword-only parameters."""

    config_name = "config_keyword_only.json"

    @register_to_config
    def __init__(
        self, base_param: int = 40, *args, keyword_only: str = "default", **kwargs
    ):
        self.base_param = base_param
        self.args = args
        self.keyword_only = keyword_only
        self.kwargs = kwargs


class ConfigVarArgsWithIgnored(ConfigMixin):
    """Configuration with var args and ignored parameters."""

    config_name = "config_var_args_ignored.json"
    ignore_for_config = ["ignored_param", "runtime_obj"]

    @register_to_config
    def __init__(
        self,
        tracked_param: int = 50,
        *args,
        ignored_param: str = "ignored",
        another_tracked: float = 1.0,
        runtime_obj: Any = None,
        **kwargs,
    ):
        self.tracked_param = tracked_param
        self.args = args
        self.ignored_param = ignored_param
        self.another_tracked = another_tracked
        self.runtime_obj = runtime_obj
        self.kwargs = kwargs


class ConfigVarArgsWithPrivate(ConfigMixin):
    """Configuration with var args and private parameters."""

    config_name = "config_var_args_private.json"

    @register_to_config
    def __init__(
        self,
        public_param: int = 60,
        *args,
        _private_param: str,  # No default - should be required
        **kwargs,
    ):
        self.public_param = public_param
        self.args = args
        self._private_param = _private_param
        self.kwargs = kwargs


class ConfigComplexVarArgs(ConfigMixin):
    """Configuration with complex nested data in var args."""

    config_name = "config_complex_args.json"

    @register_to_config
    def __init__(self, base_param: int = 70, *args, **kwargs):
        self.base_param = base_param
        self.args = args
        self.kwargs = kwargs


class TestBasicVarArgsHandling:
    """Test basic variable arguments handling."""

    def test_only_var_args_empty(self, temp_dir):
        """Test config with only *args when no args provided."""
        config = ConfigOnlyVarArgs()

        config.save_config(temp_dir)
        loaded_config = ConfigOnlyVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 10
        assert loaded_config.args == ()

    def test_only_var_args_with_values(self, temp_dir):
        """Test config with only *args when args provided."""
        config = ConfigOnlyVarArgs(15, "arg1", "arg2", 42)

        config.save_config(temp_dir)
        loaded_config = ConfigOnlyVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 15
        assert loaded_config.args == ("arg1", "arg2", 42)

    def test_only_var_kwargs_empty(self, temp_dir):
        """Test config with only **kwargs when no kwargs provided."""
        config = ConfigOnlyVarKwargs()

        config.save_config(temp_dir)
        loaded_config = ConfigOnlyVarKwargs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 20
        assert loaded_config.kwargs == {}

    def test_only_var_kwargs_with_values(self, temp_dir):
        """Test config with only **kwargs when kwargs provided."""
        config = ConfigOnlyVarKwargs(25, extra1="value1", extra2=123, extra3=[1, 2, 3])

        config.save_config(temp_dir)
        loaded_config = ConfigOnlyVarKwargs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 25
        assert loaded_config.kwargs == {
            "extra1": "value1",
            "extra2": 123,
            "extra3": [1, 2, 3],
        }

    def test_both_var_args_empty(self, temp_dir):
        """Test config with both *args and **kwargs when none provided."""
        config = ConfigBothVarArgs()

        config.save_config(temp_dir)
        loaded_config = ConfigBothVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 30
        assert loaded_config.args == ()
        assert loaded_config.kwargs == {}

    def test_both_var_args_with_values(self, temp_dir):
        """Test config with both *args and **kwargs when both provided."""
        config = ConfigBothVarArgs(35, "arg1", "arg2", kwarg1="value1", kwarg2=456)

        config.save_config(temp_dir)
        loaded_config = ConfigBothVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 35
        assert loaded_config.args == ("arg1", "arg2")
        assert loaded_config.kwargs == {"kwarg1": "value1", "kwarg2": 456}


class TestKeywordOnlyParameters:
    """Test keyword-only parameters after *args."""

    def test_keyword_only_default_values(self, temp_dir):
        """Test keyword-only parameter with default value."""
        config = ConfigVarArgsWithKeywordOnly(45, "arg1", "arg2")

        config.save_config(temp_dir)
        loaded_config = ConfigVarArgsWithKeywordOnly.from_config(
            save_directory=temp_dir
        )

        assert loaded_config.base_param == 45
        assert loaded_config.args == ("arg1", "arg2")
        assert loaded_config.keyword_only == "default"
        assert loaded_config.kwargs == {}

    def test_keyword_only_custom_values(self, temp_dir):
        """Test keyword-only parameter with custom value."""
        config = ConfigVarArgsWithKeywordOnly(
            47, "arg1", "arg2", keyword_only="custom", extra_kwarg="extra"
        )

        config.save_config(temp_dir)
        loaded_config = ConfigVarArgsWithKeywordOnly.from_config(
            save_directory=temp_dir
        )

        assert loaded_config.base_param == 47
        assert loaded_config.args == ("arg1", "arg2")
        assert loaded_config.keyword_only == "custom"
        assert loaded_config.kwargs == {"extra_kwarg": "extra"}

    def test_schema_structure_with_keyword_only(self, temp_dir):
        """Test that schema correctly stores keyword-only params."""
        config = ConfigVarArgsWithKeywordOnly(
            48, "arg1", keyword_only="test", kw="value"
        )
        config.save_config(temp_dir)

        with open(temp_dir / "config_keyword_only.json") as f:
            saved_data = json.load(f)

        # keyword_only should be in main config, not in __notes__.kwargs
        assert "keyword_only" in saved_data
        assert saved_data["keyword_only"] == "test"

        # Extra kwargs should be in __notes__.kwargs
        notes = saved_data["__notes__"]
        assert notes["args"] == ["arg1"]
        assert notes["kwargs"] == {"kw": "value"}


class TestVarArgsWithIgnoredParams:
    """Test variable arguments interaction with ignored parameters."""

    def test_ignored_params_not_in_var_kwargs(self, temp_dir):
        """Test that ignored parameters don't appear in **kwargs."""
        mock_obj = {"type": "mock", "data": [1, 2, 3]}
        config = ConfigVarArgsWithIgnored(
            55,
            "arg1",
            "arg2",
            ignored_param="should_be_ignored",
            runtime_obj=mock_obj,
            tracked_kwarg="should_be_tracked",
        )

        config.save_config(temp_dir)

        # Check saved structure
        with open(temp_dir / "config_var_args_ignored.json") as f:
            saved_data = json.load(f)

        # Ignored params should not be saved anywhere
        assert "ignored_param" not in saved_data
        assert "runtime_obj" not in saved_data

        # Tracked params should be saved
        assert "tracked_param" in saved_data
        assert "another_tracked" in saved_data

        # Non-ignored kwargs should be in __notes__.kwargs
        notes = saved_data["__notes__"]
        assert notes["kwargs"] == {"tracked_kwarg": "should_be_tracked"}

        # Load with runtime_kwargs for ignored params
        runtime_kwargs = {"ignored_param": "should_be_ignored", "runtime_obj": mock_obj}
        loaded_config = ConfigVarArgsWithIgnored.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_config.tracked_param == 55
        assert loaded_config.args == ("arg1", "arg2")
        assert loaded_config.ignored_param == "should_be_ignored"
        assert loaded_config.runtime_obj == mock_obj
        assert loaded_config.kwargs == {"tracked_kwarg": "should_be_tracked"}


class TestVarArgsWithPrivateParams:
    """Test variable arguments interaction with private parameters."""

    def test_private_params_require_runtime_kwargs(self, temp_dir):
        """Test that private parameters must be passed via runtime_kwargs."""
        config = ConfigVarArgsWithPrivate(
            65, "arg1", "arg2", _private_param="secret", public_kwarg="public"
        )

        config.save_config(temp_dir)

        # Check that private param is not saved
        with open(temp_dir / "config_var_args_private.json") as f:
            saved_data = json.load(f)

        assert "_private_param" not in saved_data
        assert "public_param" in saved_data

        notes = saved_data["__notes__"]
        assert notes["kwargs"] == {"public_kwarg": "public"}

        # Should fail without runtime_kwargs for private param
        with pytest.raises(TypeError, match="missing.*required.*_private_param"):
            ConfigVarArgsWithPrivate.from_config(save_directory=temp_dir)

        # Should succeed with runtime_kwargs
        runtime_kwargs = {"_private_param": "secret"}
        loaded_config = ConfigVarArgsWithPrivate.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_config.public_param == 65
        assert loaded_config.args == ("arg1", "arg2")
        assert loaded_config._private_param == "secret"
        assert loaded_config.kwargs == {"public_kwarg": "public"}


class TestComplexVarArgsData:
    """Test variable arguments with complex nested data structures."""

    def test_nested_data_structures(self, temp_dir):
        """Test var args with lists, dicts, and nested structures."""
        complex_list = [1, 2, {"nested": "dict"}, [3, 4, 5]]
        complex_dict = {
            "level1": {"level2": ["a", "b", "c"], "number": 42},
            "list": [{"item1": "value1"}, {"item2": "value2"}],
        }

        config = ConfigComplexVarArgs(
            75,
            complex_list,
            "simple_string",
            123,
            nested_dict=complex_dict,
            simple_list=[1, 2, 3],
            mixed_data={"numbers": [1, 2, 3], "strings": ["a", "b"]},
        )

        config.save_config(temp_dir)
        loaded_config = ConfigComplexVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 75
        assert loaded_config.args == (complex_list, "simple_string", 123)
        assert loaded_config.kwargs["nested_dict"] == complex_dict
        assert loaded_config.kwargs["simple_list"] == [1, 2, 3]
        assert loaded_config.kwargs["mixed_data"] == {
            "numbers": [1, 2, 3],
            "strings": ["a", "b"],
        }

    def test_empty_collections_in_var_args(self, temp_dir):
        """Test var args with empty lists, dicts, tuples."""
        config = ConfigComplexVarArgs(
            80,
            [],  # empty list
            {},  # empty dict
            (),  # empty tuple
            empty_list=[],
            empty_dict={},
            none_value=None,
        )

        config.save_config(temp_dir)
        loaded_config = ConfigComplexVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 80
        # Note: tuples become lists after JSON serialization
        assert loaded_config.args == ([], {}, [])  # tuple becomes list
        assert loaded_config.kwargs["empty_list"] == []
        assert loaded_config.kwargs["empty_dict"] == {}
        assert loaded_config.kwargs["none_value"] is None

    def test_unicode_and_special_characters(self, temp_dir):
        """Test var args with unicode and special characters."""
        unicode_string = "Hello 世界 🌍 café naïve résumé"
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"

        config = ConfigComplexVarArgs(
            85,
            unicode_string,
            special_chars,
            unicode_key=unicode_string,
            special_key=special_chars,
            mixed=f"{unicode_string} {special_chars}",
        )

        config.save_config(temp_dir)
        loaded_config = ConfigComplexVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.args == (unicode_string, special_chars)
        assert loaded_config.kwargs["unicode_key"] == unicode_string
        assert loaded_config.kwargs["special_key"] == special_chars
        assert loaded_config.kwargs["mixed"] == f"{unicode_string} {special_chars}"


class TestVarArgsEdgeCases:
    """Test edge cases and error scenarios."""

    def test_var_args_with_default_values_tracking(self, temp_dir):
        """Test that default values are tracked correctly with var args."""
        # Create with all defaults to test default tracking
        config = ConfigBothVarArgs()  # Use all defaults
        config.save_config(temp_dir)

        with open(temp_dir / "config_both_args.json") as f:
            saved_data = json.load(f)

        notes = saved_data["__notes__"]
        assert "base_param" in notes["using_default_values"]

    def test_large_var_args_performance(self, temp_dir):
        """Test performance with large variable arguments."""
        # Create large but reasonable data
        large_args = list(range(1000))
        large_kwargs = {f"key_{i}": f"value_{i}" for i in range(100)}

        config = ConfigBothVarArgs(90, *large_args, **large_kwargs)

        config.save_config(temp_dir)
        loaded_config = ConfigBothVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 90
        assert loaded_config.args == tuple(large_args)
        assert loaded_config.kwargs == large_kwargs

    def test_var_args_serialization_edge_cases(self, temp_dir):
        """Test serialization of edge case values in var args."""
        import math

        # Note: NaN and Infinity might not roundtrip through JSON perfectly
        numeric_values = [0, -0, 1, -1, float("inf"), float("-inf")]
        boolean_values = [True, False]
        string_values = ["", "0", "false"]

        edge_values = numeric_values + boolean_values + string_values

        config = ConfigComplexVarArgs(
            95,
            *edge_values,
            edge_kwargs={
                "infinity": float("inf"),
                "negative_infinity": float("-inf"),
                "zero": 0,
                "false": False,
                "empty_string": "",
            },
        )

        config.save_config(temp_dir)
        loaded_config = ConfigComplexVarArgs.from_config(save_directory=temp_dir)

        assert loaded_config.base_param == 95
        # Check most values (only check finite numeric values for exact equality)
        finite_numeric = [v for v in numeric_values if math.isfinite(v)]
        loaded_finite_numeric = [
            v
            for v in loaded_config.args
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
        ]
        assert loaded_finite_numeric == finite_numeric

        # Check boolean and string values
        loaded_bools = [v for v in loaded_config.args if isinstance(v, bool)]
        loaded_strings = [v for v in loaded_config.args if isinstance(v, str)]
        assert loaded_bools == boolean_values
        assert loaded_strings == string_values
