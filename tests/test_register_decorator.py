#!/usr/bin/env python3

"""
Test suite for register_to_config decorator functionality.
"""

import pytest
from typing import List, Dict, Optional

from yacm import ConfigMixin, register_to_config, FrozenDict


class SampleDecoratorBasic(ConfigMixin):
    """Basic test class using the decorator."""

    config_name = "decorator_basic.json"

    @register_to_config
    def __init__(self, param1: int = 10, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2


class SampleDecoratorWithIgnored(ConfigMixin):
    """Test class with ignored parameters."""

    config_name = "decorator_ignored.json"
    ignore_for_config = ["ignored_param", "also_ignored"]

    @register_to_config
    def __init__(self,
                 tracked_param: int = 5,
                 ignored_param: str = "ignored",
                 also_ignored: bool = True,
                 another_tracked: float = 1.0):
        self.tracked_param = tracked_param
        self.ignored_param = ignored_param
        self.also_ignored = also_ignored
        self.another_tracked = another_tracked


class SampleDecoratorWithPrivate(ConfigMixin):
    """Test class with private parameters (starting with underscore)."""

    config_name = "decorator_private.json"

    @register_to_config
    def __init__(self,
                 public_param: int = 10,
                 _private_param: str = "private",
                 normal_param: float = 2.5):
        self.public_param = public_param
        self._private_param = _private_param
        self.normal_param = normal_param


class SampleDecoratorComplexTypes(ConfigMixin):
    """Test class with complex parameter types."""

    config_name = "decorator_complex.json"

    @register_to_config
    def __init__(self,
                 list_param: List[int] = None,
                 dict_param: Dict[str, str] = None,
                 optional_param: Optional[str] = None,
                 default_list: list = None):
        self.list_param = list_param or [1, 2, 3]
        self.dict_param = dict_param or {"key": "value"}
        self.optional_param = optional_param
        self.default_list = default_list or []


class SampleDecoratorNoDefaults(ConfigMixin):
    """Test class with required parameters (no defaults)."""

    config_name = "decorator_no_defaults.json"

    @register_to_config
    def __init__(self, required_param: int, required_str: str, optional_param: bool = True):
        self.required_param = required_param
        self.required_str = required_str
        self.optional_param = optional_param


class NotConfigMixinClass:
    """Class that doesn't inherit from ConfigMixin."""

    @register_to_config
    def __init__(self, param: int = 1):
        self.param = param


class TestDecoratorRegistration:
    """Test that the decorator properly registers arguments."""

    def test_basic_registration(self):
        """Test basic automatic registration."""
        config = SampleDecoratorBasic(param1=42, param2="test")

        assert hasattr(config, "_internal_dict")
        assert isinstance(config._internal_dict, FrozenDict)
        assert config.config["param1"] == 42
        assert config.config["param2"] == "test"
        assert "_use_default_values" in config.config

    def test_default_values_tracking(self):
        """Test that default values are properly tracked."""
        # Use all defaults
        config = SampleDecoratorBasic()

        assert config.config["param1"] == 10
        assert config.config["param2"] == "default"
        assert sorted(config.config["_use_default_values"]) == ["param1", "param2"]

        # Override one parameter
        config2 = SampleDecoratorBasic(param1=99)

        assert config2.config["param1"] == 99
        assert config2.config["param2"] == "default"
        assert config2.config["_use_default_values"] == ["param2"]

        # Override all parameters
        config3 = SampleDecoratorBasic(param1=123, param2="custom")

        assert config3.config["param1"] == 123
        assert config3.config["param2"] == "custom"
        assert config3.config["_use_default_values"] == []

    def test_positional_arguments(self):
        """Test that positional arguments are handled correctly."""
        config = SampleDecoratorBasic(55, "positional")

        assert config.config["param1"] == 55
        assert config.config["param2"] == "positional"
        assert config.config["_use_default_values"] == []

    def test_mixed_positional_and_keyword(self):
        """Test mixing positional and keyword arguments."""
        config = SampleDecoratorBasic(77, param2="mixed")

        assert config.config["param1"] == 77
        assert config.config["param2"] == "mixed"
        assert config.config["_use_default_values"] == []

    def test_partial_positional_arguments(self):
        """Test partial positional arguments with some defaults."""
        config = SampleDecoratorBasic(88)  # Only first parameter provided positionally

        assert config.config["param1"] == 88
        assert config.config["param2"] == "default"
        assert config.config["_use_default_values"] == ["param2"]


class TestDecoratorIgnoreParams:
    """Test that ignored parameters are not registered."""

    def test_ignore_for_config_respected(self):
        """Test that parameters in ignore_for_config are not registered."""
        config = SampleDecoratorWithIgnored(
            tracked_param=100,
            ignored_param="should_not_be_saved",
            also_ignored=False,
            another_tracked=3.14
        )

        assert config.config["tracked_param"] == 100
        assert config.config["another_tracked"] == 3.14
        assert "ignored_param" not in config.config
        assert "also_ignored" not in config.config

        # But the attributes should still be set on the instance
        assert config.ignored_param == "should_not_be_saved"
        assert config.also_ignored is False

    def test_ignored_params_in_default_tracking(self):
        """Test that ignored parameters are not included in default value tracking."""
        config = SampleDecoratorWithIgnored()  # All defaults

        # Only tracked parameters should be in _use_default_values
        expected_defaults = ["tracked_param", "another_tracked"]
        assert set(config.config["_use_default_values"]) == set(expected_defaults)
        assert "ignored_param" not in config.config["_use_default_values"]
        assert "also_ignored" not in config.config["_use_default_values"]


class TestDecoratorPrivateParams:
    """Test that private parameters (starting with underscore) are ignored."""

    def test_private_params_ignored(self):
        """Test that parameters starting with underscore are automatically ignored."""
        config = SampleDecoratorWithPrivate(
            public_param=200,
            _private_param="should_not_be_saved",
            normal_param=5.5
        )

        assert config.config["public_param"] == 200
        assert config.config["normal_param"] == 5.5
        assert "_private_param" not in config.config

        # But the attributes should still be set on the instance
        assert config._private_param == "should_not_be_saved"

    def test_private_params_not_in_defaults_tracking(self):
        """Test that private parameters are not tracked in default values."""
        config = SampleDecoratorWithPrivate()  # All defaults

        expected_defaults = ["public_param", "normal_param"]
        assert set(config.config["_use_default_values"]) == set(expected_defaults)
        assert "_private_param" not in config.config["_use_default_values"]


class SampleDecoratorComplexTypesHandling:
    """Test decorator with complex data types."""

    def test_complex_types_registration(self):
        """Test that complex types are properly registered."""
        config = SampleDecoratorComplexTypes(
            list_param=[4, 5, 6],
            dict_param={"custom": "data"},
            optional_param="not_none",
            default_list=[7, 8, 9]
        )

        assert config.config["list_param"] == [4, 5, 6]
        assert config.config["dict_param"] == {"custom": "data"}
        assert config.config["optional_param"] == "not_none"
        assert config.config["default_list"] == [7, 8, 9]

    def test_none_values_handling(self):
        """Test handling of None values and default processing."""
        config = SampleDecoratorComplexTypes(optional_param="set_value")

        # None values should be registered as None
        assert config.config["list_param"] is None
        assert config.config["dict_param"] is None
        assert config.config["optional_param"] == "set_value"
        assert config.config["default_list"] is None

        # But instance should have processed defaults
        assert config.list_param == [1, 2, 3]
        assert config.dict_param == {"key": "value"}
        assert config.optional_param == "set_value"
        assert config.default_list == []


class TestDecoratorRequiredParams:
    """Test decorator with required parameters."""

    def test_required_params_registration(self):
        """Test that required parameters are properly registered."""
        config = SampleDecoratorNoDefaults(
            required_param=999,
            required_str="required_value",
            optional_param=False
        )

        assert config.config["required_param"] == 999
        assert config.config["required_str"] == "required_value"
        assert config.config["optional_param"] is False
        assert config.config["_use_default_values"] == []

    def test_required_params_with_defaults(self):
        """Test required params with optional param using default."""
        config = SampleDecoratorNoDefaults(
            required_param=777,
            required_str="another_value"
        )

        assert config.config["required_param"] == 777
        assert config.config["required_str"] == "another_value"
        assert config.config["optional_param"] is True
        assert config.config["_use_default_values"] == ["optional_param"]


class TestDecoratorErrorHandling:
    """Test decorator error handling."""

    def test_non_config_mixin_class_raises_error(self):
        """Test that decorator raises error when used on non-ConfigMixin class."""
        with pytest.raises(RuntimeError) as exc_info:
            NotConfigMixinClass(param=42)

        assert "`@register_to_config` was applied to NotConfigMixinClass init method" in str(exc_info.value)
        assert "does not inherit from `ConfigMixin`" in str(exc_info.value)

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        # Check that the decorated function has proper metadata
        init_func = SampleDecoratorBasic.__init__

        # Should preserve function name and signature
        assert hasattr(init_func, "__name__")
        assert hasattr(init_func, "__doc__")

        # Should be able to inspect signature
        import inspect
        sig = inspect.signature(init_func)
        assert "param1" in sig.parameters
        assert "param2" in sig.parameters


class TestDecoratorIntegration:
    """Test decorator integration with other ConfigMixin features."""

    def test_decorator_with_save_load(self):
        """Test that decorator works with save/load functionality."""
        import tempfile

        original = SampleDecoratorBasic(param1=333, param2="integration_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save config
            original.save_config(temp_dir)

            # Load back
            loaded, unused = SampleDecoratorBasic.from_config(temp_dir)

            assert loaded.param1 == 333
            assert loaded.param2 == "integration_test"
            assert loaded.config["param1"] == 333
            assert loaded.config["param2"] == "integration_test"
            assert unused == {}

    def test_decorator_with_attribute_access(self):
        """Test that decorator works with attribute access shortcuts."""
        config = SampleDecoratorBasic(param1=444, param2="attr_test")

        # Should be able to access config values as attributes
        assert config.param1 == 444
        assert config.param2 == "attr_test"

        # Config should be frozen
        with pytest.raises(Exception):
            config.config["param1"] = 999

    def test_decorator_repr(self):
        """Test string representation of decorated class."""
        config = SampleDecoratorBasic(param1=555, param2="repr_test")
        repr_str = repr(config)

        assert "SampleDecoratorBasic" in repr_str
        assert "555" in repr_str
        assert "repr_test" in repr_str

    def test_decorator_json_serialization(self):
        """Test JSON serialization of decorated class."""
        import json

        config = SampleDecoratorComplexTypes(
            list_param=[10, 11, 12],
            dict_param={"serialization": "test"}
        )

        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        assert config_dict["_class_name"] == "SampleDecoratorComplexTypes"
        assert config_dict["list_param"] == [10, 11, 12]
        assert config_dict["dict_param"] == {"serialization": "test"}
        assert "optional_param" in config_dict  # Should be None
        assert "default_list" in config_dict   # Should be None
