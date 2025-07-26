#!/usr/bin/env python3

r"""Test suite for edge cases and error handling in ConfigMixin.

This module tests various edge cases, error conditions, and boundary scenarios:
- Malformed configuration data
- Circular references and complex serialization
- Error conditions and exception handling
- Boundary cases for var args and metadata
- Integration of error handling across features
"""

import json
import pathlib
import tempfile
from typing import Any

import pytest

from configmixin import ConfigMixin, register_to_config

from .conftest import (
    BaseConfig,
    ConfigWithoutName,
    MockNonSerializableObject,
    MockSerializableObject,
    NonConfigMixinClass,
    create_config_dict,
)


class EdgeCaseConfig(ConfigMixin):
    r"""Configuration class for testing edge cases."""

    config_name = "edge_case.json"
    ignore_for_config = ["ignored_var_kwarg"]

    @register_to_config
    def __init__(
        self,
        normal_param: int = 10,
        *args,
        keyword_only: str = "kw_only",
        ignored_var_kwarg: str = "ignored",
        **kwargs,
    ):
        self.normal_param = normal_param
        self.args = args
        self.keyword_only = keyword_only
        self.ignored_var_kwarg = ignored_var_kwarg
        self.kwargs = kwargs


class ComplexTypeConfig(ConfigMixin):
    r"""Configuration class with complex types for edge case testing."""

    config_name = "complex_types.json"

    @register_to_config
    def __init__(self, base: str = "test", *args, **kwargs):
        self.base = base
        self.args = args
        self.kwargs = kwargs


class TestConfigMixinErrorHandling:
    r"""Test fundamental error handling in ConfigMixin."""

    def test_register_to_config_called_twice(self):
        r"""Test error when register_to_config is called multiple times."""

        class DoubleRegisterConfig(ConfigMixin):
            config_name = "double_register.json"

            def __init__(self, param: int = 1):
                self.register_to_config(param=param)
                # This should raise an error
                self.register_to_config(param=param)

        with pytest.raises(RuntimeError) as exc_info:
            DoubleRegisterConfig(param=5)

        assert "_internal_dict` is already set" in str(exc_info.value)

    def test_config_access_before_registration(self):
        r"""Test accessing config before register_to_config is called."""

        class UnregisteredConfig(ConfigMixin):
            config_name = "unregistered.json"

            def __init__(self, param: int = 1):
                # Don't call register_to_config
                self.param = param

        instance = UnregisteredConfig(param=5)

        # Should raise AttributeError when accessing config
        with pytest.raises(AttributeError):
            _ = instance.config

    def test_decorator_without_configmixin_inheritance(self):
        r"""Test error when decorator is used without ConfigMixin inheritance."""
        with pytest.raises(RuntimeError) as exc_info:
            NonConfigMixinClass(param=5)

        assert "does not inherit from `ConfigMixin`" in str(exc_info.value)


class TestFileSystemErrorHandling:
    r"""Test file system related error handling."""

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

    def test_from_config_file_not_found(self):
        r"""Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            BaseConfig.from_config(save_directory="/nonexistent/directory")

        assert "does not contain a file named" in str(exc_info.value)

    def test_save_config_overwrite_protection(self, temp_directory):
        r"""Test that save_config protects against overwriting by default."""
        config = BaseConfig(param1=100)

        # Save first time
        config.save_config(temp_directory)

        # Try to save again without overwrite flag
        with pytest.raises(FileExistsError) as exc_info:
            config.save_config(temp_directory)

        assert "overwrite=True" in str(exc_info.value)


class TestFromConfigErrorHandling:
    r"""Test error handling in from_config method."""

    def test_missing_both_directory_and_config(self):
        r"""Test error when neither save_directory nor config is provided."""
        with pytest.raises(ValueError) as exc_info:
            BaseConfig.from_config()

        assert "Either `save_directory` or `config` must be provided" in str(
            exc_info.value
        )

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

        config_dict = create_config_dict("RequiredParamConfig", optional_param=20)

        with pytest.raises(KeyError) as exc_info:
            RequiredParamConfig.from_config(config=config_dict)

        assert "missing required" in str(exc_info.value)

    def test_unexpected_param_in_config(self):
        r"""Test error when config contains unexpected parameters."""
        config_dict = create_config_dict(
            "BaseConfig",
            param1=42,
            param2="test",
            param3=[1, 2, 3],
            unexpected_param="should_cause_error",
        )

        with pytest.raises(TypeError) as exc_info:
            BaseConfig.from_config(config=config_dict)

        assert "unexpected" in str(exc_info.value)

    def test_from_config_malformed_var_args(self):
        r"""Test from_config with malformed var args data."""
        config_dict = create_config_dict(
            "ComplexTypeConfig",
            base="malformed",
            _var_positional="should_be_list",  # Wrong type
            _var_keyword=["should", "be", "dict"],  # Wrong type
        )

        # Should handle gracefully or raise appropriate error
        with pytest.raises((TypeError, ValueError, AttributeError)):
            ComplexTypeConfig.from_config(config=config_dict)


class TestVarArgsEdgeCases:
    r"""Test edge cases specific to var args functionality."""

    def test_ignored_var_kwargs_not_tracked(self):
        r"""Test that ignored var kwargs are not tracked in _var_keyword."""
        config = EdgeCaseConfig(
            20,
            "arg1",
            "arg2",
            keyword_only="custom",
            ignored_var_kwarg="should_not_appear",
            tracked_kwarg="should_appear",
        )

        assert config.config["normal_param"] == 20
        assert config.config["keyword_only"] == "custom"
        assert config.config["_var_positional"] == ("arg1", "arg2")
        assert config.config["_var_keyword"] == {"tracked_kwarg": "should_appear"}
        assert "ignored_var_kwarg" not in config.config["_var_keyword"]

        # But the attribute should still be set
        assert config.ignored_var_kwarg == "should_not_appear"

    def test_var_args_missing_metadata_compatibility(self):
        r"""Test from_config when var args metadata is missing (backward compatibility)."""
        # Config without var args metadata (old format)
        config_dict = create_config_dict(
            "EdgeCaseConfig", normal_param=50, keyword_only="old_format"
        )

        # Remove var args metadata to simulate old config
        config_dict.pop("_var_positional", None)
        config_dict.pop("_var_keyword", None)

        instance = EdgeCaseConfig.from_config(config=config_dict)

        # Should work with empty var args
        assert instance.normal_param == 50
        assert instance.keyword_only == "old_format"
        assert instance.args == ()
        assert instance.kwargs == {}

    def test_var_args_with_none_values(self):
        r"""Test var args containing None values."""
        config = EdgeCaseConfig(
            100, None, "arg2", keyword_only="none_test", none_kw=None
        )

        assert config.config["_var_positional"] == (None, "arg2")
        assert config.config["_var_keyword"] == {"none_kw": None}


class TestFrozenDictErrorHandling:
    r"""Test FrozenDict behavior and error handling."""

    def test_frozen_dict_immutability_errors(self):
        r"""Test that FrozenDict raises appropriate errors when modified."""
        config = BaseConfig(param1=30)

        # Should not be able to modify config
        with pytest.raises(Exception):
            config.config["param1"] = 999

        with pytest.raises(Exception):
            config.config.param1 = 999

        with pytest.raises(Exception):
            del config.config["param1"]

        with pytest.raises(Exception):
            config.config.setdefault("new_key", "value")

        with pytest.raises(Exception):
            config.config.pop("param1")

        with pytest.raises(Exception):
            config.config.update({"new_key": "value"})

    def test_frozen_dict_with_var_args_immutability(self):
        r"""Test that FrozenDict remains immutable with var args data."""
        config = EdgeCaseConfig(
            30, "arg1", "arg2", keyword_only="frozen_test", extra_kwarg="extra_value"
        )

        # Should not be able to modify var args in config
        with pytest.raises(Exception):
            config.config["_var_positional"] = ("modified",)

        with pytest.raises(Exception):
            config.config["_var_keyword"]["new_key"] = "new_value"

        with pytest.raises(Exception):
            config.config._var_positional = ("modified",)


class TestAttributeAccessEdgeCases:
    r"""Test edge cases in attribute access functionality."""

    def test_attribute_shortcut_with_var_args(self):
        r"""Test that __getattr__ works correctly with var args."""
        config = EdgeCaseConfig(
            50,
            "shortcut_arg",
            keyword_only="shortcut_test",
            shortcut_kwarg="shortcut_value",
        )

        # Should be able to access config values as attributes
        assert config.normal_param == 50
        assert config.keyword_only == "shortcut_test"
        assert config._var_positional == ("shortcut_arg",)
        assert config._var_keyword == {"shortcut_kwarg": "shortcut_value"}

    def test_attribute_shortcut_precedence(self):
        r"""Test that instance attributes take precedence over config shortcuts."""
        config = EdgeCaseConfig(60, "precedence_arg", keyword_only="precedence_test")

        # Instance attribute should take precedence
        assert config.normal_param == 60  # Instance attribute
        assert config.keyword_only == "precedence_test"  # Instance attribute

        # But config shortcut should work for config-only values
        assert config._class_name == "EdgeCaseConfig"  # From config

    def test_attribute_error_for_nonexistent(self):
        r"""Test AttributeError for non-existent attributes."""
        config = EdgeCaseConfig(keyword_only="error_test")

        with pytest.raises(AttributeError) as exc_info:
            _ = config.nonexistent_attribute

        assert "has no attribute `nonexistent_attribute`" in str(exc_info.value)


class TestJSONSerializationEdgeCases:
    r"""Test JSON serialization edge cases and error conditions."""

    def test_json_with_pathlib_objects(self):
        r"""Test JSON serialization with pathlib objects."""
        path1 = pathlib.Path("/data/input")
        path2 = pathlib.Path("/data/output")

        config = ComplexTypeConfig(
            "pathlib_test", path1, path2, output_path=pathlib.Path("/results")
        )

        # Paths should be serialized as POSIX strings
        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        assert config_dict["_var_positional"] == ["/data/input", "/data/output"]
        assert config_dict["_var_keyword"]["output_path"] == "/results"

    def test_json_with_serializable_objects(self):
        r"""Test JSON serialization with objects that have to_dict method."""
        obj1 = MockSerializableObject("test1")
        obj2 = MockSerializableObject("test2")

        config = ComplexTypeConfig("serializable", obj1, obj_kwarg=obj2)

        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        # Objects with to_dict should be serialized using that method
        expected_obj1 = {"mock_value": "test1", "type": "MockSerializableObject"}
        expected_obj2 = {"mock_value": "test2", "type": "MockSerializableObject"}

        assert config_dict["_var_positional"][0] == expected_obj1
        assert config_dict["_var_keyword"]["obj_kwarg"] == expected_obj2

    def test_json_with_nested_lists(self):
        r"""Test JSON serialization with deeply nested lists."""
        nested_list = [[1, [2, [3, 4]]], [5, 6]]
        config = ComplexTypeConfig("nested", nested_list)

        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        assert config_dict["_var_positional"][0] == nested_list

    def test_json_serialization_with_empty_var_args(self):
        r"""Test JSON serialization when var args are empty."""
        config = ComplexTypeConfig(base="empty")
        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        assert config_dict["_var_positional"] == []
        assert config_dict["_var_keyword"] == {}

    def test_json_serialization_error_handling(self):
        r"""Test JSON serialization with objects that can't be serialized."""
        # Objects without to_dict and that aren't basic types might cause issues
        non_serializable = MockNonSerializableObject("problematic")

        config = ComplexTypeConfig("error_prone", non_serializable)

        # This might succeed (storing the object as-is) or fail gracefully
        # The behavior depends on the cast function implementation
        try:
            json_str = config.get_config_json()
            # If it succeeds, parsing should work
            config_dict = json.loads(json_str)
            assert "_var_positional" in config_dict
        except (TypeError, ValueError, AttributeError):
            # It's acceptable if non-serializable objects can't be serialized
            pass


class TestComplexScenarios:
    r"""Test complex edge case scenarios combining multiple features."""

    def test_complex_inheritance_with_edge_cases(self):
        r"""Test edge cases in complex inheritance scenarios."""

        class ComplexBase:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.complex_base = True

        class ComplexEdgeConfig(ComplexBase, ConfigMixin):
            config_name = "complex_edge.json"
            ignore_for_config = ["runtime_param"]

            @register_to_config
            def __init__(
                self, param: str = "default", *args, runtime_param: Any = None, **kwargs
            ):
                super().__init__()
                self.param = param
                self.args = args
                self.runtime_param = runtime_param
                self.kwargs = kwargs

        config = ComplexEdgeConfig(
            "test",
            "arg1",
            None,
            "arg3",
            runtime_param="should_be_ignored",
            kw1=None,
            kw2={"nested": None},
        )

        # Should handle all edge cases correctly
        assert config.complex_base is True
        assert config.param == "test"
        assert config.args == ("arg1", None, "arg3")
        assert config.kwargs == {"kw1": None, "kw2": {"nested": None}}
        assert config.runtime_param == "should_be_ignored"
        assert "runtime_param" not in config.config

    def test_edge_cases_with_from_config_dict(self):
        r"""Test edge cases when loading from config dict."""
        config_dict = create_config_dict(
            "ComplexTypeConfig",
            base="edge_dict",
            _var_positional=[None, {"complex": None}, [1, None, 3]],
            _var_keyword={
                "none_value": None,
                "empty_list": [],
                "empty_dict": {},
                "nested_none": {"inner": None},
            },
        )

        instance = ComplexTypeConfig.from_config(config=config_dict)

        assert instance.base == "edge_dict"
        assert instance.args == (None, {"complex": None}, [1, None, 3])
        assert instance.kwargs == {
            "none_value": None,
            "empty_list": [],
            "empty_dict": {},
            "nested_none": {"inner": None},
        }

    def test_edge_cases_with_runtime_kwargs_and_var_args(self):
        r"""Test edge cases combining runtime kwargs with var args."""
        config_dict = create_config_dict(
            "EdgeCaseConfig",
            normal_param=99,
            keyword_only="edge_runtime",
            _var_positional=["runtime_arg"],
            _var_keyword={"config_kw": "config_value"},
        )

        runtime_kwargs = {"ignored_var_kwarg": "runtime_ignored"}

        instance = EdgeCaseConfig.from_config(
            config=config_dict, runtime_kwargs=runtime_kwargs
        )

        # Everything should work correctly
        assert instance.normal_param == 99
        assert instance.keyword_only == "edge_runtime"
        assert instance.args == ("runtime_arg",)
        assert instance.kwargs == {"config_kw": "config_value"}
        assert instance.ignored_var_kwarg == "runtime_ignored"


class TestBoundaryConditions:
    r"""Test boundary conditions and limits."""

    def test_empty_config_class(self):
        r"""Test ConfigMixin with minimal configuration."""

        class MinimalConfig(ConfigMixin):
            config_name = "minimal.json"

            @register_to_config
            def __init__(self):
                pass

        config = MinimalConfig()

        # Should work with no parameters
        assert config.config["_class_name"] == "MinimalConfig"
        assert config.config["_use_default_values"] == []
        assert config.config["_var_positional"] == ()
        assert config.config["_var_keyword"] == {}

    def test_large_number_of_parameters(self):
        r"""Test ConfigMixin with many parameters."""

        class ManyParamsConfig(ConfigMixin):
            config_name = "many_params.json"

            @register_to_config
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        # Create config with many parameters
        many_params = {f"param_{i}": i for i in range(100)}
        config = ManyParamsConfig(**many_params)

        # Should handle many parameters correctly
        assert len(config.config["_var_keyword"]) == 100
        for i in range(100):
            assert config.config["_var_keyword"][f"param_{i}"] == i

    def test_deeply_nested_config_structures(self):
        r"""Test with deeply nested configuration structures."""
        deep_structure = {
            "level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}
        }

        config = ComplexTypeConfig("deep", deep_structure)

        # Should preserve deep nesting
        json_str = config.get_config_json()
        config_dict = json.loads(json_str)

        assert config_dict["_var_positional"][0] == deep_structure
