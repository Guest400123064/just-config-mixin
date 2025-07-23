#!/usr/bin/env python3

"""
Test suite for ConfigMixin functionality.
"""

import json
import os
import pathlib
import tempfile
import pytest

from yacm import ConfigMixin, FrozenDict


class SampleConfigMixin(ConfigMixin):
    """Test class that inherits from ConfigMixin."""

    config_name = "test_config.json"
    ignore_for_config = ["ignored_param"]

    def __init__(self, param1: int, param2: str = "default", ignored_param: bool = True):
        self.param1 = param1
        self.param2 = param2
        self.ignored_param = ignored_param

        # Manually register config (normally done by decorator)
        self.register_to_config(param1=param1, param2=param2)


class SampleConfigMixinNoConfigName(ConfigMixin):
    """Test class without config_name defined."""

    def __init__(self, param: int = 1):
        self.param = param


class SampleConfigMixinWithComplexTypes(ConfigMixin):
    """Test class with complex data types."""

    config_name = "complex_config.json"

    def __init__(self, path_param: pathlib.Path = None, list_param: list = None, dict_param: dict = None):
        self.path_param = path_param or pathlib.Path("/tmp/test")
        self.list_param = list_param or [1, 2, 3]
        self.dict_param = dict_param or {"nested": "value"}

        self.register_to_config(
            path_param=self.path_param,
            list_param=self.list_param,
            dict_param=self.dict_param
        )


class TestConfigRegistration:
    """Test configuration registration functionality."""

    def test_basic_registration(self):
        """Test basic config registration."""
        config = SampleConfigMixin(param1=42, param2="test")

        assert hasattr(config, "_internal_dict")
        assert isinstance(config._internal_dict, FrozenDict)
        assert config._internal_dict["param1"] == 42
        assert config._internal_dict["param2"] == "test"
        assert "ignored_param" not in config._internal_dict

    def test_config_property_access(self):
        """Test accessing config through property."""
        config = SampleConfigMixin(param1=100, param2="hello")

        assert config.config["param1"] == 100
        assert config.config["param2"] == "hello"
        assert isinstance(config.config, FrozenDict)

    def test_config_attribute_shortcut(self):
        """Test accessing config attributes directly on instance."""
        config = SampleConfigMixin(param1=999, param2="world")

        # Should be able to access config values as attributes
        assert config.param1 == 999
        assert config.param2 == "world"

        # Non-config attributes should also work
        assert config.ignored_param is True

    def test_double_registration_raises_error(self):
        """Test that calling register_to_config twice raises an error."""
        config = SampleConfigMixin(param1=1, param2="test")

        with pytest.raises(RuntimeError) as exc_info:
            config.register_to_config(param1=2)

        assert "_internal_dict` is already set" in str(exc_info.value)
        assert "prevent unexpected inconsistencies" in str(exc_info.value)

    def test_no_config_name_raises_error(self):
        """Test that missing config_name raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            SampleConfigMixinNoConfigName(param=1).register_to_config(param=1)

        assert "defined a class attribute `config_name`" in str(exc_info.value)


class TestConfigStringRepresentation:
    """Test string representation and JSON serialization."""

    def test_repr(self):
        """Test __repr__ method."""
        config = SampleConfigMixin(param1=123, param2="repr_test")
        repr_str = repr(config)

        assert "SampleConfigMixin" in repr_str
        assert "123" in repr_str
        assert "repr_test" in repr_str

    def test_get_config_json_basic(self):
        """Test JSON serialization of config."""
        config = SampleConfigMixin(param1=456, param2="json_test")
        json_str = config.get_config_json()

        # Parse the JSON to verify structure
        config_dict = json.loads(json_str)

        assert config_dict["_class_name"] == "SampleConfigMixin"
        assert config_dict["param1"] == 456
        assert config_dict["param2"] == "json_test"
        assert "ignored_param" not in config_dict

    def test_get_config_json_complex_types(self):
        """Test JSON serialization with complex types."""
        complex_config = SampleConfigMixinWithComplexTypes(
            path_param=pathlib.Path("/custom/path"),
            list_param=[4, 5, 6],
            dict_param={"custom": "data"}
        )
        json_str = complex_config.get_config_json()

        config_dict = json.loads(json_str)

        assert config_dict["_class_name"] == "SampleConfigMixinWithComplexTypes"
        assert config_dict["path_param"] == "/custom/path"  # Path converted to string
        assert config_dict["list_param"] == [4, 5, 6]
        assert config_dict["dict_param"] == {"custom": "data"}

    def test_json_sorted_and_indented(self):
        """Test that JSON output is sorted and indented."""
        config = SampleConfigMixin(param1=1, param2="b")
        json_str = config.get_config_json()

        # Should be indented (multiple lines)
        assert "\n" in json_str

        # Should be sorted (parse and check order)
        lines = [line.strip() for line in json_str.split('\n') if line.strip() and not line.strip().startswith('{')]
        # Remove lines that are just braces
        field_lines = [line for line in lines if ':' in line and not line.endswith(('{', '}'))]
        # Should have _class_name first (alphabetically)
        assert '"_class_name"' in field_lines[0]


class TestConfigSaveLoad:
    """Test saving and loading configurations."""

    def test_save_config_basic(self):
        """Test basic config saving."""
        config = SampleConfigMixin(param1=789, param2="save_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_config(temp_dir)

            # Check that file was created
            config_file = pathlib.Path(temp_dir) / "test_config.json"
            assert config_file.exists()

            # Verify file contents
            with open(config_file) as f:
                saved_data = json.load(f)

            assert saved_data["_class_name"] == "SampleConfigMixin"
            assert saved_data["param1"] == 789
            assert saved_data["param2"] == "save_test"

    def test_save_config_file_exists_no_overwrite(self):
        """Test that saving fails when file exists and overwrite=False."""
        config = SampleConfigMixin(param1=1, param2="test")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the file first
            config.save_config(temp_dir)

            # Try to save again without overwrite
            with pytest.raises(FileExistsError) as exc_info:
                config.save_config(temp_dir, overwrite=False)

            assert "already contains a file named test_config.json" in str(exc_info.value)
            assert "set `overwrite=True`" in str(exc_info.value)

    def test_save_config_overwrite(self):
        """Test saving with overwrite=True."""
        config1 = SampleConfigMixin(param1=1, param2="first")
        config2 = SampleConfigMixin(param1=2, param2="second")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save first config
            config1.save_config(temp_dir)

            # Overwrite with second config
            config2.save_config(temp_dir, overwrite=True)

            # Verify second config was saved
            config_file = pathlib.Path(temp_dir) / "test_config.json"
            with open(config_file) as f:
                saved_data = json.load(f)

            assert saved_data["param1"] == 2
            assert saved_data["param2"] == "second"

    def test_save_config_creates_directory(self):
        """Test that save_config creates directories if they don't exist."""
        config = SampleConfigMixin(param1=123, param2="mkdir_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = pathlib.Path(temp_dir) / "nested" / "path"
            config.save_config(nested_dir)

            assert nested_dir.exists()
            assert (nested_dir / "test_config.json").exists()

    def test_save_config_invalid_path_is_file(self):
        """Test that save_config raises error if path is a file."""
        config = SampleConfigMixin(param1=1, param2="test")

        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(AssertionError) as exc_info:
                config.save_config(temp_file.name)

            assert "should be a directory, not a file" in str(exc_info.value)

    def test_save_config_no_config_name(self):
        """Test that save_config raises error when config_name is None."""
        config = SampleConfigMixinNoConfigName(param=1)

        with pytest.raises(NotImplementedError) as exc_info:
            config.save_config("/tmp")

        assert "defined a class attribute `config_name`" in str(exc_info.value)


class TestConfigFromConfig:
    """Test instantiating classes from config dictionaries."""

    def test_from_config_basic(self):
        """Test basic from_config functionality."""
        # First save a config
        original = SampleConfigMixin(param1=555, param2="from_config_test")

        with tempfile.TemporaryDirectory() as temp_dir:
            original.save_config(temp_dir)

            # Load it back
            loaded, unused = SampleConfigMixin.from_config(temp_dir)

            assert isinstance(loaded, SampleConfigMixin)
            assert loaded.param1 == 555
            assert loaded.param2 == "from_config_test"
            assert loaded.config["param1"] == 555
            assert loaded.config["param2"] == "from_config_test"
            assert unused == {}

    def test_from_config_missing_file(self):
        """Test from_config with missing config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError) as exc_info:
                SampleConfigMixin.from_config(temp_dir)

            assert "does not contain a file named test_config.json" in str(exc_info.value)

    def test_from_config_invalid_path_is_file(self):
        """Test from_config with file path instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(AssertionError) as exc_info:
                SampleConfigMixin.from_config(temp_file.name)

            assert "should be a directory, not a file" in str(exc_info.value)

    def test_from_config_wrong_class_name(self):
        """Test from_config with wrong class name in config."""
        config = SampleConfigMixin(param1=1, param2="test")

        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_config(temp_dir)

            # Manually modify the class name in the saved file
            config_file = pathlib.Path(temp_dir) / "test_config.json"
            with open(config_file) as f:
                data = json.load(f)

            data["_class_name"] = "WrongClassName"

            with open(config_file, 'w') as f:
                json.dump(data, f)

            with pytest.raises(ValueError) as exc_info:
                SampleConfigMixin.from_config(temp_dir)

            assert "is not a config for SampleConfigMixin" in str(exc_info.value)

    def test_from_config_missing_required_param(self):
        """Test from_config with missing required parameter."""
        # Create a config manually without required parameter
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = pathlib.Path(temp_dir) / "test_config.json"
            incomplete_config = {
                "_class_name": "SampleConfigMixin",
                "param2": "test"  # Missing param1
            }

            with open(config_file, 'w') as f:
                json.dump(incomplete_config, f)

            with pytest.raises(ValueError) as exc_info:
                SampleConfigMixin.from_config(temp_dir)

            assert "Config is missing required parameter(s): param1" in str(exc_info.value)

    def test_from_config_with_extra_parameters(self):
        """Test from_config with extra parameters not in __init__."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = pathlib.Path(temp_dir) / "test_config.json"
            config_with_extra = {
                "_class_name": "SampleConfigMixin",
                "param1": 42,
                "param2": "test",
                "extra_param": "should_be_unused"
            }

            with open(config_file, 'w') as f:
                json.dump(config_with_extra, f)

            loaded, unused = SampleConfigMixin.from_config(temp_dir)

            assert loaded.param1 == 42
            assert loaded.param2 == "test"
            assert unused == {"extra_param": "should_be_unused"}


class TestConfigAttributeAccess:
    """Test attribute access functionality."""

    def test_getattr_config_attribute(self):
        """Test __getattr__ returns config attributes."""
        config = SampleConfigMixin(param1=777, param2="getattr_test")

        # Should be able to access config attributes
        assert config.param1 == 777
        assert config.param2 == "getattr_test"

    def test_getattr_instance_attribute_priority(self):
        """Test that instance attributes take priority over config attributes."""
        config = SampleConfigMixin(param1=100, param2="test")

        # Set instance attribute that shadows config
        config.__dict__["param1"] = 999

        # Should return instance attribute, not config attribute
        assert config.param1 == 999
        assert config.config["param1"] == 100  # Config unchanged

    def test_getattr_nonexistent_attribute(self):
        """Test __getattr__ raises AttributeError for nonexistent attributes."""
        config = SampleConfigMixin(param1=1, param2="test")

        with pytest.raises(AttributeError) as exc_info:
            _ = config.nonexistent_attribute

        assert "`SampleConfigMixin` object has no attribute `nonexistent_attribute`" in str(exc_info.value)

    def test_getattr_no_config(self):
        """Test __getattr__ when no config is registered."""
        class NoConfigClass(ConfigMixin):
            pass

        obj = NoConfigClass()

        with pytest.raises(AttributeError) as exc_info:
            _ = obj.some_attribute

        assert "`NoConfigClass` object has no attribute `some_attribute`" in str(exc_info.value)
