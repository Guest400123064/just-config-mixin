#!/usr/bin/env python3

"""
Test suite for FrozenDict functionality.
"""

import pytest
from collections import OrderedDict

from yacm import FrozenDict


class TestFrozenDictInitialization:
    """Test FrozenDict initialization scenarios."""

    def test_empty_initialization(self):
        """Test creating an empty FrozenDict."""
        frozen = FrozenDict()
        assert len(frozen) == 0
        assert isinstance(frozen, OrderedDict)

    def test_dict_initialization(self):
        """Test creating FrozenDict from dict."""
        data = {"key1": "value1", "key2": "value2"}
        frozen = FrozenDict(data)

        assert len(frozen) == 2
        assert frozen["key1"] == "value1"
        assert frozen["key2"] == "value2"

    def test_kwargs_initialization(self):
        """Test creating FrozenDict from keyword arguments."""
        frozen = FrozenDict(name="test", value=42, flag=True)

        assert frozen["name"] == "test"
        assert frozen["value"] == 42
        assert frozen["flag"] is True

    def test_mixed_initialization(self):
        """Test creating FrozenDict with both dict and kwargs."""
        data = {"existing": "value"}
        frozen = FrozenDict(data, new_key="new_value")

        assert frozen["existing"] == "value"
        assert frozen["new_key"] == "new_value"

    def test_attribute_access_after_init(self):
        """Test that attributes are set after initialization."""
        frozen = FrozenDict(name="test", count=5)

        assert hasattr(frozen, "name")
        assert hasattr(frozen, "count")
        assert frozen.name == "test"
        assert frozen.count == 5


class TestFrozenDictImmutability:
    """Test FrozenDict immutability after initialization."""

    def test_setitem_raises_exception(self):
        """Test that __setitem__ raises exception after frozen."""
        frozen = FrozenDict({"key": "value"})

        with pytest.raises(Exception) as exc_info:
            frozen["new_key"] = "new_value"

        assert "You cannot use `__setitem__`" in str(exc_info.value)
        assert "FrozenDict" in str(exc_info.value)

    def test_setattr_raises_exception(self):
        """Test that __setattr__ raises exception after frozen."""
        frozen = FrozenDict({"key": "value"})

        with pytest.raises(Exception) as exc_info:
            frozen.new_attr = "new_value"

        assert "You cannot use `__setattr__`" in str(exc_info.value)
        assert "FrozenDict" in str(exc_info.value)

    def test_delitem_raises_exception(self):
        """Test that __delitem__ raises exception."""
        frozen = FrozenDict({"key": "value"})

        with pytest.raises(Exception) as exc_info:
            del frozen["key"]

        assert "You cannot use `__delitem__`" in str(exc_info.value)
        assert "FrozenDict" in str(exc_info.value)

    def test_setdefault_raises_exception(self):
        """Test that setdefault raises exception."""
        frozen = FrozenDict({"key": "value"})

        with pytest.raises(Exception) as exc_info:
            frozen.setdefault("new_key", "default_value")

        assert "You cannot use `setdefault`" in str(exc_info.value)
        assert "FrozenDict" in str(exc_info.value)

    def test_pop_raises_exception(self):
        """Test that pop raises exception."""
        frozen = FrozenDict({"key": "value"})

        with pytest.raises(Exception) as exc_info:
            frozen.pop("key")

        assert "You cannot use `pop`" in str(exc_info.value)
        assert "FrozenDict" in str(exc_info.value)

    def test_update_raises_exception(self):
        """Test that update raises exception."""
        frozen = FrozenDict({"key": "value"})

        with pytest.raises(Exception) as exc_info:
            frozen.update({"new_key": "new_value"})

        assert "You cannot use `update`" in str(exc_info.value)
        assert "FrozenDict" in str(exc_info.value)


class TestFrozenDictReadOperations:
    """Test FrozenDict read operations work normally."""

    def test_getitem(self):
        """Test item access via brackets."""
        frozen = FrozenDict({"key1": "value1", "key2": 42})

        assert frozen["key1"] == "value1"
        assert frozen["key2"] == 42

    def test_getattr(self):
        """Test attribute access."""
        frozen = FrozenDict({"name": "test", "count": 10})

        assert frozen.name == "test"
        assert frozen.count == 10

    def test_get_method(self):
        """Test get method works."""
        frozen = FrozenDict({"existing": "value"})

        assert frozen.get("existing") == "value"
        assert frozen.get("missing") is None
        assert frozen.get("missing", "default") == "default"

    def test_keys_values_items(self):
        """Test keys, values, and items methods."""
        data = {"a": 1, "b": 2, "c": 3}
        frozen = FrozenDict(data)

        assert list(frozen.keys()) == ["a", "b", "c"]
        assert list(frozen.values()) == [1, 2, 3]
        assert list(frozen.items()) == [("a", 1), ("b", 2), ("c", 3)]

    def test_len_and_bool(self):
        """Test len and boolean evaluation."""
        empty_frozen = FrozenDict()
        non_empty_frozen = FrozenDict({"key": "value"})

        assert len(empty_frozen) == 0
        assert len(non_empty_frozen) == 1
        assert not empty_frozen
        assert non_empty_frozen

    def test_contains(self):
        """Test 'in' operator."""
        frozen = FrozenDict({"key1": "value1", "key2": "value2"})

        assert "key1" in frozen
        assert "key2" in frozen
        assert "key3" not in frozen


class TestFrozenDictSpecialCases:
    """Test FrozenDict special cases and edge cases."""

    def test_nested_data_structures(self):
        """Test with nested dictionaries and lists."""
        data = {
            "config": {"param1": 1, "param2": 2},
            "items": [1, 2, 3],
            "metadata": {"version": "1.0", "author": "test"}
        }
        frozen = FrozenDict(data)

        assert frozen["config"]["param1"] == 1
        assert frozen["items"] == [1, 2, 3]
        assert frozen.metadata["version"] == "1.0"

        # Note: nested structures are still mutable
        frozen["items"].append(4)
        assert frozen["items"] == [1, 2, 3, 4]

    def test_different_value_types(self):
        """Test FrozenDict with various value types."""
        import pathlib

        data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "path": pathlib.Path("/tmp/test")
        }
        frozen = FrozenDict(data)

        assert frozen.string == "test"
        assert frozen.integer == 42
        assert frozen.float == 3.14
        assert frozen.boolean is True
        assert frozen.none is None
        assert frozen.list == [1, 2, 3]
        assert frozen.dict == {"nested": "value"}
        assert frozen.path == pathlib.Path("/tmp/test")

    def test_order_preservation(self):
        """Test that order is preserved (OrderedDict behavior)."""
        data = [("z", 1), ("a", 2), ("m", 3)]
        frozen = FrozenDict(data)

        assert list(frozen.keys()) == ["z", "a", "m"]
        assert list(frozen.values()) == [1, 2, 3]

    def test_repr_and_str(self):
        """Test string representation."""
        frozen = FrozenDict({"key": "value", "number": 42})

        # Should behave like OrderedDict for repr/str
        repr_str = repr(frozen)
        assert "FrozenDict" in repr_str
        assert "key" in repr_str
        assert "value" in repr_str
