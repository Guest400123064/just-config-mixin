#!/usr/bin/env python3

"""Tests for fully-qualified class names and recursive (de)serialization.

This module verifies two related features:

- ``__notes__.class_name`` now stores the fully-qualified name (module path + qualified
  name) instead of only the bare class name.
- ``ConfigMixin`` parameters are serialized recursively and automatically reconstructed
  on load, using the private ``__is_configmixin__`` marker.
"""

import json
from typing import Any, List

import pytest

from configmixin import ConfigMixin, register_to_config
from configmixin._core import (
    CONFIGMIXIN_MARKER,
    _qualified_name,
    _resolve_class,
)


class InnerConfig(ConfigMixin):
    """A leaf configuration used as a nested parameter."""

    config_name = "inner_config.json"

    @register_to_config
    def __init__(self, width: int = 8, activation: str = "relu"):
        self.width = width
        self.activation = activation


class OuterConfig(ConfigMixin):
    """A configuration that nests another ``ConfigMixin`` instance."""

    config_name = "outer_config.json"

    @register_to_config
    def __init__(self, name: str = "outer", inner: InnerConfig = None):
        self.name = name
        self.inner = inner if inner is not None else InnerConfig()


class GrandParentConfig(ConfigMixin):
    """Three-level deep nesting: grandparent -> outer -> inner."""

    config_name = "grandparent_config.json"

    @register_to_config
    def __init__(self, depth: int = 3, child: OuterConfig = None):
        self.depth = depth
        self.child = child if child is not None else OuterConfig()


class ConfigWithNestedContainers(ConfigMixin):
    """Configuration nesting ``ConfigMixin`` instances inside containers."""

    config_name = "nested_containers.json"

    @register_to_config
    def __init__(
        self,
        items: List[InnerConfig] = None,
        mapping: dict = None,
    ):
        self.items = items if items is not None else []
        self.mapping = mapping if mapping is not None else {}


class TestQualifiedClassName:
    """Test that the fully-qualified class name is stored in the config."""

    def test_class_name_includes_module(self):
        config = InnerConfig(width=16)
        expected = f"{InnerConfig.__module__}.{InnerConfig.__qualname__}"

        assert config.config["__notes__"]["class_name"] == expected
        assert "." in config.config["__notes__"]["class_name"]

    def test_qualified_name_helper(self):
        assert _qualified_name(InnerConfig) == f"{__name__}.InnerConfig"
        assert _qualified_name(InnerConfig).endswith("test_nested_config.InnerConfig")

    def test_resolve_class_roundtrip(self):
        qualified = _qualified_name(OuterConfig)
        assert _resolve_class(qualified) is OuterConfig

    def test_resolve_class_invalid(self):
        with pytest.raises(ImportError):
            _resolve_class("nonexistent.module.SomeClass")

    def test_from_config_rejects_wrong_qualified_name(self, temp_dir):
        config = InnerConfig()
        config.save_config(temp_dir)

        config_file = temp_dir / "inner_config.json"
        with open(config_file) as f:
            data = json.load(f)

        # Even a class with the same short name but different module must be rejected.
        data["__notes__"]["class_name"] = "some.other.module.InnerConfig"
        with open(config_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="not a config for InnerConfig"):
            InnerConfig.from_config(save_directory=temp_dir)


class TestNestedSerialization:
    """Test recursive serialization/deserialization of nested ConfigMixin."""

    def test_marker_present_in_serialized_output(self):
        inner = InnerConfig(width=32, activation="gelu")
        outer = OuterConfig(name="parent", inner=inner)

        data = json.loads(outer.config_dumps().decode())

        # The nested instance is serialized as a dict carrying the marker.
        assert data["inner"][CONFIGMIXIN_MARKER] is True
        assert data["inner"]["width"] == 32
        assert data["inner"]["activation"] == "gelu"
        # The top-level config is NOT marked.
        assert CONFIGMIXIN_MARKER not in data

    def test_nested_roundtrip_via_serialized_dict(self):
        inner = InnerConfig(width=64, activation="silu")
        outer = OuterConfig(name="parent", inner=inner)

        # Round-trip through JSON so the marker-based reconstruction path is exercised
        # (rather than simply deep-copying a live nested instance).
        serialized = json.loads(outer.config_dumps().decode())
        loaded = OuterConfig.from_config(config=serialized)

        assert loaded.name == "parent"
        assert isinstance(loaded.inner, InnerConfig)
        assert loaded.inner.width == 64
        assert loaded.inner.activation == "silu"

    def test_nested_roundtrip_via_file(self, temp_dir):
        inner = InnerConfig(width=128, activation="tanh")
        outer = OuterConfig(name="file_parent", inner=inner)

        outer.save_config(temp_dir)
        loaded = OuterConfig.from_config(save_directory=temp_dir)

        assert loaded.name == "file_parent"
        assert isinstance(loaded.inner, InnerConfig)
        assert loaded.inner.width == 128
        assert loaded.inner.activation == "tanh"

        # The reconstructed nested instance is a fully-functional ConfigMixin.
        assert loaded.inner.config["__notes__"]["class_name"] == _qualified_name(
            InnerConfig
        )

    def test_nested_with_defaults(self, temp_dir):
        outer = OuterConfig()  # inner defaults to InnerConfig()

        outer.save_config(temp_dir)
        loaded = OuterConfig.from_config(save_directory=temp_dir)

        assert isinstance(loaded.inner, InnerConfig)
        assert loaded.inner.width == 8
        assert loaded.inner.activation == "relu"

    def test_deeply_nested_roundtrip(self, temp_dir):
        inner = InnerConfig(width=256, activation="elu")
        outer = OuterConfig(name="mid", inner=inner)
        grandparent = GrandParentConfig(depth=7, child=outer)

        grandparent.save_config(temp_dir)
        loaded = GrandParentConfig.from_config(save_directory=temp_dir)

        assert loaded.depth == 7
        assert isinstance(loaded.child, OuterConfig)
        assert loaded.child.name == "mid"
        assert isinstance(loaded.child.inner, InnerConfig)
        assert loaded.child.inner.width == 256
        assert loaded.child.inner.activation == "elu"

    def test_nested_instances_are_distinct_objects(self):
        inner = InnerConfig(width=10)
        outer = OuterConfig(inner=inner)

        loaded = OuterConfig.from_config(config=dict(outer.config))

        # Reconstruction produces a new, independent instance.
        assert loaded.inner is not inner
        assert loaded.inner.width == inner.width


class TestNestedInContainers:
    """Test ConfigMixin instances nested inside lists and dicts."""

    def test_list_of_configmixins(self, temp_dir):
        items = [InnerConfig(width=1), InnerConfig(width=2, activation="gelu")]
        config = ConfigWithNestedContainers(items=items)

        config.save_config(temp_dir)
        loaded = ConfigWithNestedContainers.from_config(save_directory=temp_dir)

        assert len(loaded.items) == 2
        assert all(isinstance(item, InnerConfig) for item in loaded.items)
        assert loaded.items[0].width == 1
        assert loaded.items[1].width == 2
        assert loaded.items[1].activation == "gelu"

    def test_dict_of_configmixins(self, temp_dir):
        mapping = {"a": InnerConfig(width=3), "b": InnerConfig(width=4)}
        config = ConfigWithNestedContainers(mapping=mapping)

        config.save_config(temp_dir)
        loaded = ConfigWithNestedContainers.from_config(save_directory=temp_dir)

        assert isinstance(loaded.mapping["a"], InnerConfig)
        assert isinstance(loaded.mapping["b"], InnerConfig)
        assert loaded.mapping["a"].width == 3
        assert loaded.mapping["b"].width == 4


class TestNestedSpawn:
    """Test that spawn works with nested ConfigMixin parameters."""

    def test_spawn_preserves_nested_config(self):
        inner = InnerConfig(width=99, activation="mish")
        outer = OuterConfig(name="spawn_me", inner=inner)

        spawned = outer.spawn()

        assert spawned is not outer
        assert isinstance(spawned.inner, InnerConfig)
        assert spawned.inner.width == 99
        assert spawned.inner.activation == "mish"

    def test_spawn_nested_independence(self):
        inner = InnerConfig(width=5)
        outer = OuterConfig(inner=inner)

        spawned = outer.spawn()
        spawned.inner.width = 12345

        # Mutating the spawned nested instance must not affect the original.
        assert outer.inner.width == 5
