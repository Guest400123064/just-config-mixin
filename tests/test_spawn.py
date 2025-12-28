#!/usr/bin/env python3

"""Tests for ConfigMixin.spawn() method.

This module tests the spawn functionality which creates a new instance
of the same class with the same configuration but without state inheritance.
"""

import pytest

from configmixin import ConfigMixin, register_to_config


class StatefulModel(ConfigMixin):
    """Model with stateful attributes for testing spawn."""

    config_name = "stateful_model.json"

    @register_to_config
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Stateful attributes not in config
        self.trained_epochs = 0
        self.weights = []

    def train(self, epochs: int):
        """Simulate training that modifies state."""
        self.trained_epochs += epochs
        self.weights.append(f"weights_epoch_{self.trained_epochs}")


class ConfigWithPrivateParams(ConfigMixin):
    """Configuration with private parameters."""

    config_name = "config_private.json"

    @register_to_config
    def __init__(self, public_param: int = 50, _private_param: str = "private"):
        self.public_param = public_param
        self._private_param = _private_param
        self.internal_state = []


class ConfigWithIgnoredParams(ConfigMixin):
    """Configuration with ignored parameters (e.g., runtime objects)."""

    config_name = "config_ignored.json"
    ignore_for_config = ["runtime_object", "device"]

    @register_to_config
    def __init__(self, tracked_param: int = 100, runtime_object=None, device: str = "cpu"):
        self.tracked_param = tracked_param
        self.runtime_object = runtime_object
        self.device = device
        self.state_counter = 0


class MockComplexObject:
    """Mock complex object for testing runtime_kwargs."""

    def __init__(self, name: str, data):
        self.name = name
        self.data = data

    def __eq__(self, other):
        return (
            isinstance(other, MockComplexObject)
            and self.name == other.name
            and self.data == other.data
        )


class TestSpawnBasic:
    """Test basic spawn functionality."""

    def test_spawn_creates_new_instance(self):
        """Test that spawn creates a distinct new instance."""
        model = StatefulModel(hidden_size=1024, num_layers=24)
        spawned = model.spawn()

        # Should be different objects
        assert spawned is not model
        assert id(spawned) != id(model)

    def test_spawn_preserves_config(self):
        """Test that spawn preserves configuration parameters."""
        model = StatefulModel(hidden_size=1024, num_layers=24)
        spawned = model.spawn()

        # Config should match
        assert spawned.hidden_size == model.hidden_size
        assert spawned.num_layers == model.num_layers
        assert spawned.config == model.config

    def test_spawn_does_not_inherit_state(self):
        """Test that spawn does NOT inherit state."""
        model = StatefulModel(hidden_size=512)
        model.train(5)
        model.train(3)

        spawned = model.spawn()

        # State should be reset to initial values
        assert model.trained_epochs == 8
        assert len(model.weights) == 2

        assert spawned.trained_epochs == 0
        assert spawned.weights == []

    def test_spawn_with_default_values(self):
        """Test spawn with all default configuration values."""
        model = StatefulModel()  # All defaults
        spawned = model.spawn()

        assert spawned.hidden_size == 768
        assert spawned.num_layers == 12

    def test_spawn_multiple_times(self):
        """Test that spawn can be called multiple times."""
        model = StatefulModel(hidden_size=256, num_layers=6)

        spawned1 = model.spawn()
        spawned2 = model.spawn()
        spawned3 = spawned1.spawn()

        # All should have same config
        assert spawned1.hidden_size == spawned2.hidden_size == spawned3.hidden_size == 256
        assert spawned1.num_layers == spawned2.num_layers == spawned3.num_layers == 6

        # But all should be different objects
        assert len({id(model), id(spawned1), id(spawned2), id(spawned3)}) == 4


class TestSpawnWithPrivateParams:
    """Test spawn with private parameters."""

    def test_spawn_fails_with_required_private_params_no_runtime_kwargs(self):
        """Test that spawn fails when private params are required but not provided."""
        config = ConfigWithPrivateParams(public_param=100, _private_param="secret")

        # spawn should fail because _private_param is not in config
        # but is required by __init__
        with pytest.raises(KeyError, match="missing required parameter"):
            config.spawn()

    def test_spawn_succeeds_with_private_params_via_runtime_kwargs(self):
        """Test that spawn succeeds when private params are provided via runtime_kwargs."""
        config = ConfigWithPrivateParams(public_param=100, _private_param="secret")
        config.internal_state.append("item1")

        # Spawn with runtime_kwargs should work
        spawned = config.spawn(runtime_kwargs={"_private_param": "new_secret"})

        # Config params preserved
        assert spawned.public_param == 100

        # Private param from runtime_kwargs
        assert spawned._private_param == "new_secret"

        # State not inherited
        assert spawned.internal_state == []


class TestSpawnWithIgnoredParams:
    """Test spawn with ignored parameters (like runtime objects)."""

    def test_spawn_fails_with_ignored_params_no_runtime_kwargs(self):
        """Test that spawn fails when ignored params are required but not provided."""
        mock_obj = MockComplexObject("test", [1, 2, 3])
        config = ConfigWithIgnoredParams(tracked_param=200, runtime_object=mock_obj, device="gpu")

        # spawn should fail because runtime_object and device are not in config
        with pytest.raises(KeyError, match="missing required parameter"):
            config.spawn()

    def test_spawn_succeeds_with_ignored_params_via_runtime_kwargs(self):
        """Test that spawn succeeds when ignored params are provided via runtime_kwargs."""
        mock_obj = MockComplexObject("test", [1, 2, 3])
        config = ConfigWithIgnoredParams(tracked_param=200, runtime_object=mock_obj, device="gpu")
        config.state_counter = 42

        # Spawn with different runtime objects
        new_mock_obj = MockComplexObject("new_test", [4, 5, 6])
        spawned = config.spawn(runtime_kwargs={"runtime_object": new_mock_obj, "device": "tpu"})

        # Config params preserved
        assert spawned.tracked_param == 200

        # Runtime params from runtime_kwargs (not inherited from original)
        assert spawned.runtime_object == new_mock_obj
        assert spawned.device == "tpu"

        # State not inherited
        assert spawned.state_counter == 0

    def test_spawn_with_partial_runtime_kwargs(self):
        """Test that spawn works with partial runtime_kwargs when some have defaults."""
        config = ConfigWithIgnoredParams(tracked_param=150, device="gpu")
        config.state_counter = 10

        # Only provide runtime_object, let device fail or require it
        # Actually, we need to provide all required ignored params
        spawned = config.spawn(runtime_kwargs={"runtime_object": None, "device": "cpu"})

        assert spawned.tracked_param == 150
        assert spawned.runtime_object is None
        assert spawned.device == "cpu"
        assert spawned.state_counter == 0


class TestSpawnIndependence:
    """Test independence between original and spawned instances."""

    def test_spawned_instances_are_independent(self):
        """Test that modifying spawned instance doesn't affect original."""
        model = StatefulModel(hidden_size=512)
        model.train(5)

        spawned = model.spawn()
        spawned.train(10)

        # Original unchanged
        assert model.trained_epochs == 5
        assert len(model.weights) == 1

        # Spawned has its own state
        assert spawned.trained_epochs == 10
        assert len(spawned.weights) == 1

    def test_config_modifications_dont_affect_spawn(self):
        """Test that modifying config before spawn works correctly."""
        model = StatefulModel(hidden_size=256)

        # Spawn captures current config state
        spawned1 = model.spawn()

        # Even if we change the original instance attributes
        model.hidden_size = 512

        # New spawn should still use the original config
        spawned2 = model.spawn()

        # Both spawns should have the config value, not the modified attribute
        assert spawned1.hidden_size == 256
        assert spawned2.hidden_size == 256
        assert model.hidden_size == 512  # Original was modified


class TestSpawnEdgeCases:
    """Test edge cases for spawn functionality."""

    def test_spawn_preserves_class_type(self):
        """Test that spawn preserves the exact class type."""
        model = StatefulModel(hidden_size=768)
        spawned = model.spawn()

        assert type(spawned) is type(model)
        assert isinstance(spawned, StatefulModel)
        assert spawned.__class__.__name__ == "StatefulModel"

    def test_spawn_repr(self):
        """Test that spawned instance has proper repr."""
        model = StatefulModel(hidden_size=384, num_layers=8)
        spawned = model.spawn()

        # Both should have similar repr (with same config)
        assert "StatefulModel" in repr(spawned)
        assert "384" in repr(spawned)
        assert "8" in repr(spawned)

    def test_spawn_with_empty_runtime_kwargs(self):
        """Test that spawn works with empty runtime_kwargs dict."""
        model = StatefulModel(hidden_size=512)
        spawned = model.spawn(runtime_kwargs={})

        assert spawned.hidden_size == 512

    def test_spawn_runtime_kwargs_override_behavior(self):
        """Test that runtime_kwargs properly override parameters."""
        config = ConfigWithPrivateParams(public_param=100, _private_param="original")

        # Spawn with different private param value
        spawned1 = config.spawn(runtime_kwargs={"_private_param": "value1"})
        spawned2 = config.spawn(runtime_kwargs={"_private_param": "value2"})

        # Each spawn should have its own runtime_kwargs value
        assert spawned1._private_param == "value1"
        assert spawned2._private_param == "value2"
        assert config._private_param == "original"  # Original unchanged
