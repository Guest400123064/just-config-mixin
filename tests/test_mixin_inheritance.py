#!/usr/bin/env python3

"""Tests for ConfigMixin inheritance patterns in ML workflows.

This module tests inheritance scenarios common in ML workflows, such as
adding ConfigMixin to neural network modules, trainers, and other base classes.
Mock classes are used instead of actual ML frameworks to keep dependencies light.
"""

from typing import Any

from configmixin import ConfigMixin, register_to_config


class MockModule:
    """Mock neural network module (like nn.Module)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = True
        self._parameters = {}

    def forward(self, x):
        return x

    def train(self, mode: bool = True):
        self.training = mode
        return self


class MockOptimizer:
    """Mock optimizer (like torch.optim.Adam)."""

    def __init__(self, parameters, lr: float = 0.001):
        self.param_groups = [{"params": parameters, "lr": lr}]
        self.lr = lr

    def step(self):
        pass

    def zero_grad(self):
        pass


class MockDataset:
    """Mock dataset class."""

    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.size = 1000

    def __len__(self):
        return self.size


# Test Configuration Classes with Multiple Inheritance

class ModelConfig(MockModule, ConfigMixin):
    """Neural network model with configuration mixin."""

    config_name = "model_config.json"
    ignore_for_config = ["optimizer", "_parameters"]

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        optimizer: MockOptimizer = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.optimizer = optimizer

        # Simulate model parameters
        self._parameters = {"weights": f"tensor_{hidden_size}x{num_layers}"}


class TrainerConfig(ConfigMixin):
    """Trainer class with configuration mixin."""

    config_name = "trainer_config.json"
    ignore_for_config = ["model", "dataset", "optimizer"]

    @register_to_config
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        model: MockModule = None,
        dataset: MockDataset = None,
        optimizer: MockOptimizer = None
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.step_count = 0

    def train_step(self):
        self.step_count += 1
        return f"Training step {self.step_count}"


class MultiInheritanceConfig(MockModule, ConfigMixin):
    """Complex multiple inheritance scenario."""

    config_name = "multi_config.json"
    ignore_for_config = ["_internal_state"]

    @register_to_config
    def __init__(
        self,
        model_param: int = 64,
        training_param: float = 0.5,
        _internal_state: Any = None
    ):
        super().__init__()
        self.model_param = model_param
        self.training_param = training_param
        self._internal_state = _internal_state or {"initialized": True}


# Test Classes

class TestMLWorkflowInheritance:
    """Test ConfigMixin with ML workflow inheritance patterns."""

    def test_neural_module_inheritance(self, temp_dir):
        """Test ConfigMixin with neural network module inheritance."""
        optimizer = MockOptimizer([], lr=0.01)
        model = ModelConfig(hidden_size=256, num_layers=5, dropout=0.2, optimizer=optimizer)

        # Test that both MockModule and ConfigMixin functionality work
        assert model.training is True  # MockModule functionality
        assert model.hidden_size == 256  # ConfigMixin functionality

        # Test configuration saving/loading
        model.save_config(temp_dir)

        # Load with runtime_kwargs for ignored parameters
        runtime_kwargs = {"optimizer": optimizer}
        loaded_model = ModelConfig.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_model.hidden_size == 256
        assert loaded_model.num_layers == 5
        assert loaded_model.dropout == 0.2
        assert loaded_model.optimizer == optimizer
        assert loaded_model.training is True  # MockModule state preserved

        # Test that model methods still work
        assert loaded_model.forward("test") == "test"
        loaded_model.train(False)
        assert loaded_model.training is False

    def test_trainer_configuration(self, temp_dir):
        """Test trainer class with complex runtime dependencies."""
        model = ModelConfig()
        dataset = MockDataset("/path/to/data")
        optimizer = MockOptimizer(model._parameters, lr=0.002)

        trainer = TrainerConfig(
            learning_rate=0.002,
            batch_size=64,
            epochs=20,
            model=model,
            dataset=dataset,
            optimizer=optimizer
        )

        # Test trainer functionality
        assert trainer.train_step() == "Training step 1"
        assert trainer.step_count == 1

        # Test configuration roundtrip
        trainer.save_config(temp_dir)

        runtime_kwargs = {
            "model": model,
            "dataset": dataset,
            "optimizer": optimizer
        }
        loaded_trainer = TrainerConfig.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_trainer.learning_rate == 0.002
        assert loaded_trainer.batch_size == 64
        assert loaded_trainer.epochs == 20
        assert loaded_trainer.model == model
        assert loaded_trainer.dataset == dataset
        assert loaded_trainer.optimizer == optimizer
        assert loaded_trainer.step_count == 0  # Fresh instance

    def test_multiple_inheritance_mro(self, temp_dir):
        """Test Method Resolution Order (MRO) with multiple inheritance."""
        config = MultiInheritanceConfig(model_param=128, training_param=0.8)

        # Verify MRO includes both parent classes
        mro_names = [cls.__name__ for cls in MultiInheritanceConfig.__mro__]
        assert "MockModule" in mro_names
        assert "ConfigMixin" in mro_names

        # Test that both parent functionalities work
        assert config.training is True  # MockModule
        assert hasattr(config, "config")  # ConfigMixin
        assert config.model_param == 128  # ConfigMixin

        # Test configuration functionality
        config.save_config(temp_dir)
        runtime_kwargs = {"_internal_state": {"initialized": True}}
        loaded_config = MultiInheritanceConfig.from_config(
            save_directory=temp_dir, runtime_kwargs=runtime_kwargs
        )

        assert loaded_config.model_param == 128
        assert loaded_config.training_param == 0.8
        assert loaded_config.training is True
        assert loaded_config._internal_state == {"initialized": True}

    def test_inheritance_with_super_calls(self, temp_dir):
        """Test that super() calls work correctly in inheritance chain."""

        class ExtendedModel(ModelConfig):
            """Model that extends ModelConfig with additional functionality."""

            config_name = "extended_model_config.json"
            ignore_for_config = ["optimizer", "_parameters", "extended_state"]

            # Don't use @register_to_config on derived class to avoid conflicts
            def __init__(
                self,
                hidden_size: int = 128,
                num_layers: int = 3,
                dropout: float = 0.1,
                extended_param: str = "default",
                optimizer: MockOptimizer = None
            ):
                # Call parent with @register_to_config behavior
                super().__init__(hidden_size, num_layers, dropout, optimizer)
                self.extended_param = extended_param
                self.extended_state = {"extra": "data"}

                # Manually register the additional param to config
                if hasattr(self, '_internal_dict'):
                    self._internal_dict['extended_param'] = extended_param

        model = ExtendedModel(hidden_size=512, extended_param="custom")

        # Verify inheritance worked correctly
        assert model.hidden_size == 512
        assert model.extended_param == "custom"
        assert model.training is True  # From MockModule

        # Test that config contains both base and extended params
        assert "hidden_size" in model.config
        assert "extended_param" in model.config
        assert model.config["hidden_size"] == 512
        assert model.config["extended_param"] == "custom"


class TestInheritanceEdgeCases:
    """Test edge cases and potential conflicts in inheritance."""

    def test_config_name_inheritance(self):
        """Test that config_name is properly inherited/overridden."""

        class BaseConfig(ConfigMixin):
            config_name = "base_config.json"

            @register_to_config
            def __init__(self, param: int = 1):
                self.param = param

        class DerivedConfig(BaseConfig):
            config_name = "derived_config.json"  # Override parent config_name

            @register_to_config
            def __init__(self, param: int = 1, extra: str = "test"):
                super().__init__(param)
                self.extra = extra

        base = BaseConfig()
        derived = DerivedConfig()

        assert base.config_name == "base_config.json"
        assert derived.config_name == "derived_config.json"

    def test_ignore_for_config_inheritance(self):
        """Test that ignore_for_config is properly inherited/extended."""

        class BaseConfig(ConfigMixin):
            config_name = "base_ignore.json"
            ignore_for_config = ["base_ignored"]

            @register_to_config
            def __init__(self, tracked: int = 1, base_ignored: str = "ignored"):
                self.tracked = tracked
                self.base_ignored = base_ignored

        class DerivedConfig(BaseConfig):
            # Extend parent's ignore list
            ignore_for_config = BaseConfig.ignore_for_config + ["derived_ignored"]

            @register_to_config
            def __init__(self, tracked: int = 1, base_ignored: str = "ignored", derived_ignored: int = 42):
                super().__init__(tracked, base_ignored)
                self.derived_ignored = derived_ignored

        config = DerivedConfig()

        # Only tracked parameter should be in config
        config_keys = set(config.config.keys()) - {"__notes__"}
        assert config_keys == {"tracked"}
        assert "base_ignored" not in config.config
        assert "derived_ignored" not in config.config

    def test_diamond_inheritance_pattern(self, temp_dir):
        """Test diamond inheritance pattern (multiple paths to same base)."""

        # Simplified diamond pattern - only use @register_to_config at the leaf level
        class BaseA:
            def __init__(self, base_param: int = 10):
                self.base_param = base_param

        class BaseB:
            def __init__(self, extra_param: str = "default"):
                self.extra_param = extra_param

        class Diamond(BaseA, BaseB, ConfigMixin):
            config_name = "diamond_config.json"

            @register_to_config
            def __init__(self, base_param: int = 10, extra_param: str = "default", diamond_param: float = 1.0):
                BaseA.__init__(self, base_param)
                BaseB.__init__(self, extra_param)
                ConfigMixin.__init__(self)
                self.diamond_param = diamond_param

        diamond = Diamond(base_param=20, extra_param="custom", diamond_param=2.5)

        assert diamond.base_param == 20
        assert diamond.extra_param == "custom"
        assert diamond.diamond_param == 2.5

        # Test configuration roundtrip
        diamond.save_config(temp_dir)
        loaded_diamond = Diamond.from_config(save_directory=temp_dir)

        assert loaded_diamond.base_param == 20
        assert loaded_diamond.extra_param == "custom"
        assert loaded_diamond.diamond_param == 2.5
