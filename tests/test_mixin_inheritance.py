#!/usr/bin/env python3

r"""Test suite for ConfigMixin inheritance patterns and usage as a mixin.

This module tests ConfigMixin when used as a mixin class rather than base class,
including:
- Multiple inheritance with abstract base classes
- Diamond inheritance patterns
- Method resolution order (MRO) testing
- Integration with other mixins
- Real-world scenarios with models, trainers, and processors
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from unittest.mock import Mock

from configmixin import ConfigMixin, register_to_config

from .conftest import (
    BaseModel,
    SerializableMixin,
    TrainableMixin,
    assert_config_roundtrip,
    create_config_dict,
)


class LinearModel(BaseModel, ConfigMixin):
    r"""Model class inheriting from BaseModel and using ConfigMixin."""

    config_name = "linear_model.json"

    @register_to_config
    def __init__(self, input_size: int = 128, output_size: int = 64, bias: bool = True):
        super().__init__()  # Call BaseModel.__init__
        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias

    def forward(self, x):
        return f"Linear forward: {x} -> {self.output_size}D output"


class TrainableTransformer(BaseModel, TrainableMixin, ConfigMixin):
    r"""Transformer using multiple inheritance with ConfigMixin."""

    config_name = "trainable_transformer.json"
    ignore_for_config = ["optimizer"]  # Runtime training component

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        vocab_size: int = 30000,
        optimizer: Any = None,
    ):
        super().__init__()  # Calls both BaseModel and TrainableMixin __init__
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.optimizer = optimizer

    def forward(self, x):
        return f"Transformer forward: {x} through {self.num_layers} layers"


class SerializableConfigModel(BaseModel, SerializableMixin, ConfigMixin):
    r"""Model using all three mixins."""

    config_name = "serializable_model.json"

    @register_to_config
    def __init__(
        self,
        model_type: str = "custom",
        version: str = "1.0",
        parameters: Dict[str, Any] = None,
    ):
        super().__init__()
        self.model_type = model_type
        self.version = version
        self.parameters = parameters or {"default": "value"}

    def forward(self, x):
        return f"{self.model_type} v{self.version} forward: {x}"


class DataProcessor(ConfigMixin):
    r"""Non-model class using ConfigMixin for configuration."""

    config_name = "data_processor.json"

    @register_to_config
    def __init__(
        self,
        batch_size: int = 32,
        preprocessing_steps: List[str] = None,
        output_format: str = "tensor",
    ):
        self.batch_size = batch_size
        self.preprocessing_steps = preprocessing_steps or ["normalize", "tokenize"]
        self.output_format = output_format

    def process(self, data):
        return f"Processing {data} with batch_size={self.batch_size}"


class ConfigurableTrainer(TrainableMixin, ConfigMixin):
    r"""Trainer class using TrainableMixin and ConfigMixin."""

    config_name = "trainer_config.json"
    ignore_for_config = ["model", "dataset"]  # Runtime components

    @register_to_config
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 100,
        save_every: int = 10,
        model: Any = None,
        dataset: Any = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.save_every = save_every
        self.model = model
        self.dataset = dataset


class TestConfigMixinAsBaseMixin:
    r"""Test ConfigMixin behavior when used as a base mixin."""

    def test_linear_model_mixin(self):
        r"""Test LinearModel using ConfigMixin as mixin."""
        model = LinearModel(input_size=256, output_size=128, bias=False)

        # Should have BaseModel functionality
        assert hasattr(model, "is_initialized")
        assert model.is_initialized is True
        assert model.get_info() == "LinearModel model"
        assert model.forward("test") == "Linear forward: test -> 128D output"

        # Should have ConfigMixin functionality
        assert model.config["input_size"] == 256
        assert model.config["output_size"] == 128
        assert model.config["bias"] is False
        assert model.config["_class_name"] == "LinearModel"

        # Should be able to serialize config
        json_str = model.get_config_json()
        config_dict = json.loads(json_str)
        assert config_dict["input_size"] == 256

    def test_data_processor_standalone(self):
        r"""Test DataProcessor using ConfigMixin without other base classes."""
        processor = DataProcessor(batch_size=128, output_format="numpy")

        # Should have ConfigMixin functionality
        assert processor.config["batch_size"] == 128
        assert processor.config["output_format"] == "numpy"
        assert processor.config["preprocessing_steps"] == ["normalize", "tokenize"]

        # Should work as expected
        result = processor.process("input_data")
        assert "batch_size=128" in result


class TestMultipleInheritanceWithConfigMixin:
    r"""Test ConfigMixin in multiple inheritance scenarios."""

    def test_trainable_transformer_multiple_inheritance(self):
        r"""Test transformer with multiple inheritance."""
        transformer = TrainableTransformer(
            hidden_size=1024, num_layers=24, num_heads=16, optimizer="AdamW"
        )

        # Should have BaseModel functionality
        assert hasattr(transformer, "is_initialized")
        assert transformer.get_info() == "TrainableTransformer model"
        assert "24 layers" in transformer.forward("input")

        # Should have TrainableMixin functionality
        assert hasattr(transformer, "is_trainable")
        assert transformer.is_trainable is True
        assert transformer.training_steps == 0
        step_result = transformer.train_step()
        assert "Training step 1" in step_result
        assert transformer.training_steps == 1

        # Should have ConfigMixin functionality
        assert transformer.config["hidden_size"] == 1024
        assert transformer.config["num_layers"] == 24
        assert transformer.config["num_heads"] == 16
        assert "optimizer" not in transformer.config  # Should be ignored

        # Optimizer should still be accessible as attribute
        assert transformer.optimizer == "AdamW"

    def test_serializable_config_model_triple_inheritance(self):
        r"""Test model with three mixins."""
        model = SerializableConfigModel(
            model_type="advanced",
            version="2.1",
            parameters={"depth": 50, "width": 1024},
        )

        # BaseModel functionality
        assert model.is_initialized is True
        assert model.get_info() == "SerializableConfigModel model"
        assert "advanced v2.1" in model.forward("data")

        # SerializableMixin functionality
        assert hasattr(model, "serialization_format")
        assert model.serialization_format == "json"
        bytes_data = model.to_bytes()
        assert b"SerializableConfigModel" in bytes_data

        # ConfigMixin functionality
        assert model.config["model_type"] == "advanced"
        assert model.config["version"] == "2.1"
        assert model.config["parameters"] == {"depth": 50, "width": 1024}

    def test_configurable_trainer_inheritance(self):
        r"""Test trainer with multiple inheritance."""
        trainer = ConfigurableTrainer(
            learning_rate=0.01,
            batch_size=256,
            max_epochs=50,
            model="mock_model",
            dataset="mock_dataset",
        )

        # TrainableMixin functionality
        assert trainer.is_trainable is True
        assert trainer.training_steps == 0
        trainer.train_step()
        assert trainer.training_steps == 1

        # ConfigMixin functionality
        assert trainer.config["learning_rate"] == 0.01
        assert trainer.config["batch_size"] == 256
        assert trainer.config["max_epochs"] == 50
        assert "model" not in trainer.config
        assert "dataset" not in trainer.config

        # Runtime components should be accessible
        assert trainer.model == "mock_model"
        assert trainer.dataset == "mock_dataset"


class TestMixinOrderAndMRO:
    r"""Test method resolution order and mixin ordering."""

    def test_mro_with_config_mixin(self):
        r"""Test method resolution order includes ConfigMixin properly."""
        transformer = TrainableTransformer()
        mro = transformer.__class__.__mro__

        # Should include all parent classes
        assert BaseModel in mro
        assert TrainableMixin in mro
        assert ConfigMixin in mro
        assert object in mro

        # ConfigMixin should come after the concrete classes
        config_index = mro.index(ConfigMixin)
        trainable_index = mro.index(TrainableMixin)
        base_model_index = mro.index(BaseModel)

        assert config_index > base_model_index
        assert config_index > trainable_index

    def test_super_calls_work_correctly(self):
        r"""Test that super() calls work correctly in multiple inheritance."""
        # This test ensures that the __init__ chains work properly
        model = SerializableConfigModel(model_type="test_super")

        # All parent classes should have been initialized
        assert hasattr(model, "is_initialized")  # From BaseModel
        assert hasattr(model, "serialization_format")  # From SerializableMixin
        assert hasattr(model, "_internal_dict")  # From ConfigMixin

        # Attributes should be set correctly
        assert model.is_initialized is True
        assert model.serialization_format == "json"
        assert model.config["model_type"] == "test_super"

    def test_mixin_attribute_precedence(self):
        r"""Test that attribute access precedence works correctly with mixins."""
        transformer = TrainableTransformer(hidden_size=512)

        # Instance attributes should take precedence
        assert transformer.hidden_size == 512
        assert transformer.is_trainable is True
        assert transformer.is_initialized is True

        # Config attributes should be accessible
        assert transformer._class_name == "TrainableTransformer"


class TestMixinConfigSaveLoad:
    r"""Test save/load functionality with mixin classes."""

    def test_save_and_load_mixin_config(self, temp_directory):
        r"""Test saving and loading config from mixin class."""
        original = LinearModel(input_size=512, output_size=256, bias=True)

        # Save config
        original.save_config(temp_directory)

        # Load config
        loaded = LinearModel.from_config(temp_directory)

        # Should have same config
        assert loaded.input_size == 512
        assert loaded.output_size == 256
        assert loaded.bias is True

        # Should have base class functionality
        assert loaded.is_initialized is True
        assert loaded.get_info() == "LinearModel model"

    def test_from_config_with_runtime_kwargs_mixin(self):
        r"""Test from_config with runtime kwargs on mixin class."""
        config_dict = create_config_dict(
            "TrainableTransformer",
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            vocab_size=50000,
        )

        runtime_kwargs = {"optimizer": "SGD"}

        transformer = TrainableTransformer.from_config(
            config=config_dict, runtime_kwargs=runtime_kwargs
        )

        # Config values
        assert transformer.hidden_size == 512
        assert transformer.num_layers == 8

        # Runtime values
        assert transformer.optimizer == "SGD"

        # Base class functionality
        assert transformer.is_initialized is True
        assert transformer.is_trainable is True

    def test_round_trip_with_multiple_inheritance(self):
        r"""Test round-trip save/load with multiple inheritance."""
        trainer = ConfigurableTrainer(
            learning_rate=0.005,
            batch_size=128,
            max_epochs=200,
            model=Mock(name="trainable_model"),
            dataset=Mock(name="training_data"),
        )

        # Test config round trip (without runtime components)
        assert_config_roundtrip(trainer)


class TestMixinEdgeCases:
    r"""Test edge cases with ConfigMixin as mixin."""

    def test_diamond_inheritance_scenario(self):
        r"""Test diamond inheritance patterns with ConfigMixin."""

        class Base:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.base_init = True

        class MixinA(Base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mixin_a_init = True

        class MixinB(Base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mixin_b_init = True

        class DiamondConfig(MixinA, MixinB, ConfigMixin):
            config_name = "diamond.json"

            @register_to_config
            def __init__(self, param: str = "diamond"):
                super().__init__()
                self.param = param

        instance = DiamondConfig(param="test_diamond")

        # All mixins should be initialized
        assert instance.base_init is True
        assert instance.mixin_a_init is True
        assert instance.mixin_b_init is True

        # ConfigMixin should work
        assert instance.config["param"] == "test_diamond"

    def test_mixin_with_property_conflicts(self):
        r"""Test ConfigMixin when there might be property name conflicts."""

        class PropertyMixin:
            @property
            def config(self):
                return "mixin_config"

        # ConfigMixin's config property should take precedence due to MRO
        class ConflictModel(PropertyMixin, ConfigMixin):
            config_name = "conflict.json"

            @register_to_config
            def __init__(self, value: int = 42):
                self.value = value

        model = ConflictModel(value=100)

        # ConfigMixin's config should take precedence
        assert isinstance(model.config, dict)  # Should be FrozenDict, not string
        assert model.config["value"] == 100

    def test_mixin_with_abstract_base_class(self):
        r"""Test ConfigMixin with abstract base classes."""

        class AbstractProcessor(ABC):
            @abstractmethod
            def process(self, data):
                pass

        class ConcreteProcessor(AbstractProcessor, ConfigMixin):
            config_name = "concrete_processor.json"

            @register_to_config
            def __init__(self, mode: str = "default"):
                self.mode = mode

            def process(self, data):
                return f"Processing {data} in {self.mode} mode"

        processor = ConcreteProcessor(mode="advanced")

        # Should work as both abstract implementation and config
        assert processor.process("test") == "Processing test in advanced mode"
        assert processor.config["mode"] == "advanced"

    def test_complex_multiple_inheritance_chain(self):
        r"""Test complex inheritance chains with ConfigMixin."""

        class BaseProcessor:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.base_processor = True

        class CacheableMixin:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.cacheable = True

        class VersionedMixin:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.versioned = True

        class ComplexProcessor(
            BaseProcessor, CacheableMixin, VersionedMixin, ConfigMixin
        ):
            config_name = "complex_processor.json"

            @register_to_config
            def __init__(self, name: str = "complex", version: int = 1):
                super().__init__()
                self.name = name
                self.version = version

        processor = ComplexProcessor(name="multi_inheritance", version=2)

        # All mixins should be initialized
        assert processor.base_processor is True
        assert processor.cacheable is True
        assert processor.versioned is True

        # ConfigMixin should work
        assert processor.config["name"] == "multi_inheritance"
        assert processor.config["version"] == 2


class TestMixinCompatibility:
    r"""Test compatibility of ConfigMixin with various inheritance patterns."""

    def test_mixin_with_dataclass_style(self):
        r"""Test ConfigMixin compatibility with dataclass-style patterns."""

        # While we can't use @dataclass with ConfigMixin directly,
        # we can test similar patterns
        class DataClassStyleConfig(ConfigMixin):
            config_name = "dataclass_style.json"

            @register_to_config
            def __init__(
                self,
                name: str = "default",
                value: int = 0,
                enabled: bool = True,
                tags: List[str] = None,
            ):
                self.name = name
                self.value = value
                self.enabled = enabled
                self.tags = tags or []

        config = DataClassStyleConfig(
            name="test", value=42, enabled=False, tags=["tag1", "tag2"]
        )

        assert config.config["name"] == "test"
        assert config.config["value"] == 42
        assert config.config["enabled"] is False
        assert config.config["tags"] == ["tag1", "tag2"]

    def test_mixin_with_factory_pattern(self):
        r"""Test ConfigMixin with factory pattern classes."""

        class ModelFactory:
            @staticmethod
            def create_model(model_type: str):
                if model_type == "linear":
                    return LinearModel()
                elif model_type == "transformer":
                    return TrainableTransformer()
                else:
                    msg = f"Unknown model type: {model_type}"
                    raise ValueError(msg)

        # Test that factory-created objects work with ConfigMixin
        linear = ModelFactory.create_model("linear")
        transformer = ModelFactory.create_model("transformer")

        assert isinstance(linear, ConfigMixin)
        assert isinstance(transformer, ConfigMixin)
        assert linear.config["_class_name"] == "LinearModel"
        assert transformer.config["_class_name"] == "TrainableTransformer"

    def test_mixin_serialization_with_inheritance(self):
        r"""Test that serialization works correctly with inheritance."""
        model = SerializableConfigModel(model_type="serialization_test", version="1.5")

        # Test ConfigMixin serialization
        config_json = model.get_config_json()
        config_dict = json.loads(config_json)
        assert config_dict["model_type"] == "serialization_test"

        # Test SerializableMixin serialization
        bytes_data = model.to_bytes()
        assert isinstance(bytes_data, bytes)

        # Both should work independently
        assert len(config_json) > 0
        assert len(bytes_data) > 0
