#!/usr/bin/env python3

"""
Integration test suite testing all core components working together.
"""

import json
import pathlib
import tempfile
import pytest
from typing import List, Dict, Optional, Union

from yacm import ConfigMixin, register_to_config, FrozenDict


class ModelConfig(ConfigMixin):
    """Example model configuration for integration testing."""

    config_name = "model_config.json"
    ignore_for_config = ["verbose", "_internal_state"]

    @register_to_config
    def __init__(self,
                 # Model architecture
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,

                 # Training hyperparameters
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 max_epochs: int = 100,

                 # Regularization
                 dropout_rate: float = 0.1,
                 weight_decay: float = 0.01,

                 # Advanced options
                 activation: str = "gelu",
                 use_bias: bool = True,
                 gradient_clipping: Optional[float] = None,

                 # Complex types
                 layer_sizes: List[int] = None,
                 optimizer_params: Dict[str, float] = None,

                 # Private/ignored parameters
                 _internal_state: str = "private",
                 verbose: bool = False):

        # Set all parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        self.activation = activation
        self.use_bias = use_bias
        self.gradient_clipping = gradient_clipping

        self.layer_sizes = layer_sizes or [hidden_size] * num_layers
        self.optimizer_params = optimizer_params or {"beta1": 0.9, "beta2": 0.999}

        self._internal_state = _internal_state
        self.verbose = verbose

        if verbose:
            print(f"Initialized {self.__class__.__name__}")


class DataConfig(ConfigMixin):
    """Example data configuration for integration testing."""

    config_name = "data_config.json"

    @register_to_config
    def __init__(self,
                 # Data loading
                 dataset_path: Union[str, pathlib.Path] = "/data/train",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 shuffle: bool = True,

                 # Data processing
                 max_sequence_length: int = 512,
                 vocab_size: int = 30000,
                 pad_token_id: int = 0,

                 # Data augmentation
                 augmentation_prob: float = 0.15,
                 augmentation_strategies: List[str] = None):

        self.dataset_path = pathlib.Path(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.augmentation_prob = augmentation_prob
        self.augmentation_strategies = augmentation_strategies or ["mask", "replace", "insert"]


class TestCompleteWorkflow:
    """Test complete workflows combining all components."""

    def test_full_config_lifecycle(self):
        """Test complete lifecycle: create, modify, save, load, use."""
        # 1. Create configuration with custom values
        model_config = ModelConfig(
            hidden_size=1024,
            num_layers=24,
            learning_rate=2e-4,
            layer_sizes=[1024, 512, 256],
            optimizer_params={"beta1": 0.95, "beta2": 0.999, "eps": 1e-6},
            verbose=True  # This should be ignored
        )

        # 2. Verify configuration is properly registered
        assert model_config.config["hidden_size"] == 1024
        assert model_config.config["num_layers"] == 24
        assert model_config.config["learning_rate"] == 2e-4
        assert model_config.config["layer_sizes"] == [1024, 512, 256]
        assert model_config.config["optimizer_params"] == {"beta1": 0.95, "beta2": 0.999, "eps": 1e-6}
        assert "verbose" not in model_config.config
        assert "_internal_state" not in model_config.config

        # 3. Test attribute access
        assert model_config.hidden_size == 1024
        assert model_config.verbose is True  # Instance attribute, not in config

        # 4. Test config immutability
        with pytest.raises(Exception):
            model_config.config["hidden_size"] = 2048

        # 5. Save configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            model_config.save_config(temp_dir)

            # 6. Verify file contents
            config_file = pathlib.Path(temp_dir) / "model_config.json"
            assert config_file.exists()

            with open(config_file) as f:
                saved_data = json.load(f)

            assert saved_data["_class_name"] == "ModelConfig"
            assert saved_data["hidden_size"] == 1024
            assert saved_data["layer_sizes"] == [1024, 512, 256]
            assert "verbose" not in saved_data

            # 7. Load configuration
            loaded_config, unused_kwargs = ModelConfig.from_config(temp_dir)

            # 8. Verify loaded configuration
            assert loaded_config.hidden_size == 1024
            assert loaded_config.num_layers == 24
            assert loaded_config.layer_sizes == [1024, 512, 256]
            assert loaded_config.optimizer_params == {"beta1": 0.95, "beta2": 0.999, "eps": 1e-6}
            assert loaded_config.verbose is False  # Default value
            assert unused_kwargs == {}

    def test_multiple_configs_workflow(self):
        """Test working with multiple configuration classes."""
        # Create multiple configs
        model_config = ModelConfig(
            hidden_size=512,
            num_layers=8,
            learning_rate=1e-3
        )

        data_config = DataConfig(
            dataset_path="/custom/data",
            batch_size=64,
            max_sequence_length=1024,
            augmentation_strategies=["mask", "substitute"]
        )

        # Save both configs to same directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_config.save_config(temp_dir)
            data_config.save_config(temp_dir)

            # Verify both files exist
            assert (pathlib.Path(temp_dir) / "model_config.json").exists()
            assert (pathlib.Path(temp_dir) / "data_config.json").exists()

            # Load both configs
            loaded_model, _ = ModelConfig.from_config(temp_dir)
            loaded_data, _ = DataConfig.from_config(temp_dir)

            # Verify they're independent and correct
            assert loaded_model.hidden_size == 512
            assert loaded_model.batch_size == 32  # Model's default batch_size
            assert loaded_data.batch_size == 64   # Data's batch_size
            assert loaded_data.dataset_path == pathlib.Path("/custom/data")

    def test_config_inheritance_and_extension(self):
        """Test config inheritance and extension scenarios."""
        class ExtendedModelConfig(ModelConfig):
            """Extended model config with additional parameters."""

            config_name = "extended_model_config.json"
            ignore_for_config = ModelConfig.ignore_for_config + ["debug_mode"]

            @register_to_config
            def __init__(self,
                         # Inherit all parent parameters
                         hidden_size: int = 768,
                         num_layers: int = 12,
                         num_attention_heads: int = 12,
                         intermediate_size: int = 3072,
                         learning_rate: float = 1e-4,
                         batch_size: int = 32,
                         max_epochs: int = 100,
                         dropout_rate: float = 0.1,
                         weight_decay: float = 0.01,
                         activation: str = "gelu",
                         use_bias: bool = True,
                         gradient_clipping: Optional[float] = None,
                         layer_sizes: List[int] = None,
                         optimizer_params: Dict[str, float] = None,
                         _internal_state: str = "private",
                         verbose: bool = False,

                         # New parameters
                         model_type: str = "transformer",
                         use_attention_bias: bool = False,
                         attention_dropout: float = 0.05,
                         debug_mode: bool = False):

                # Call parent initialization manually
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.num_attention_heads = num_attention_heads
                self.intermediate_size = intermediate_size
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.max_epochs = max_epochs
                self.dropout_rate = dropout_rate
                self.weight_decay = weight_decay
                self.activation = activation
                self.use_bias = use_bias
                self.gradient_clipping = gradient_clipping
                self.layer_sizes = layer_sizes or [hidden_size] * num_layers
                self.optimizer_params = optimizer_params or {"beta1": 0.9, "beta2": 0.999}
                self._internal_state = _internal_state
                self.verbose = verbose

                # Set new parameters
                self.model_type = model_type
                self.use_attention_bias = use_attention_bias
                self.attention_dropout = attention_dropout
                self.debug_mode = debug_mode

        # Test extended config
        extended_config = ExtendedModelConfig(
            hidden_size=2048,
            model_type="gpt",
            use_attention_bias=True,
            attention_dropout=0.1,
            debug_mode=True
        )

        # Verify all parameters are correctly registered
        assert extended_config.config["hidden_size"] == 2048
        assert extended_config.config["model_type"] == "gpt"
        assert extended_config.config["use_attention_bias"] is True
        assert extended_config.config["attention_dropout"] == 0.1
        assert "debug_mode" not in extended_config.config  # Should be ignored
        assert extended_config.debug_mode is True  # But available as instance attribute

        # Test save/load cycle
        with tempfile.TemporaryDirectory() as temp_dir:
            extended_config.save_config(temp_dir)
            loaded_extended, _ = ExtendedModelConfig.from_config(temp_dir)

            assert loaded_extended.hidden_size == 2048
            assert loaded_extended.model_type == "gpt"
            assert loaded_extended.use_attention_bias is True
            assert loaded_extended.attention_dropout == 0.1
            assert loaded_extended.debug_mode is False  # Default value


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_config_validation_errors(self):
        """Test various configuration validation errors."""
        # Test missing required parameter in from_config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = pathlib.Path(temp_dir) / "model_config.json"

            # Create incomplete config
            incomplete_config = {
                "_class_name": "ModelConfig",
                "hidden_size": 768
                # Missing other required parameters like num_layers
            }

            with open(config_file, 'w') as f:
                json.dump(incomplete_config, f)

            # This should work because num_layers has a default
            loaded_config, _ = ModelConfig.from_config(temp_dir)
            assert loaded_config.hidden_size == 768
            assert loaded_config.num_layers == 12  # Default value

    def test_type_safety_in_complex_workflow(self):
        """Test type safety across the workflow."""
        # Create config with various types
        config = ModelConfig(
            hidden_size=1024,
            layer_sizes=[512, 256, 128],
            optimizer_params={"lr": 0.001, "momentum": 0.9},
            gradient_clipping=1.0
        )

        # Save and load
        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_config(temp_dir)
            loaded_config, _ = ModelConfig.from_config(temp_dir)

            # Verify types are preserved
            assert isinstance(loaded_config.hidden_size, int)
            assert isinstance(loaded_config.layer_sizes, list)
            assert isinstance(loaded_config.optimizer_params, dict)
            assert isinstance(loaded_config.gradient_clipping, float)
            assert loaded_config.gradient_clipping == 1.0

    def test_frozen_dict_integration_errors(self):
        """Test FrozenDict behavior in integrated scenarios."""
        config = ModelConfig(hidden_size=512)

        # Config should be frozen
        assert isinstance(config.config, FrozenDict)

        # All modification methods should raise exceptions
        with pytest.raises(Exception):
            config.config["hidden_size"] = 1024

        with pytest.raises(Exception):
            config.config.update({"hidden_size": 1024})

        with pytest.raises(Exception):
            del config.config["hidden_size"]

        with pytest.raises(Exception):
            config.config.pop("hidden_size")

        with pytest.raises(Exception):
            config.config.setdefault("new_param", "value")


class TestPerformanceAndComplexity:
    """Test performance and complexity scenarios."""

    def test_large_config_handling(self):
        """Test handling of large configurations."""
        # Create config with many parameters
        large_layer_sizes = list(range(1000, 0, -10))  # 100 layer sizes
        large_optimizer_params = {f"param_{i}": float(i) for i in range(100)}

        config = ModelConfig(
            layer_sizes=large_layer_sizes,
            optimizer_params=large_optimizer_params
        )

        # Verify large data is handled correctly
        assert len(config.config["layer_sizes"]) == 100
        assert len(config.config["optimizer_params"]) == 100

        # Test save/load with large data
        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_config(temp_dir)
            loaded_config, _ = ModelConfig.from_config(temp_dir)

            assert loaded_config.layer_sizes == large_layer_sizes
            assert loaded_config.optimizer_params == large_optimizer_params

    def test_nested_complex_structures(self):
        """Test deeply nested and complex data structures."""
        complex_optimizer_params = {
            "schedulers": {
                "learning_rate": {
                    "type": "cosine",
                    "params": {"T_max": 100, "eta_min": 1e-6}
                },
                "weight_decay": {
                    "type": "linear",
                    "params": {"start": 0.01, "end": 0.001}
                }
            },
            "optimizers": [
                {"name": "adam", "params": {"lr": 1e-4, "betas": [0.9, 0.999]}},
                {"name": "sgd", "params": {"lr": 1e-3, "momentum": 0.9}}
            ]
        }

        config = ModelConfig(optimizer_params=complex_optimizer_params)

        # Test that nested structures are preserved
        assert config.config["optimizer_params"]["schedulers"]["learning_rate"]["type"] == "cosine"
        assert config.config["optimizer_params"]["optimizers"][0]["params"]["betas"] == [0.9, 0.999]

        # Test save/load preserves complex structures
        with tempfile.TemporaryDirectory() as temp_dir:
            config.save_config(temp_dir)
            loaded_config, _ = ModelConfig.from_config(temp_dir)

            assert loaded_config.optimizer_params == complex_optimizer_params
