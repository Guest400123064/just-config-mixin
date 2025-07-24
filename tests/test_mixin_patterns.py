#!/usr/bin/env python3

"""
Test suite for ConfigMixin as a mixin pattern.

This test suite focuses on realistic use cases where ConfigMixin is mixed into
model classes, data loaders, trainers, and experiment managers - demonstrating
the primary intended usage pattern for machine learning and experimentation workflows.
"""

import json
import pathlib
import tempfile
import pytest
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

from yacm import ConfigMixin, register_to_config, FrozenDict


# Mock base classes to simulate common ML frameworks
class Module(ABC):
    """Mock base class simulating a neural network module (like torch.nn.Module)."""

    def __init__(self):
        self.training = True
        self._parameters = {}

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass

    def state_dict(self):
        """Get model state."""
        return self._parameters.copy()


class DataLoader:
    """Mock data loader class."""

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Mock iteration
        for i in range(len(self.dataset) // self.batch_size):
            yield self.dataset[i * self.batch_size:(i + 1) * self.batch_size]


class Dataset:
    """Mock dataset class."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Model classes using ConfigMixin as a mixin
class TransformerModel(Module, ConfigMixin):
    """Transformer model with configuration management."""

    config_name = "transformer_config.json"
    ignore_for_config = ["_cache", "training_stats"]

    @register_to_config
    def __init__(self,
                 # Architecture parameters
                 vocab_size: int = 30000,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 max_position_embeddings: int = 512,

                 # Regularization
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,

                 # Advanced options
                 activation_function: str = "gelu",
                 use_cache: bool = True,
                 initializer_range: float = 0.02,

                 # Architecture variants
                 tie_word_embeddings: bool = True,
                 position_embedding_type: str = "absolute",

                 # Training-specific (ignored)
                 _cache: Dict = None,
                 training_stats: Dict = None):

        # Initialize parent classes
        Module.__init__(self)

        # Set architecture parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        # Set regularization parameters
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps

        # Set advanced options
        self.activation_function = activation_function
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        # Set architecture variants
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_type = position_embedding_type

        # Private/ignored attributes
        self._cache = _cache or {}
        self.training_stats = training_stats or {"epochs": 0, "steps": 0}

        # Initialize model parameters (mock)
        self._parameters = {"embeddings": "mock_embeddings", "layers": "mock_layers"}

    def forward(self, x):
        """Mock forward pass."""
        return f"TransformerModel output for input: {x}"

    def get_num_parameters(self):
        """Get number of model parameters."""
        return self.hidden_size * self.num_layers * 1000  # Mock calculation


class CNNModel(Module, ConfigMixin):
    """CNN model with configuration management."""

    config_name = "cnn_config.json"
    ignore_for_config = ["device", "_buffers"]

    @register_to_config
    def __init__(self,
                 # Architecture
                 input_channels: int = 3,
                 num_classes: int = 10,
                 channel_dims: List[int] = None,
                 kernel_sizes: List[int] = None,
                 stride_sizes: List[int] = None,

                 # Regularization
                 dropout_rate: float = 0.5,
                 batch_norm: bool = True,

                 # Activation
                 activation: str = "relu",
                 final_activation: str = "softmax",

                 # Pooling
                 pooling_type: str = "max",
                 global_pooling: bool = True,

                 # Training settings (ignored)
                 device: str = "cpu",
                 _buffers: Dict = None):

        Module.__init__(self)

        # Architecture
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.channel_dims = channel_dims or [32, 64, 128]
        self.kernel_sizes = kernel_sizes or [3, 3, 3]
        self.stride_sizes = stride_sizes or [1, 1, 1]

        # Regularization
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Activation
        self.activation = activation
        self.final_activation = final_activation

        # Pooling
        self.pooling_type = pooling_type
        self.global_pooling = global_pooling

        # Ignored parameters
        self.device = device
        self._buffers = _buffers or {}

    def forward(self, x):
        """Mock forward pass."""
        return f"CNNModel output for input shape: {x}"


class ConfigurableDataLoader(DataLoader, ConfigMixin):
    """Data loader with configuration management."""

    config_name = "dataloader_config.json"
    ignore_for_config = ["_iterator", "dataset"]

    @register_to_config
    def __init__(self,
                 # Data parameters
                 data_path: Union[str, pathlib.Path] = "./data",
                 split: str = "train",

                 # Loading parameters
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = False,

                 # Preprocessing
                 normalize: bool = True,
                 augment: bool = True,
                 augmentation_prob: float = 0.5,

                 # Data filtering
                 max_sequence_length: Optional[int] = None,
                 min_sequence_length: Optional[int] = None,
                 filter_empty: bool = True,

                 # Advanced options
                 cache_data: bool = False,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = False,

                 # Internal (ignored)
                 dataset = None,
                 _iterator = None):

        # Create mock dataset
        if dataset is None:
            mock_data = list(range(1000))  # Mock data
            dataset = Dataset(mock_data)

        # Initialize DataLoader
        DataLoader.__init__(self, dataset, batch_size, shuffle)

        # Set data parameters
        self.data_path = pathlib.Path(data_path)
        self.split = split

        # Set loading parameters
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Set preprocessing
        self.normalize = normalize
        self.augment = augment
        self.augmentation_prob = augmentation_prob

        # Set data filtering
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.filter_empty = filter_empty

        # Set advanced options
        self.cache_data = cache_data
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        # Ignored attributes
        self._iterator = _iterator


class ModelTrainer(ConfigMixin):
    """Training pipeline with configuration management."""

    config_name = "trainer_config.json"
    ignore_for_config = ["model", "optimizer", "_state"]

    @register_to_config
    def __init__(self,
                 # Training hyperparameters
                 learning_rate: float = 1e-4,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 warmup_steps: int = 1000,

                 # Optimization
                 optimizer_type: str = "adam",
                 weight_decay: float = 0.01,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,

                 # Learning rate scheduling
                 lr_scheduler: str = "cosine",
                 min_lr: float = 1e-6,
                 lr_decay_steps: int = 10000,

                 # Regularization
                 gradient_clipping: Optional[float] = 1.0,
                 dropout_rate: float = 0.1,
                 label_smoothing: float = 0.0,

                 # Training behavior
                 save_every: int = 1000,
                 eval_every: int = 500,
                 log_every: int = 100,
                 early_stopping_patience: int = 10,

                 # Mixed precision
                 use_amp: bool = False,
                 amp_loss_scale: str = "dynamic",

                 # Checkpointing
                 checkpoint_dir: Union[str, pathlib.Path] = "./checkpoints",
                 save_optimizer_state: bool = True,
                 max_checkpoints: int = 5,

                 # Objects (ignored)
                 model = None,
                 optimizer = None,
                 _state: Dict = None):

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

        # Optimization
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Learning rate scheduling
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr
        self.lr_decay_steps = lr_decay_steps

        # Regularization
        self.gradient_clipping = gradient_clipping
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        # Training behavior
        self.save_every = save_every
        self.eval_every = eval_every
        self.log_every = log_every
        self.early_stopping_patience = early_stopping_patience

        # Mixed precision
        self.use_amp = use_amp
        self.amp_loss_scale = amp_loss_scale

        # Checkpointing
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.save_optimizer_state = save_optimizer_state
        self.max_checkpoints = max_checkpoints

        # Ignored objects
        self.model = model
        self.optimizer = optimizer
        self._state = _state or {"step": 0, "epoch": 0}

    def train(self, model, dataloader):
        """Mock training method."""
        return f"Training {model.__class__.__name__} for {self.num_epochs} epochs"


class ExperimentManager(ConfigMixin):
    """Experiment management with configuration."""

    config_name = "experiment_config.json"
    ignore_for_config = ["_results", "_logger", "start_time"]

    @register_to_config
    def __init__(self,
                 # Experiment metadata
                 experiment_name: str = "default_experiment",
                 description: str = "",
                 tags: List[str] = None,

                 # Experiment setup
                 seed: int = 42,
                 device: str = "auto",
                 precision: str = "float32",

                 # Paths and directories
                 output_dir: Union[str, pathlib.Path] = "./experiments",
                 data_dir: Union[str, pathlib.Path] = "./data",
                 checkpoint_dir: Union[str, pathlib.Path] = "./checkpoints",

                 # Logging and monitoring
                 log_level: str = "INFO",
                 log_to_file: bool = True,
                 wandb_project: Optional[str] = None,
                 tensorboard_dir: Optional[str] = None,

                 # Resource management
                 max_memory_gb: Optional[float] = None,
                 max_time_hours: Optional[float] = None,
                 distributed: bool = False,

                 # Experiment tracking
                 track_metrics: List[str] = None,
                 save_predictions: bool = False,
                 save_model_every: int = 5,

                 # Advanced settings
                 resume_from: Optional[str] = None,
                 auto_resume: bool = True,
                 debug_mode: bool = False,

                 # Internal state (ignored)
                 _results: Dict = None,
                 _logger = None,
                 start_time = None):

        # Experiment metadata
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or []

        # Experiment setup
        self.seed = seed
        self.device = device
        self.precision = precision

        # Paths and directories
        self.output_dir = pathlib.Path(output_dir)
        self.data_dir = pathlib.Path(data_dir)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)

        # Logging and monitoring
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.wandb_project = wandb_project
        self.tensorboard_dir = tensorboard_dir

        # Resource management
        self.max_memory_gb = max_memory_gb
        self.max_time_hours = max_time_hours
        self.distributed = distributed

        # Experiment tracking
        self.track_metrics = track_metrics or ["loss", "accuracy"]
        self.save_predictions = save_predictions
        self.save_model_every = save_model_every

        # Advanced settings
        self.resume_from = resume_from
        self.auto_resume = auto_resume
        self.debug_mode = debug_mode

        # Ignored state
        self._results = _results or {}
        self._logger = _logger
        self.start_time = start_time

    def run_experiment(self, model, trainer, dataloader):
        """Mock experiment execution."""
        return f"Running experiment '{self.experiment_name}' with {model.__class__.__name__}"


# Test classes for mixin patterns
class TestModelMixinPatterns:
    """Test ConfigMixin with model classes."""

    def test_transformer_model_basic(self):
        """Test basic TransformerModel functionality."""
        model = TransformerModel(
            vocab_size=50000,
            hidden_size=1024,
            num_layers=24,
            num_attention_heads=16
        )

        # Test model functionality
        assert model.vocab_size == 50000
        assert model.hidden_size == 1024
        assert model.num_layers == 24
        assert model.training is True

        # Test config registration
        assert model.config["vocab_size"] == 50000
        assert model.config["hidden_size"] == 1024
        assert model.config["num_layers"] == 24
        assert "_cache" not in model.config
        assert "training_stats" not in model.config

        # Test model methods
        output = model.forward("test input")
        assert "TransformerModel output" in output

        model.eval()
        assert model.training is False

        num_params = model.get_num_parameters()
        assert num_params == 1024 * 24 * 1000

    def test_cnn_model_configuration(self):
        """Test CNN model with custom configuration."""
        model = CNNModel(
            input_channels=1,
            num_classes=100,
            channel_dims=[64, 128, 256, 512],
            kernel_sizes=[5, 3, 3, 3],
            dropout_rate=0.3,
            activation="leaky_relu"
        )

        # Test configuration
        assert model.config["input_channels"] == 1
        assert model.config["num_classes"] == 100
        assert model.config["channel_dims"] == [64, 128, 256, 512]
        assert model.config["kernel_sizes"] == [5, 3, 3, 3]
        assert model.config["dropout_rate"] == 0.3
        assert model.config["activation"] == "leaky_relu"

        # Test ignored parameters
        assert "device" not in model.config
        assert "_buffers" not in model.config
        assert model.device == "cpu"

    def test_model_save_load_cycle(self):
        """Test saving and loading model configurations."""
        original_model = TransformerModel(
            vocab_size=32000,
            hidden_size=512,
            num_layers=8,
            activation_function="swish",
            position_embedding_type="relative"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save configuration
            original_model.save_config(temp_dir)

            # Load configuration
            loaded_model, unused = TransformerModel.from_config(temp_dir)

            # Test that configuration is preserved
            assert loaded_model.vocab_size == 32000
            assert loaded_model.hidden_size == 512
            assert loaded_model.num_layers == 8
            assert loaded_model.activation_function == "swish"
            assert loaded_model.position_embedding_type == "relative"

            # Test that model functionality works
            assert loaded_model.training is True
            assert loaded_model.get_num_parameters() == 512 * 8 * 1000

    def test_model_with_complex_architecture(self):
        """Test model with complex architectural parameters."""
        model = TransformerModel(
            vocab_size=50257,  # GPT-2 vocab size
            hidden_size=1600,
            num_layers=48,
            num_attention_heads=25,  # Non-standard
            intermediate_size=6400,
            max_position_embeddings=2048,
            hidden_dropout_prob=0.15,
            attention_probs_dropout_prob=0.15,
            tie_word_embeddings=False,
            position_embedding_type="rotary"
        )

        # Verify complex configuration
        config = model.config
        assert config["vocab_size"] == 50257
        assert config["hidden_size"] == 1600
        assert config["num_layers"] == 48
        assert config["num_attention_heads"] == 25
        assert config["intermediate_size"] == 6400
        assert config["max_position_embeddings"] == 2048
        assert config["tie_word_embeddings"] is False
        assert config["position_embedding_type"] == "rotary"


class TestDataLoaderMixinPatterns:
    """Test ConfigMixin with data loader classes."""

    def test_configurable_dataloader_basic(self):
        """Test basic data loader configuration."""
        dataloader = ConfigurableDataLoader(
            batch_size=64,
            shuffle=False,
            num_workers=8,
            normalize=True,
            augment=False
        )

        # Test data loader functionality
        assert dataloader.batch_size == 64
        assert dataloader.shuffle is False
        assert len(dataloader.dataset) == 1000

        # Test config registration
        assert dataloader.config["batch_size"] == 64
        assert dataloader.config["shuffle"] is False
        assert dataloader.config["num_workers"] == 8
        assert dataloader.config["normalize"] is True
        assert dataloader.config["augment"] is False

        # Test ignored parameters
        assert "dataset" not in dataloader.config
        assert "_iterator" not in dataloader.config

    def test_dataloader_with_preprocessing(self):
        """Test data loader with preprocessing configuration."""
        dataloader = ConfigurableDataLoader(
            data_path="/custom/data/path",
            split="validation",
            max_sequence_length=1024,
            min_sequence_length=50,
            augmentation_prob=0.3,
            cache_data=True,
            prefetch_factor=4
        )

        config = dataloader.config
        assert str(config["data_path"]) == "/custom/data/path"
        assert config["split"] == "validation"
        assert config["max_sequence_length"] == 1024
        assert config["min_sequence_length"] == 50
        assert config["augmentation_prob"] == 0.3
        assert config["cache_data"] is True
        assert config["prefetch_factor"] == 4

    def test_dataloader_save_load(self):
        """Test saving and loading data loader configurations."""
        original_loader = ConfigurableDataLoader(
            data_path="./training_data",
            batch_size=128,
            num_workers=16,
            augment=True,
            augmentation_prob=0.7,
            max_sequence_length=2048
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            original_loader.save_config(temp_dir)
            loaded_loader, _ = ConfigurableDataLoader.from_config(temp_dir)

            assert str(loaded_loader.data_path) == "training_data"
            assert loaded_loader.batch_size == 128
            assert loaded_loader.num_workers == 16
            assert loaded_loader.augment is True
            assert loaded_loader.augmentation_prob == 0.7
            assert loaded_loader.max_sequence_length == 2048


class TestTrainerMixinPatterns:
    """Test ConfigMixin with trainer classes."""

    def test_model_trainer_basic(self):
        """Test basic trainer configuration."""
        trainer = ModelTrainer(
            learning_rate=2e-4,
            batch_size=64,
            num_epochs=50,
            optimizer_type="adamw"
        )

        # Test trainer configuration
        assert trainer.learning_rate == 2e-4
        assert trainer.batch_size == 64
        assert trainer.num_epochs == 50
        assert trainer.optimizer_type == "adamw"

        # Test config registration
        config = trainer.config
        assert config["learning_rate"] == 2e-4
        assert config["batch_size"] == 64
        assert config["num_epochs"] == 50
        assert config["optimizer_type"] == "adamw"

        # Test ignored parameters
        assert "model" not in config
        assert "optimizer" not in config
        assert "_state" not in config

    def test_trainer_with_advanced_options(self):
        """Test trainer with advanced training options."""
        trainer = ModelTrainer(
            learning_rate=1e-3,
            lr_scheduler="polynomial",
            gradient_clipping=0.5,
            use_amp=True,
            amp_loss_scale="128",
            early_stopping_patience=5,
            checkpoint_dir="./my_checkpoints",
            save_optimizer_state=False
        )

        config = trainer.config
        assert config["learning_rate"] == 1e-3
        assert config["lr_scheduler"] == "polynomial"
        assert config["gradient_clipping"] == 0.5
        assert config["use_amp"] is True
        assert config["amp_loss_scale"] == "128"
        assert config["early_stopping_patience"] == 5
        assert str(config["checkpoint_dir"]) == "./my_checkpoints"
        assert config["save_optimizer_state"] is False

    def test_trainer_training_method(self):
        """Test trainer functionality with model."""
        model = TransformerModel(hidden_size=256, num_layers=4)
        dataloader = ConfigurableDataLoader(batch_size=32)
        trainer = ModelTrainer(num_epochs=10)

        result = trainer.train(model, dataloader)
        assert "Training TransformerModel for 10 epochs" in result


class TestExperimentMixinPatterns:
    """Test ConfigMixin with experiment management classes."""

    def test_experiment_manager_basic(self):
        """Test basic experiment manager configuration."""
        experiment = ExperimentManager(
            experiment_name="bert_pretraining",
            description="BERT pretraining on custom dataset",
            tags=["nlp", "pretraining", "bert"],
            seed=123
        )

        # Test experiment configuration
        assert experiment.experiment_name == "bert_pretraining"
        assert experiment.description == "BERT pretraining on custom dataset"
        assert experiment.tags == ["nlp", "pretraining", "bert"]
        assert experiment.seed == 123

        # Test config registration
        config = experiment.config
        assert config["experiment_name"] == "bert_pretraining"
        assert config["tags"] == ["nlp", "pretraining", "bert"]
        assert config["seed"] == 123

        # Test ignored parameters
        assert "_results" not in config
        assert "_logger" not in config
        assert "start_time" not in config

    def test_experiment_with_monitoring(self):
        """Test experiment with monitoring and logging configuration."""
        experiment = ExperimentManager(
            experiment_name="gpt_finetuning",
            wandb_project="language_models",
            tensorboard_dir="./tb_logs",
            track_metrics=["perplexity", "bleu", "rouge"],
            log_level="DEBUG",
            save_predictions=True,
            max_time_hours=24.0
        )

        config = experiment.config
        assert config["experiment_name"] == "gpt_finetuning"
        assert config["wandb_project"] == "language_models"
        assert config["tensorboard_dir"] == "./tb_logs"
        assert config["track_metrics"] == ["perplexity", "bleu", "rouge"]
        assert config["log_level"] == "DEBUG"
        assert config["save_predictions"] is True
        assert config["max_time_hours"] == 24.0

    def test_experiment_run_method(self):
        """Test experiment running with integrated components."""
        model = CNNModel(num_classes=1000)
        trainer = ModelTrainer(num_epochs=20)
        dataloader = ConfigurableDataLoader(batch_size=256)
        experiment = ExperimentManager(experiment_name="imagenet_training")

        result = experiment.run_experiment(model, trainer, dataloader)
        assert "Running experiment 'imagenet_training' with CNNModel" in result


class TestIntegratedMixinWorkflows:
    """Test complete workflows with multiple mixin classes."""

    def test_complete_training_pipeline(self):
        """Test complete training pipeline with all components."""
        # Create model
        model = TransformerModel(
            vocab_size=30000,
            hidden_size=768,
            num_layers=12,
            max_position_embeddings=1024
        )

        # Create data loader
        dataloader = ConfigurableDataLoader(
            batch_size=32,
            max_sequence_length=1024,
            augment=True
        )

        # Create trainer
        trainer = ModelTrainer(
            learning_rate=1e-4,
            num_epochs=100,
            gradient_clipping=1.0
        )

        # Create experiment manager
        experiment = ExperimentManager(
            experiment_name="transformer_training",
            tags=["transformer", "nlp"],
            track_metrics=["loss", "accuracy", "perplexity"]
        )

        # Test that all components work together
        assert model.vocab_size == 30000
        assert dataloader.batch_size == 32
        assert trainer.learning_rate == 1e-4
        assert experiment.experiment_name == "transformer_training"

        # Test running the pipeline
        result = experiment.run_experiment(model, trainer, dataloader)
        assert "transformer_training" in result
        assert "TransformerModel" in result

    def test_save_all_configurations(self):
        """Test saving all component configurations to the same directory."""
        model = CNNModel(input_channels=3, num_classes=10)
        dataloader = ConfigurableDataLoader(batch_size=64)
        trainer = ModelTrainer(learning_rate=1e-3, num_epochs=50)
        experiment = ExperimentManager(experiment_name="cnn_experiment")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save all configurations
            model.save_config(temp_dir)
            dataloader.save_config(temp_dir)
            trainer.save_config(temp_dir)
            experiment.save_config(temp_dir)

            # Verify all config files exist
            assert (pathlib.Path(temp_dir) / "cnn_config.json").exists()
            assert (pathlib.Path(temp_dir) / "dataloader_config.json").exists()
            assert (pathlib.Path(temp_dir) / "trainer_config.json").exists()
            assert (pathlib.Path(temp_dir) / "experiment_config.json").exists()

            # Load all configurations
            loaded_model, _ = CNNModel.from_config(temp_dir)
            loaded_dataloader, _ = ConfigurableDataLoader.from_config(temp_dir)
            loaded_trainer, _ = ModelTrainer.from_config(temp_dir)
            loaded_experiment, _ = ExperimentManager.from_config(temp_dir)

            # Verify configurations are preserved
            assert loaded_model.input_channels == 3
            assert loaded_model.num_classes == 10
            assert loaded_dataloader.batch_size == 64
            assert loaded_trainer.learning_rate == 1e-3
            assert loaded_trainer.num_epochs == 50
            assert loaded_experiment.experiment_name == "cnn_experiment"

    def test_configuration_inheritance_with_mixins(self):
        """Test configuration inheritance patterns with mixin classes."""
        class LargeTransformerModel(Module, ConfigMixin):
            """Large transformer model with preset configuration."""

            config_name = "large_transformer_config.json"
            ignore_for_config = ["_cache", "training_stats"]

            @register_to_config
            def __init__(self,
                         vocab_size: int = 50000,
                         hidden_size: int = 1024,
                         num_layers: int = 24,
                         num_attention_heads: int = 16,
                         intermediate_size: int = 4096,
                         max_position_embeddings: int = 2048,

                         # Inherited parameters with different defaults
                         hidden_dropout_prob: float = 0.1,
                         attention_probs_dropout_prob: float = 0.1,
                         layer_norm_eps: float = 1e-12,
                         activation_function: str = "gelu",
                         use_cache: bool = True,
                         initializer_range: float = 0.02,
                         tie_word_embeddings: bool = True,
                         position_embedding_type: str = "absolute",

                         # New parameters
                         model_variant: str = "large",
                         use_gradient_checkpointing: bool = True,

                         # Ignored parameters
                         _cache: Dict = None,
                         training_stats: Dict = None):

                # Initialize Module
                Module.__init__(self)

                # Set all parameters like TransformerModel does
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.num_attention_heads = num_attention_heads
                self.intermediate_size = intermediate_size
                self.max_position_embeddings = max_position_embeddings

                self.hidden_dropout_prob = hidden_dropout_prob
                self.attention_probs_dropout_prob = attention_probs_dropout_prob
                self.layer_norm_eps = layer_norm_eps

                self.activation_function = activation_function
                self.use_cache = use_cache
                self.initializer_range = initializer_range

                self.tie_word_embeddings = tie_word_embeddings
                self.position_embedding_type = position_embedding_type

                self._cache = _cache or {}
                self.training_stats = training_stats or {"epochs": 0, "steps": 0}

                # Set new parameters
                self.model_variant = model_variant
                self.use_gradient_checkpointing = use_gradient_checkpointing

                # Initialize model parameters (mock)
                self._parameters = {"embeddings": "mock_embeddings", "layers": "mock_layers"}

            def forward(self, x):
                """Mock forward pass."""
                return f"LargeTransformerModel output for input: {x}"

            def get_num_parameters(self):
                """Get number of model parameters."""
                return self.hidden_size * self.num_layers * 1000

        # Test inherited model
        large_model = LargeTransformerModel(
            vocab_size=100000,
            model_variant="xl",
            use_gradient_checkpointing=False
        )

        # Test that all parameters are correctly configured
        assert large_model.vocab_size == 100000
        assert large_model.hidden_size == 1024  # Default from LargeTransformerModel
        assert large_model.num_layers == 24
        assert large_model.model_variant == "xl"
        assert large_model.use_gradient_checkpointing is False

        # Test configuration
        config = large_model.config
        assert config["vocab_size"] == 100000
        assert config["hidden_size"] == 1024
        assert config["model_variant"] == "xl"
        assert config["use_gradient_checkpointing"] is False

    def test_mixin_with_multiple_inheritance(self):
        """Test ConfigMixin with multiple inheritance scenarios."""
        class LoggingMixin:
            """Mock logging mixin."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.log_entries = []

            def log(self, message):
                self.log_entries.append(message)

        class LoggingTransformerModel(LoggingMixin, Module, ConfigMixin):
            """Transformer model with logging capabilities."""

            config_name = "logging_transformer_config.json"
            ignore_for_config = ["_cache", "training_stats", "log_entries"]

            @register_to_config
            def __init__(self,
                         vocab_size: int = 30000,
                         hidden_size: int = 768,
                         num_layers: int = 12,
                         num_attention_heads: int = 12,
                         intermediate_size: int = 3072,
                         max_position_embeddings: int = 512,

                         # Regularization
                         hidden_dropout_prob: float = 0.1,
                         attention_probs_dropout_prob: float = 0.1,
                         layer_norm_eps: float = 1e-12,

                         # Advanced options
                         activation_function: str = "gelu",
                         use_cache: bool = True,
                         initializer_range: float = 0.02,

                         # Architecture variants
                         tie_word_embeddings: bool = True,
                         position_embedding_type: str = "absolute",

                         # Training-specific (ignored)
                         _cache: Dict = None,
                         training_stats: Dict = None):

                # Initialize all parent classes
                LoggingMixin.__init__(self)
                Module.__init__(self)

                # Set architecture parameters
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.num_attention_heads = num_attention_heads
                self.intermediate_size = intermediate_size
                self.max_position_embeddings = max_position_embeddings

                # Set regularization parameters
                self.hidden_dropout_prob = hidden_dropout_prob
                self.attention_probs_dropout_prob = attention_probs_dropout_prob
                self.layer_norm_eps = layer_norm_eps

                # Set advanced options
                self.activation_function = activation_function
                self.use_cache = use_cache
                self.initializer_range = initializer_range

                # Set architecture variants
                self.tie_word_embeddings = tie_word_embeddings
                self.position_embedding_type = position_embedding_type

                # Private/ignored attributes
                self._cache = _cache or {}
                self.training_stats = training_stats or {"epochs": 0, "steps": 0}

                # Initialize model parameters (mock)
                self._parameters = {"embeddings": "mock_embeddings", "layers": "mock_layers"}

                self.log("Model initialized")

            def forward(self, x):
                """Mock forward pass."""
                return f"LoggingTransformerModel output for input: {x}"

        # Test multiple inheritance with ConfigMixin
        model = LoggingTransformerModel(
            vocab_size=25000,
            hidden_size=512,
            num_layers=6
        )

        # Test that all functionality works
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert len(model.log_entries) == 1
        assert model.log_entries[0] == "Model initialized"

        # Test config functionality
        assert model.config["vocab_size"] == 25000
        assert model.config["hidden_size"] == 512

        model.log("Forward pass")
        assert len(model.log_entries) == 2
