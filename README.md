# YACM - Yet Another Config Mixin

A lightweight configuration management library for machine learning and experimentation. YACM provides a `ConfigMixin` class that can be mixed into your model classes, training pipelines, and experiment managers to automatically handle configuration serialization and management.

## Features

- 🔗 **Mixin Pattern**: Add configuration management to any class (models, trainers, data loaders)
- 💾 **Save/Load**: Automatic JSON serialization of configurations with type preservation
- 🔒 **Immutable Configs**: FrozenDict ensures configuration consistency after creation
- 🎯 **Type-Aware**: Handles complex types including paths, lists, and nested dictionaries
- ⚡ **Decorator Support**: `@register_to_config` decorator for automatic parameter registration
- 🧪 **ML-Focused**: Designed for model configurations, hyperparameters, and experiment tracking

## Installation

```bash
# Using poetry (recommended)
poetry add yacm

# Using pip
pip install yacm
```

## Quick Start

### Model with Configuration

```python
import torch.nn as nn
from yacm import ConfigMixin, register_to_config


class TransformerModel(nn.Module, ConfigMixin):
    config_name = "transformer_config.json"

    @register_to_config
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ... model implementation

        # Build your model layers here
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_attention_heads, dropout)
            for _ in range(num_layers)
        ])


model = TransformerModel(hidden_size=1024, num_layers=24)

print(model.hidden_size)  # 1024
print(model.config)       # Access configuration as FrozenDict
```

### Save and Load Model Configurations

```python
# Save model configuration
model.save_config("./model_checkpoints")

# Load and recreate model with same configuration
loaded_model, unused = TransformerModel.from_config("./model_checkpoints")

# Configuration is preserved exactly
assert loaded_model.hidden_size == model.hidden_size
assert loaded_model.num_layers == model.num_layers
```

## Advanced Usage

### Training Pipeline with Configuration

```python
class ModelTrainer(ConfigMixin):
    config_name = "trainer_config.json"
    ignore_for_config = ["model", "optimizer"]  # Exclude runtime objects

    @register_to_config
    def __init__(
        self,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        model=None,
        optimizer=None
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.model = model
        self.optimizer = optimizer


trainer = ModelTrainer(learning_rate=2e-4, num_epochs=50)
trainer.save_config("./experiment_1")
```

### Complete ML Workflow

```python
# Create all components with their configurations
model = TransformerModel(hidden_size=1024, num_layers=24)
trainer = ModelTrainer(learning_rate=1e-4, batch_size=64)

# Save all configurations to the same experiment directory
experiment_dir = "./experiments/run_001"
model.save_config(experiment_dir)
trainer.save_config(experiment_dir)

# Later: reproduce the exact same setup
loaded_model, _ = TransformerModel.from_config(experiment_dir)
loaded_trainer, _ = ModelTrainer.from_config(experiment_dir)
```

### Ignoring Runtime Objects

```python
class ExperimentManager(ConfigMixin):
    config_name = "experiment_config.json"
    ignore_for_config = ["_results", "_logger"]  # Exclude runtime state

    @register_to_config
    def __init__(self, experiment_name: str, seed: int = 42, _logger=None):
        self.experiment_name = experiment_name
        self.seed = seed
        self._logger = _logger  # Not saved to config
```

## API Reference

### ConfigMixin

Mixin class that adds configuration management to any class.

**Class Attributes:**
- `config_name`: Filename for saving configurations (required)
- `ignore_for_config`: List of parameters to exclude from config (optional)

**Methods:**
- `save_config(directory)`: Save configuration to JSON file
- `from_config(directory)`: Load and create instance from saved configuration
- `config`: Property to access configuration as FrozenDict

**Decorator:**
- `@register_to_config`: Automatically register `__init__` parameters to config

## Why Use YACM?

YACM makes it easy to manage configurations in ML workflows:

- **Reproducible Experiments**: Save exact model and training configurations
- **Easy Hyperparameter Management**: Configuration built into your classes
- **Type Safety**: Automatic handling of complex types (lists, paths, etc.)
- **No Boilerplate**: Just inherit from ConfigMixin and use the decorator

Perfect for model training, hyperparameter tuning, and experiment tracking.

## CLI Support (Experimental)

Basic CLI integration is available but experimental:

```python
from yacm import parse_config_from_args

# Parse config from command line
config = parse_config_from_args(TransformerModel,
    ["--hidden-size", "1024", "--num-layers", "24"])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Inspiration

This project is inspired by the `ConfigMixin` class from the [🤗 Diffusers](https://github.com/huggingface/diffusers) library, simplified to work as a standalone package using only the Python standard library and basic dependencies.
