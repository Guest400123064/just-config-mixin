[![Python 3.10](https://img.shields.io/badge/python-%203.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![test](https://github.com/Guest400123064/configmixin/actions/workflows/test.yaml/badge.svg)](https://github.com/Guest400123064/configmixin/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/Guest400123064/configmixin/branch/main/graph/badge.svg?token=6TG5R1AYQK)](https://codecov.io/gh/Guest400123064/configmixin)
[![PyPI](https://img.shields.io/pypi/v/just-config-mixin)](https://pypi.org/project/just-config-mixin/)

# configmixin

An ultra lightweight configuration management library for machine learning inspired by the Hugging Face 🤗 [`diffusers`](https://github.com/huggingface/diffusers) library. Add automatic configuration handling to any class with a simple mixin pattern. Please refer to the [documentation](https://guest400123064.github.io/just-config-mixin/) for more details.

## Features

- 🔗 **Mixin Pattern**: Add config management to models, trainers, or any class
- 💾 **Save/Load**: Automatic JSON serialization with customizable serialization logic
- ⚡ **Decorator Support**: `@register_to_config` for automatic parameter registration
- 🎯 **Selective Exclusion**: Control which parameters are saved to config

## Installation

From PyPI:

```bash
pip install just-config-mixin
```

If you are interested in the experimental (i.e., unstable and undertested) version, you can install it from GitHub:

```bash
pip install git+https://github.com/Guest400123064/configmixin.git
```

## Core Use Cases

### 1. PyTorch Model with Configuration

```python
import torch.nn as nn
from configmixin import ConfigMixin, register_to_config


class TransformerModel(nn.Module, ConfigMixin):
    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Build model layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True)
            for _ in range(num_layers)
        ])


model = TransformerModel(hidden_size=1024, num_layers=24)
print(model.config)
```

### 2. Configuration Save/Load

```python
# Save configuration
model.save_config("./checkpoints/experiment_1")

# Load model with exact same configuration
loaded_model = TransformerModel.from_config(save_directory="./checkpoints/experiment_1")

# Or load from config dict
config_dict = {"vocab_size": 50000, "hidden_size": 512, "num_layers": 6, "dropout": 0.2}
model_from_dict = TransformerModel.from_config(config=config_dict)
```

### 3. Training Pipeline with Ignored Parameters

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
        # Runtime objects (ignored)
        model=None,
        optimizer=None,
        # Private params (auto-ignored due to underscore)
        _internal_state=None
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.model = model
        self.optimizer = optimizer
        self._internal_state = _internal_state

# Only hyperparameters are saved, not runtime objects
trainer = ModelTrainer(
    learning_rate=2e-4,
    batch_size=64,
    model=some_model,
    optimizer=some_optimizer
)

trainer.save_config("./experiments/run_001")

# Config only contains: learning_rate, batch_size, num_epochs, weight_decay
# and the runtime objects are passed via `runtime_kwargs` in `from_config()`
trainer = ModelTrainer.from_config(save_directory="./experiments/run_001", runtime_kwargs={"model": <some_model>})
```

## Complete Workflow

```python
# Create components with configurations
model = TransformerModel(hidden_size=1024, num_layers=24)
trainer = ModelTrainer(learning_rate=1e-4, batch_size=64)

# Save all configurations to experiment directory
experiment_dir = "./experiments/run_001"
model.save_config(experiment_dir)
trainer.save_config(experiment_dir)

# Later: reproduce exact setup
loaded_model = TransformerModel.from_config(save_directory=experiment_dir)
loaded_trainer = ModelTrainer.from_config(save_directory=experiment_dir, runtime_kwargs={"model": loaded_model})
```

## Why configmixin?

Perfect for ML workflows where you need:
- **Reproducible experiments** with exact parameter tracking
- **Easy hyperparameter management** built into your classes
- **Clean separation** between config and runtime state

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

Apache License 2.0 - see LICENSE file for details.
