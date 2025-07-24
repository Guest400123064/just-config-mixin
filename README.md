# YACM - Yet Another Config Mixin

A simple, standalone configuration management package inspired by the `ConfigMixin` class from the diffusers library. YACM provides an easy way to manage configuration objects with automatic serialization, CLI argument parsing, and more.

## Features

- 🔧 **Easy Configuration Management**: Define configuration classes that automatically handle parameter registration
- 💾 **Save/Load Functionality**: Serialize configurations to/from JSON files
- 🖥️ **CLI Integration**: Automatically generate command-line arguments from configuration classes
- 🔒 **Immutable Configs**: FrozenDict ensures configuration immutability after creation
- 🎯 **Type-Aware**: Automatic type inference for CLI arguments based on type hints
- 🔗 **Decorator Support**: `@register_to_config` decorator for automatic parameter registration

## Installation

```bash
# Using poetry (recommended)
poetry add yacm

# Using pip
pip install yacm
```

## Quick Start

### Basic Configuration Class

```python
from yacm import ConfigMixin, register_to_config

class ModelConfig(ConfigMixin):
    config_name = "model_config.json"

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        learning_rate: float = 1e-4,
        dropout: float = 0.1,
        use_bias: bool = True,
        activation: str = "gelu"
    ):
        pass

# Create a configuration
config = ModelConfig(hidden_size=512, learning_rate=2e-4)
print(config.hidden_size)  # 512
print(config.config)       # Access the full config as a FrozenDict
```

### Save and Load Configurations

```python
# Save configuration
config.save_config("./configs")

# Load configuration
config_dict = ModelConfig.load_config("./configs")
new_config = ModelConfig.from_config(config_dict)

# Or load directly from file
config_dict = ModelConfig.load_config("./configs/model_config.json")
```

### CLI Integration

```python
import argparse
from yacm import add_argparse_arguments, parse_config_from_args

# Method 1: Add to existing parser
parser = argparse.ArgumentParser()
add_argparse_arguments(parser, ModelConfig)
args = parser.parse_args()
config = config_from_args(ModelConfig, args)

# Method 2: Convenience function
config = parse_config_from_args(
    ModelConfig,
    args=["--hidden-size", "1024", "--learning-rate", "1e-3"]
)
```

### Command Line Usage

Your configuration classes automatically generate CLI arguments:

```bash
python your_script.py --hidden-size 1024 --num-layers 16 --learning-rate 1e-3 --no-use-bias
```

Boolean parameters create smart flags:
- `use_bias: bool = True` → `--no-use-bias` flag to set to False
- `use_bias: bool = False` → `--use-bias` flag to set to True

## Advanced Usage

### Multiple Configuration Classes

```python
class ModelConfig(ConfigMixin):
    config_name = "model_config.json"

    @register_to_config
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        pass

class DataConfig(ConfigMixin):
    config_name = "data_config.json"

    @register_to_config
    def __init__(self, batch_size: int = 32, sequence_length: int = 128):
        pass

# Use prefixes to avoid conflicts
parser = argparse.ArgumentParser()
add_argparse_arguments(parser, ModelConfig, prefix="model-")
add_argparse_arguments(parser, DataConfig, prefix="data-")

# CLI: --model-hidden-size 1024 --data-batch-size 64
```

### Ignoring Parameters

```python
class Config(ConfigMixin):
    config_name = "config.json"
    ignore_for_config = ["verbose", "debug"]  # Won't be saved to config

    @register_to_config
    def __init__(self, hidden_size: int = 768, verbose: bool = False):
        self.verbose = verbose  # Not included in config
```

### Working with Complex Types

```python
from typing import List

class Config(ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        layers: List[int] = [512, 256, 128],
        model_path: str = "/path/to/model"
    ):
        pass

# CLI usage: --layers 1024 512 256 --model-path /new/path
```

## API Reference

### ConfigMixin

Base class for all configuration classes.

**Class Attributes:**
- `config_name`: Filename for saving configurations
- `ignore_for_config`: List of parameters to exclude from config

**Methods:**
- `register_to_config(**kwargs)`: Register parameters to config
- `save_config(directory)`: Save config to JSON file
- `load_config(path)`: Load config from file/directory
- `from_config(config_dict)`: Create instance from config dictionary

### CLI Functions

- `add_argparse_arguments(parser, config_class, prefix="", exclude=[])`: Add CLI arguments
- `config_from_args(config_class, args, exclude=[])`: Create config from parsed args
- `parse_config_from_args(config_class, args=None, ...)`: Convenience function

### Decorators

- `@register_to_config`: Automatically register `__init__` parameters to config

## Examples

Check out the `test_example.py` file for comprehensive examples showing:
- Basic configuration usage
- Save/load functionality
- CLI argument parsing
- Multiple configuration classes
- Type handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Inspiration

This project is inspired by the `ConfigMixin` class from the [🤗 Diffusers](https://github.com/huggingface/diffusers) library, simplified to work as a standalone package using only the Python standard library and basic dependencies.
