#!/usr/bin/env python3

"""
Example demonstrating the YACM (Yet Another Config Manager) functionality.
"""

import os
import tempfile
from src.yacm import ConfigMixin, register_to_config, add_argparse_arguments, parse_config_from_args
import argparse


class ModelConfig(ConfigMixin):
    """Example model configuration class."""
    
    config_name = "model_config.json"
    ignore_for_config = ["verbose"]  # This won't be saved to config
    
    @register_to_config
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        learning_rate: float = 1e-4,
        dropout: float = 0.1,
        use_bias: bool = True,
        activation: str = "gelu",
        max_length: int = 512,
        verbose: bool = False  # This will be ignored for config
    ):
        self.verbose = verbose
        if verbose:
            print(f"Initialized {self.__class__.__name__} with config: {self.config}")


class DataConfig(ConfigMixin):
    """Example data configuration class."""
    
    config_name = "data_config.json"
    
    @register_to_config
    def __init__(
        self,
        batch_size: int = 32,
        sequence_length: int = 128,
        dataset_path: str = "/data/train",
        shuffle: bool = True,
        num_workers: int = 4
    ):
        pass


def test_basic_functionality():
    """Test basic config functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Create a config instance
    config = ModelConfig(
        hidden_size=512,
        num_layers=6,
        learning_rate=2e-4,
        verbose=True
    )
    
    print(f"Config: {config.config}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Learning rate: {config.learning_rate}")
    print()


def test_save_and_load():
    """Test saving and loading configurations."""
    print("=== Testing Save and Load ===")
    
    # Create and save config
    original_config = ModelConfig(hidden_size=256, num_layers=8, learning_rate=5e-4)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save config
        original_config.save_config(temp_dir)
        config_path = os.path.join(temp_dir, "model_config.json")
        
        print(f"Config saved to: {config_path}")
        
        # Load config dict
        loaded_config_dict = ModelConfig.load_config(temp_dir)
        print(f"Loaded config dict: {loaded_config_dict}")
        
        # Create new instance from config
        new_config = ModelConfig.from_config(loaded_config_dict)
        print(f"New config: {new_config.config}")
        
        # Verify they match
        assert new_config.hidden_size == original_config.hidden_size
        assert new_config.num_layers == original_config.num_layers
        assert new_config.learning_rate == original_config.learning_rate
        print("✓ Save and load test passed!")
    print()


def test_cli_functionality():
    """Test CLI argument parsing."""
    print("=== Testing CLI Functionality ===")
    
    # Test manual argument parser creation
    parser = argparse.ArgumentParser(description="Test CLI")
    add_argparse_arguments(parser, ModelConfig)
    
    # Simulate command line arguments
    test_args = [
        "--hidden-size", "1024",
        "--num-layers", "16", 
        "--learning-rate", "1e-3",
        "--no-use-bias",  # This should set use_bias to False
        "--activation", "relu"
    ]
    
    args = parser.parse_args(test_args)
    print(f"Parsed args: {args}")
    
    # Create config from args
    from src.yacm import config_from_args
    config = config_from_args(ModelConfig, args)
    
    print(f"Config from CLI: {config.config}")
    assert config.hidden_size == 1024
    assert config.num_layers == 16
    assert config.learning_rate == 1e-3
    assert config.use_bias == False  # Should be False due to --no-use-bias
    assert config.activation == "relu"
    print("✓ CLI test passed!")
    print()


def test_convenience_function():
    """Test the convenience function for parsing."""
    print("=== Testing Convenience Function ===")
    
    test_args = [
        "--hidden-size", "2048",
        "--dropout", "0.2",
        "--activation", "swish"
    ]
    
    config = parse_config_from_args(ModelConfig, args=test_args)
    print(f"Config from convenience function: {config.config}")
    
    assert config.hidden_size == 2048
    assert config.dropout == 0.2
    assert config.activation == "swish"
    print("✓ Convenience function test passed!")
    print()


def test_multiple_configs():
    """Test working with multiple config classes."""
    print("=== Testing Multiple Configs ===")
    
    # Create configs
    model_config = ModelConfig(hidden_size=512, num_layers=8)
    data_config = DataConfig(batch_size=64, sequence_length=256)
    
    print(f"Model config: {model_config.config}")
    print(f"Data config: {data_config.config}")
    
    # Test CLI with multiple configs using prefixes
    parser = argparse.ArgumentParser(description="Multiple configs test")
    add_argparse_arguments(parser, ModelConfig, prefix="model-")
    add_argparse_arguments(parser, DataConfig, prefix="data-")
    
    test_args = [
        "--model-hidden-size", "1024",
        "--model-num-layers", "12",
        "--data-batch-size", "128",
        "--data-sequence-length", "512"
    ]
    
    args = parser.parse_args(test_args)
    print(f"Combined parsed args: {args}")
    
    from src.yacm import config_from_args
    model_config_from_cli = config_from_args(ModelConfig, args)
    data_config_from_cli = config_from_args(DataConfig, args)
    
    print(f"Model config from CLI: {model_config_from_cli.config}")
    print(f"Data config from CLI: {data_config_from_cli.config}")
    
    assert model_config_from_cli.hidden_size == 1024
    assert data_config_from_cli.batch_size == 128
    print("✓ Multiple configs test passed!")
    print()


if __name__ == "__main__":
    print("YACM (Yet Another Config Manager) - Example and Test")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_save_and_load()
        test_cli_functionality()
        test_convenience_function()
        test_multiple_configs()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 