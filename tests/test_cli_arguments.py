import argparse
import pytest
from typing import List

from yacm import ConfigMixin, register_to_config


class BasicTestConfig(ConfigMixin):
    """Basic config for testing CLI arguments."""
    config_name = "basic_test_config.json"

    @register_to_config
    def __init__(
        self,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        model_name: str = "transformer",
        use_cuda: bool = True,
        debug: bool = False,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.debug = debug


class AdvancedTestConfig(ConfigMixin):
    """Advanced config for testing more complex scenarios."""
    config_name = "advanced_test_config.json"
    ignore_for_config = ["secret_key"]

    @register_to_config
    def __init__(
        self,
        layers: List[int] = None,
        dropout_rate: float = 0.1,
        secret_key: str = "hidden",
        _private_param: int = 42,
        enable_logging: bool = True,
        threshold: float = None,
    ):
        if layers is None:
            layers = [128, 64]
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.secret_key = secret_key
        self._private_param = _private_param
        self.enable_logging = enable_logging
        self.threshold = threshold


class TestAddCliArguments:
    """Test suite for add_cli_arguments functionality."""

    def test_basic_argument_addition(self):
        """Test that basic arguments are added correctly."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser)

        # Check that arguments are present
        help_text = parser.format_help()
        assert "--learning-rate" in help_text
        assert "--batch-size" in help_text
        assert "--model-name" in help_text
        assert "--no-use-cuda" in help_text  # True default gets --no- flag
        assert "--debug" in help_text  # False default gets normal flag

    def test_argument_parsing_with_defaults(self):
        """Test parsing arguments with default values."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser)

        # Parse with no arguments (should use defaults)
        args = parser.parse_args([])
        assert args.learning_rate == 0.001
        assert args.batch_size == 32
        assert args.model_name == "transformer"
        assert args.use_cuda is True
        assert args.debug is False

    def test_argument_parsing_with_values(self):
        """Test parsing arguments with provided values."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser)

        # Parse with provided arguments
        args = parser.parse_args([
            "--learning-rate", "0.01",
            "--batch-size", "64",
            "--model-name", "bert",
            "--no-use-cuda",
            "--debug"
        ])

        assert args.learning_rate == 0.01
        assert args.batch_size == 64
        assert args.model_name == "bert"
        assert args.use_cuda is False
        assert args.debug is True

    def test_prefix_functionality(self):
        """Test that prefix is applied correctly."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser, prefix="model-")

        help_text = parser.format_help()
        assert "--model-learning-rate" in help_text
        assert "--model-batch-size" in help_text
        assert "--model-model-name" in help_text
        assert "--no-model-use-cuda" in help_text
        assert "--model-debug" in help_text

    def test_exclude_functionality(self):
        """Test that exclude parameter works correctly."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser, exclude=["learning_rate", "batch_size"])

        help_text = parser.format_help()
        assert "--learning-rate" not in help_text
        assert "--batch-size" not in help_text
        assert "--model-name" in help_text
        assert "--no-use-cuda" in help_text
        assert "--debug" in help_text

    def test_ignore_for_config_respected(self):
        """Test that ignore_for_config class attribute is respected."""
        parser = argparse.ArgumentParser()
        AdvancedTestConfig.add_cli_arguments(parser)

        help_text = parser.format_help()
        assert "--secret-key" not in help_text  # Ignored by class
        assert "--private-param" not in help_text  # Private (starts with _)
        assert "--dropout-rate" in help_text
        assert "--no-enable-logging" in help_text

    def test_list_type_handling(self):
        """Test that list types are handled correctly."""
        parser = argparse.ArgumentParser()
        AdvancedTestConfig.add_cli_arguments(parser)

        # Test parsing list arguments
        args = parser.parse_args(["--layers", "256", "128", "64"])
        assert args.layers == [256, 128, 64]

    def test_none_default_handling(self):
        """Test that None defaults are handled correctly."""
        parser = argparse.ArgumentParser()
        AdvancedTestConfig.add_cli_arguments(parser)

        # Parse with no threshold argument
        args = parser.parse_args([])
        assert args.threshold is None

        # Parse with threshold argument
        args = parser.parse_args(["--threshold", "0.5"])
        assert args.threshold == 0.5

    def test_combined_prefix_and_exclude(self):
        """Test using both prefix and exclude together."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(
            parser,
            prefix="train-",
            exclude=["model_name"]
        )

        help_text = parser.format_help()
        assert "--train-learning-rate" in help_text
        assert "--train-batch-size" in help_text
        assert "--model-name" not in help_text  # Excluded
        assert "--train-model-name" not in help_text  # Excluded
        assert "--no-train-use-cuda" in help_text

    def test_return_parser(self):
        """Test that the method returns the parser object."""
        parser = argparse.ArgumentParser()
        returned_parser = BasicTestConfig.add_cli_arguments(parser)
        assert returned_parser is parser

    def test_type_inference_from_defaults(self):
        """Test that types are correctly inferred from default values."""
        class TypeInferenceConfig(ConfigMixin):
            config_name = "type_test.json"

            @register_to_config
            def __init__(self, int_param=42, float_param=3.14, str_param="hello"):
                self.int_param = int_param
                self.float_param = float_param
                self.str_param = str_param

        parser = argparse.ArgumentParser()
        TypeInferenceConfig.add_cli_arguments(parser)

        # Test that types are correctly applied
        args = parser.parse_args(["--int-param", "100", "--float-param", "2.71", "--str-param", "world"])
        assert args.int_param == 100
        assert args.float_param == 2.71
        assert args.str_param == "world"

    def test_boolean_flag_variations(self):
        """Test different boolean flag scenarios."""
        class BooleanConfig(ConfigMixin):
            config_name = "bool_test.json"

            @register_to_config
            def __init__(self, flag_true: bool = True, flag_false: bool = False, flag_none=None):
                self.flag_true = flag_true
                self.flag_false = flag_false
                self.flag_none = flag_none

        parser = argparse.ArgumentParser()
        BooleanConfig.add_cli_arguments(parser)

        help_text = parser.format_help()
        assert "--no-flag-true" in help_text  # True default gets --no- flag
        assert "--flag-false" in help_text  # False default gets normal flag
        assert "--flag-none" in help_text  # None treated as normal argument

    def test_empty_class_handling(self):
        """Test handling of class with no parameters."""
        class EmptyConfig(ConfigMixin):
            config_name = "empty.json"

            @register_to_config
            def __init__(self):
                pass

        parser = argparse.ArgumentParser()
        EmptyConfig.add_cli_arguments(parser)

        # Should not crash and should parse empty args
        args = parser.parse_args([])
        assert args is not None

    def test_kwargs_parameter_ignored(self):
        """Test that **kwargs parameters are ignored."""
        class KwargsConfig(ConfigMixin):
            config_name = "kwargs.json"

            @register_to_config
            def __init__(self, param1: int = 1, **kwargs):
                self.param1 = param1

        parser = argparse.ArgumentParser()
        KwargsConfig.add_cli_arguments(parser)

        help_text = parser.format_help()
        assert "--param1" in help_text
        assert "--kwargs" not in help_text


# Integration tests with actual argument parsing
class TestCliArgumentsIntegration:
    """Integration tests for CLI arguments with ConfigMixin."""

    def test_full_workflow_with_parsing(self):
        """Test the complete workflow from CLI arguments to config creation."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser)

        # Simulate command line arguments
        test_args = [
            "--learning-rate", "0.05",
            "--batch-size", "128",
            "--model-name", "gpt",
            "--debug"
        ]

        args = parser.parse_args(test_args)

        # Create config using parsed arguments
        config = BasicTestConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            model_name=args.model_name,
            use_cuda=args.use_cuda,
            debug=args.debug
        )

        # Verify config values
        assert config.learning_rate == 0.05
        assert config.batch_size == 128
        assert config.model_name == "gpt"
        assert config.use_cuda is True  # Default value
        assert config.debug is True

    def test_error_handling_with_invalid_types(self):
        """Test error handling when providing invalid argument types."""
        parser = argparse.ArgumentParser()
        BasicTestConfig.add_cli_arguments(parser)

        # Test invalid integer
        with pytest.raises(SystemExit):
            parser.parse_args(["--batch-size", "not_a_number"])

        # Test invalid float
        with pytest.raises(SystemExit):
            parser.parse_args(["--learning-rate", "not_a_float"])
