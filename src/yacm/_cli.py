import argparse
import inspect
from typing import Any, Dict, Optional, Type, Union
from ._core import ConfigMixin


def add_argparse_arguments(
    parser: argparse.ArgumentParser,
    config_class: Type[ConfigMixin],
    prefix: str = "",
    exclude: Optional[list] = None,
) -> argparse.ArgumentParser:
    """
    Add arguments to an ArgumentParser based on a ConfigMixin class.
    
    Args:
        parser: The ArgumentParser to add arguments to.
        config_class: The ConfigMixin subclass to extract arguments from.
        prefix: Optional prefix to add to argument names (e.g., "--model-").
        exclude: List of parameter names to exclude from the arguments.
    
    Returns:
        The ArgumentParser with added arguments.
    """
    if exclude is None:
        exclude = []
    
    # Get the signature of the __init__ method
    signature = inspect.signature(config_class.__init__)
    
    # Get parameters, excluding 'self' and items in ignore_for_config
    ignore_list = getattr(config_class, 'ignore_for_config', []) + exclude + ['self', 'kwargs']
    
    for param_name, param in signature.parameters.items():
        if param_name in ignore_list:
            continue
        
        # Create argument name with optional prefix
        arg_name = f"--{prefix}{param_name.replace('_', '-')}" if prefix else f"--{param_name.replace('_', '-')}"
        
        # Determine the argument type and default value
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
        default_value = param.default if param.default != inspect.Parameter.empty else None
        
        # Configure argument based on type
        kwargs = {}
        
        # Handle boolean parameters
        if param_type == bool or isinstance(default_value, bool):
            if default_value is True:
                # If default is True, add a --no-prefix flag to set it to False
                no_arg_name = f"--no-{prefix}{param_name.replace('_', '-')}" if prefix else f"--no-{param_name.replace('_', '-')}"
                parser.add_argument(
                    no_arg_name,
                    dest=param_name,
                    action='store_false',
                    help=f"Disable {param_name} (default: {default_value})"
                )
            else:
                # If default is False or None, add a normal flag
                parser.add_argument(
                    arg_name,
                    dest=param_name,
                    action='store_true',
                    help=f"Enable {param_name} (default: {default_value})"
                )
        else:
            # Handle other types
            if param_type == int or isinstance(default_value, int):
                kwargs['type'] = int
            elif param_type == float or isinstance(default_value, float):
                kwargs['type'] = float
            elif param_type == str or isinstance(default_value, str):
                kwargs['type'] = str
            elif param_type is not None and hasattr(param_type, '__origin__') and param_type.__origin__ == list:
                kwargs['nargs'] = '+'
                # Try to get the inner type for lists
                if hasattr(param_type, '__args__') and param_type.__args__:
                    inner_type = param_type.__args__[0]
                    if inner_type in (int, float, str):
                        kwargs['type'] = inner_type
            
            # Set default value
            if default_value is not None:
                kwargs['default'] = default_value
            
            # Add help text
            kwargs['help'] = f"{param_name} (default: {default_value})"
            
            parser.add_argument(arg_name, dest=param_name, **kwargs)
    
    return parser


def config_from_args(
    config_class: Type[ConfigMixin],
    args: argparse.Namespace,
    exclude: Optional[list] = None
) -> ConfigMixin:
    """
    Create a config instance from parsed command line arguments.
    
    Args:
        config_class: The ConfigMixin subclass to instantiate.
        args: Parsed command line arguments from ArgumentParser.
        exclude: List of parameter names to exclude when creating the config.
    
    Returns:
        An instance of the config class with values from command line arguments.
    """
    if exclude is None:
        exclude = []
    
    # Get the signature to know which parameters are expected
    signature = inspect.signature(config_class.__init__)
    ignore_list = getattr(config_class, 'ignore_for_config', []) + exclude + ['self', 'kwargs']
    
    # Extract relevant arguments
    config_kwargs = {}
    for param_name in signature.parameters.keys():
        if param_name in ignore_list:
            continue
        
        if hasattr(args, param_name):
            value = getattr(args, param_name)
            if value is not None:  # Only include non-None values
                config_kwargs[param_name] = value
    
    return config_class(**config_kwargs)


def parse_config_from_args(
    config_class: Type[ConfigMixin],
    args: Optional[list] = None,
    prefix: str = "",
    exclude: Optional[list] = None,
    description: Optional[str] = None
) -> ConfigMixin:
    """
    Convenience function to create an ArgumentParser, add config arguments, parse, and return config.
    
    Args:
        config_class: The ConfigMixin subclass to use.
        args: Optional list of arguments to parse (defaults to sys.argv).
        prefix: Optional prefix for argument names.
        exclude: List of parameter names to exclude.
        description: Description for the ArgumentParser.
    
    Returns:
        An instance of the config class with values from command line arguments.
    """
    parser = argparse.ArgumentParser(description=description or f"Arguments for {config_class.__name__}")
    parser = add_argparse_arguments(parser, config_class, prefix=prefix, exclude=exclude)
    parsed_args = parser.parse_args(args)
    return config_from_args(config_class, parsed_args, exclude=exclude)


def create_config_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    config_class: Type[ConfigMixin],
    prefix: str = "",
    exclude: Optional[list] = None,
    help_text: Optional[str] = None
) -> argparse.ArgumentParser:
    """
    Create a subparser for a specific config class.
    
    Args:
        subparsers: The subparsers object from add_subparsers().
        name: Name of the subcommand.
        config_class: The ConfigMixin subclass to use.
        prefix: Optional prefix for argument names.
        exclude: List of parameter names to exclude.
        help_text: Help text for the subcommand.
    
    Returns:
        The created subparser.
    """
    subparser = subparsers.add_parser(name, help=help_text or f"Configure {config_class.__name__}")
    add_argparse_arguments(subparser, config_class, prefix=prefix, exclude=exclude)
    return subparser
