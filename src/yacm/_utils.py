import argparse
import inspect
from typing import Any, Dict, List, Optional, Type, Union


def _get_parameter_type_info(param: inspect.Parameter) -> tuple[Any, Any]:
    r"""Extract type and default value information from a parameter.

    Parameters
    ----------
    param : inspect.Parameter
        The parameter to analyze.

    Returns
    -------
    tuple[Any, Any]
        A tuple of (type_annotation, default_value).
    """
    param_type = (
        param.annotation if param.annotation != inspect.Parameter.empty else None
    )
    default_value = param.default if param.default != inspect.Parameter.empty else None
    return param_type, default_value


def _create_argument_name(param_name: str, prefix: str = "") -> str:
    r"""Create CLI argument name from parameter name and optional prefix.

    Parameters
    ----------
    param_name : str
        The parameter name.
    prefix : str, default=""
        Optional prefix to add.

    Returns
    -------
    str
        The formatted argument name with dashes instead of underscores.
    """
    formatted_name = param_name.replace("_", "-")
    return f"--{prefix}{formatted_name}" if prefix else f"--{formatted_name}"


def _configure_boolean_argument(
    parser: argparse.ArgumentParser,
    param_name: str,
    default_value: Any,
    prefix: str = "",
) -> None:
    r"""Configure boolean argument for the parser.

    For boolean parameters, this function creates appropriate flags:
    - If default is True, creates a ``--no-param`` flag that sets the value to False
    - If default is False or None, creates a normal ``--param`` flag that sets the value to True

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser to add the argument to.
    param_name : str
        The parameter name.
    default_value : Any
        The default value for the parameter.
    prefix : str, default=""
        Optional prefix for argument names.
    """
    if default_value is True:
        # If default is True, add a --no-prefix flag to set it to False
        no_arg_name = (
            f"--no-{prefix}{param_name.replace('_', '-')}"
            if prefix
            else f"--no-{param_name.replace('_', '-')}"
        )
        parser.add_argument(
            no_arg_name,
            dest=param_name,
            action="store_false",
            help=f"Disable {param_name} (default: {default_value})",
        )
    else:
        # If default is False or None, add a normal flag
        arg_name = _create_argument_name(param_name, prefix)
        parser.add_argument(
            arg_name,
            dest=param_name,
            action="store_true",
            help=f"Enable {param_name} (default: {default_value})",
        )


def _configure_typed_argument(
    parser: argparse.ArgumentParser,
    param_name: str,
    param_type: Any,
    default_value: Any,
    prefix: str = "",
) -> None:
    r"""Configure typed argument for the parser.

    This function handles type inference and configuration for non-boolean parameters.
    It supports int, float, str, and List types with automatic type conversion.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser to add the argument to.
    param_name : str
        The parameter name.
    param_type : Any
        The parameter type annotation.
    default_value : Any
        The default value for the parameter.
    prefix : str, default=""
        Optional prefix for argument names.
    """
    arg_name = _create_argument_name(param_name, prefix)
    kwargs = {}

    # Handle different types
    if param_type == int or isinstance(default_value, int):
        kwargs["type"] = int
    elif param_type == float or isinstance(default_value, float):
        kwargs["type"] = float
    elif param_type == str or isinstance(default_value, str):
        kwargs["type"] = str
    elif (
        param_type is not None
        and hasattr(param_type, "__origin__")
        and param_type.__origin__ == list
    ):
        kwargs["nargs"] = "+"
        # Try to get the inner type for lists
        if hasattr(param_type, "__args__") and param_type.__args__:
            inner_type = param_type.__args__[0]
            if inner_type in (int, float, str):
                kwargs["type"] = inner_type

    # Set default value
    if default_value is not None:
        kwargs["default"] = default_value

    # Add help text
    kwargs["help"] = f"{param_name} (default: {default_value})"

    parser.add_argument(arg_name, dest=param_name, **kwargs)


def _should_include_parameter(
    param_name: str, ignore_for_config: List[str], exclude: Optional[List[str]] = None
) -> bool:
    r"""Determine if a parameter should be included in CLI arguments.

    Parameters are excluded if they:
    - Start with an underscore (private parameters)
    - Are in the class's ignore_for_config list
    - Are in the exclude list provided to the function
    - Are standard parameters like 'self' or 'kwargs'

    Parameters
    ----------
    param_name : str
        The parameter name to check.
    ignore_for_config : List[str]
        List of parameters to ignore from the config class.
    exclude : List[str], optional
        Additional list of parameters to exclude.

    Returns
    -------
    bool
        True if the parameter should be included, False otherwise.
    """
    if exclude is None:
        exclude = []

    # Skip private parameters (starting with underscore)
    if param_name.startswith("_"):
        return False

    ignore_list = ignore_for_config + exclude + ["self", "kwargs"]
    return param_name not in ignore_list


def add_arguments_to_parser(
    parser: argparse.ArgumentParser,
    config_class: Type,
    prefix: str = "",
    exclude: Optional[List[str]] = None,
) -> argparse.ArgumentParser:
    r"""Add arguments to an ArgumentParser based on a ConfigMixin class.

    This function inspects the ``__init__`` method of the provided class and creates
    corresponding command-line arguments based on the parameter types and defaults.
    Boolean parameters are handled specially with ``--flag`` and ``--no-flag`` options.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser to add arguments to.
    config_class : Type
        The ConfigMixin subclass to extract arguments from.
    prefix : str, default=""
        Optional prefix to add to argument names (e.g., "model-").
    exclude : List[str], optional
        List of parameter names to exclude from the arguments.

    Returns
    -------
    argparse.ArgumentParser
        The ArgumentParser with added arguments.

    Examples
    --------
    >>> import argparse
    >>> from yacm import ConfigMixin, register_to_config
    >>>
    >>> class MyConfig(ConfigMixin):
    ...     config_name = "my_config.json"
    ...     @register_to_config
    ...     def __init__(self, learning_rate: float = 0.001, use_cuda: bool = True):
    ...         self.learning_rate = learning_rate
    ...         self.use_cuda = use_cuda
    ...
    >>> parser = argparse.ArgumentParser()
    >>> updated_parser = add_arguments_to_parser(parser, MyConfig)
    >>> "--learning-rate" in updated_parser.format_help()
    True
    >>> "--no-use-cuda" in updated_parser.format_help()
    True
    """
    if exclude is None:
        exclude = []

    # Get the signature of the __init__ method
    signature = inspect.signature(config_class.__init__)

    # Get parameters to ignore
    ignore_for_config = getattr(config_class, "ignore_for_config", [])

    for param_name, param in signature.parameters.items():
        if not _should_include_parameter(param_name, ignore_for_config, exclude):
            continue

        param_type, default_value = _get_parameter_type_info(param)

        # Handle boolean parameters specially
        if param_type == bool or isinstance(default_value, bool):
            _configure_boolean_argument(parser, param_name, default_value, prefix)
        else:
            _configure_typed_argument(
                parser, param_name, param_type, default_value, prefix
            )

    return parser
