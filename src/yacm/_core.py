import functools
import inspect
import json
import pathlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from os import PathLike


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__frozen = True
        for key, value in self.items():
            setattr(self, key, value)

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use `__delitem__` on a {self.__class__.__name__} instance.")
    
    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use `setdefault` on a {self.__class__.__name__} instance.")
    
    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use `pop` on a {self.__class__.__name__} instance.")
    
    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use `update` on a {self.__class__.__name__} instance.")
    
    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use `__setattr__` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)
    
    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use `__setitem__` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)


class ConfigMixin:
    r"""Mixin class for automated configuration registration and IO.

    Attributes
    ----------
    config_name : str, default=None
        Class attribute that specifies the filename under which the config should be stored when calling
        `save_config`. Should be overridden by the subclass.
    ignore_for_config : list[str], default=[]
        Class attribute that specifies a list of attributes that should not be saved in the config. Should
        be overridden by the subclass.

    Examples
    --------
    In this example, we have a model with 3 arguments:

    - ``hidden_size``: The hidden size of the model.
    - ``_num_layers``: The number of layers in the model.
    - ``dropout``: The dropout rate of the model.

    Among the three arguments, the number of layers is implicitly ignored by the decorator because of the leading
    underscore; the ``dropout`` argument is explicitly based on the specification in ``ignore_for_config`` class
    variable. The ``hidden_size`` argument is registered to the config.

    >>> class MyModel(nn.Module, ConfigMixin):
    ...     config_name = "my_model_config.json"
    ...     ignore_for_config = ["dropout"]
    ...
    ...     @register_to_config
    ...     def __init__(self, hidden_size: int = 768, _num_layers: int = 12, dropout: float = 0.1):
    ...         super().__init__()
    ...         self.hidden_size = hidden_size
    ...         self.num_layers = _num_layers
    ...         self.dropout = dropout  # This will be ignored because of the specification in `ignore_for_config`
    ...
    >>> model = MyModel(hidden_size=1024, num_layers=20, dropout=0.2)
    >>> model.config
    {'hidden_size': 1024}
    """

    config_name = None
    ignore_for_config = []

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __getattr__(self, name: str):
        r"""Create a shortcut to access the config attributes."""

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            return self._internal_dict[name]

        msg = f"`{type(self).__name__}` object has no attribute `{name}`"
        raise AttributeError(msg)

    @property
    def config(self) -> dict[str, Any]:
        r"""Returns the config of the class as a frozen dictionary.

        Returns
        -------
        dict[str, Any]
            The config of the class as a frozen dictionary. This is a shortcut to access the config
            attributes of the class.
        """
        return self._internal_dict

    def register_to_config(self, **kwargs):
        r"""Register keyword arguments to the configuration.

        There are two ways to register keyword arguments to the configuration:

        - By explicitly calling `register_to_config` in the `__init__` method of the subclass.
        - By using the `@register_to_config` decorator (for the `__init__` method of the subclass).

        It is recommended to use the `@register_to_config` decorator to register keyword arguments
        to automatically register keyword arguments to the configuration.

        Note that, multiple calls to `register_to_config` will raise an error to prevent updating
        the config after the class has been instantiated since it may cause unexpected inconsistencies
        between the config and the class attributes.

        Please refer to the documentation of `register_to_config` decorator for usage examples.
        """
        if self.config_name is None:
            msg = f"Make sure that {self.__class__} has defined a class attribute `config_name`."
            raise NotImplementedError(msg)

        if hasattr(self, "_internal_dict"):
            msg = (
                "`_internal_dict` is already set. Please do not call `register_to_config` again "
                "to prevent unexpected inconsistencies between the config and the class attributes."
            )
            raise RuntimeError(msg)

        self._internal_dict = FrozenDict(kwargs)

    def save_config(self, save_directory: str | PathLike, overwrite: bool = False):
        r"""Save a configuration object to the directory specified in ``save_directory``.

        The configuration is saved as a JSON file named as ``self.config_name`` in the directory specified
        in ``save_directory``.

        It is recommended to save the configuration in the same directory as the main
        objects, e.g., a model checkpoint, or other metadata files.

        Parameters
        ----------
        save_directory : str or PathLike
            Directory where the configuration JSON file, named as ``self.config_name``, is saved.
        overwrite : bool, default=False
            Whether to overwrite the configuration file if it already exists.
        """
        if self.config_name is None:
            msg = f"Make sure that {self.__class__} has defined a class attribute `config_name`."
            raise NotImplementedError(msg)

        dest = pathlib.Path(save_directory)
        if dest.is_file():
            msg = f"Provided path ({save_directory}) should be a directory, not a file"
            raise AssertionError(msg)

        dest.mkdir(parents=True, exist_ok=True)
        file = dest / self.config_name
        if file.is_file() and not overwrite:
            msg = (
                f"Provided path ({save_directory}) already contains a file named {self.config_name}. "
                "Please set `overwrite=True` to overwrite the existing file."
            )
            raise FileExistsError(msg)

        with open(file, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_config(
        cls, 
        config: FrozenDict | dict[str, Any],  # noqa: F821
        return_unused_kwargs: bool = False,
        **kwargs,
    ):
        r"""
        Instantiate a Python class from a config dictionary.
        
        Args:
            config (`Dict[str, Any]`): A config dictionary from which the Python class is instantiated.
            return_unused_kwargs (`bool`, optional): Whether kwargs that are not consumed should be returned.
            kwargs: Can be used to update the configuration object and overwrite same named arguments.
        
        Returns:
            Instance of the class or tuple of (instance, unused_kwargs) if return_unused_kwargs=True.
        """
        if config is None:
            raise ValueError("Please make sure to provide a config as the first positional argument.")
        
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary.")
        
        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)
        
        # Create model instance
        model = cls(**init_dict)
        
        # Register hidden config parameters
        model.register_to_config(**hidden_dict)
        
        # Add hidden kwargs to unused_kwargs
        unused_kwargs = {**unused_kwargs, **hidden_dict}
        
        if return_unused_kwargs:
            return (model, unused_kwargs)
        else:
            return model

    @staticmethod
    def _get_init_keys(input_class):
        """Get the parameter names from a class's __init__ method."""
        return set(dict(inspect.signature(input_class.__init__).parameters).keys())

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        """
        Extract initialization dictionary from config_dict and kwargs.
        
        Returns:
            Tuple of (init_dict, unused_kwargs, hidden_dict)
        """
        # Copy original config dict
        original_dict = dict(config_dict.items())
        
        # Get expected config attributes from __init__ signature
        expected_keys = cls._get_init_keys(cls)
        expected_keys.remove("self")
        
        # Remove general kwargs if present in dict
        if "kwargs" in expected_keys:
            expected_keys.remove("kwargs")
        
        # Remove keys to be ignored
        if len(cls.ignore_for_config) > 0:
            expected_keys = expected_keys - set(cls.ignore_for_config)
        
        # Remove private attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        
        # Create keyword arguments that will be passed to __init__
        init_dict = {}
        for key in expected_keys:
            # If config param is passed to kwarg and is present in config dict
            # it should overwrite existing config dict key
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)
            
            if key in kwargs:
                # Overwrite key
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                # Use value from config dict
                init_dict[key] = config_dict.pop(key)
        
        # Give nice warning if unexpected values have been passed
        if len(config_dict) > 0:
            print(
                f"Warning: The config attributes {list(config_dict.keys())} were passed to {cls.__name__}, "
                "but are not expected and will be ignored."
            )
        
        # Give nice info if config attributes are initialized to default values
        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            print(
                f"Info: {expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )
        
        # Define unused keyword arguments
        unused_kwargs = {**config_dict, **kwargs}
        
        # Define "hidden" config parameters
        hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}
        
        return init_dict, unused_kwargs, hidden_config_dict
    
    @classmethod
    def _dict_from_json_file(cls, json_file: str | PathLike):
        """Load dictionary from JSON file."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.
        
        Returns:
            `str`: String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict = dict(config_dict)
        config_dict["_class_name"] = self.__class__.__name__

        def to_json_saveable(value):
            if isinstance(value, pathlib.Path):
                value = value.as_posix()
            elif hasattr(value, "to_dict") and callable(value.to_dict):
                value = value.to_dict()
            elif isinstance(value, list):
                value = [to_json_saveable(v) for v in value]
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}

        return json.dumps(config_dict, indent=2, sort_keys=True)


def register_to_config(init):
    r"""Decorator for the init of classes inheriting from `ConfigMixin` for auto argument-registration.

    Users should apply this decorator to the ``__init__(self, ...)`` method of the sub-class so that
    all the arguments are automatically sent to ``self.register_to_config``. To ignore a specific argument
    accepted by the init but that shouldn't be registered in the config, use the ``ignore_for_config``
    class variable. **Note that**, once decorated, all private arguments (beginning with an underscore) are
    trashed and not sent to the init!

    Examples
    --------
    In this example, we have a model with 3 arguments:

    - ``hidden_size``: The hidden size of the model.
    - ``_num_layers``: The number of layers in the model.
    - ``dropout``: The dropout rate of the model.

    Among the three arguments, the number of layers is implicitly ignored by the decorator because of the leading
    underscore; the ``dropout`` argument is explicitly based on the specification in ``ignore_for_config`` class
    variable. The ``hidden_size`` argument is registered to the config.

    >>> class MyModel(nn.Module, ConfigMixin):
    ...     config_name = "my_model_config.json"
    ...     ignore_for_config = ["dropout"]
    ...
    ...     @register_to_config
    ...     def __init__(self, hidden_size: int = 768, _num_layers: int = 12, dropout: float = 0.1):
    ...         super().__init__()
    ...         self.hidden_size = hidden_size
    ...         self.num_layers = _num_layers
    ...         self.dropout = dropout  # This will be ignored because of the specification in `ignore_for_config`
    ...
    >>> model = MyModel(hidden_size=1024, num_layers=20, dropout=0.2)
    >>> model.config
    {'hidden_size': 1024}
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        if not isinstance(self, ConfigMixin):
            msg = (
                f"`@register_to_config` was applied to {self.__class__.__name__} init method, "
                "but this class does not inherit from `ConfigMixin`."
            )
            raise RuntimeError(msg)

        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg
        
        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        
        # Take note of the parameters that were not present in the loaded config
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))
        
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)
    
    return inner_init
