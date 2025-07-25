from atexit import register
import functools
import inspect
import json
import pathlib
from collections import OrderedDict
from os import PathLike
from typing import Any, TypeVar

_Self = TypeVar("_Self", bound="ConfigMixin")


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        msg = f"Cannot use `__delitem__` on a {self.__class__.__name__} instance."
        raise Exception(msg)

    def __setattr__(self, name, value):
        if hasattr(self, "_FrozenDict__frozen") and self.__frozen:
            msg = f"Cannot use `__setattr__` on a {self.__class__.__name__} instance."
            raise Exception(msg)
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "_FrozenDict__frozen") and self.__frozen:
            msg = f"Cannot use `__setitem__` on a {self.__class__.__name__} instance."
            raise Exception(msg)
        super().__setitem__(name, value)

    def setdefault(self, *args, **kwargs):
        msg = f"Cannot use `setdefault` on a {self.__class__.__name__} instance."
        raise Exception(msg)

    def pop(self, *args, **kwargs):
        msg = f"Cannot use `pop` on a {self.__class__.__name__} instance."
        raise Exception(msg)

    def update(self, *args, **kwargs):
        msg = f"Cannot use `update` on a {self.__class__.__name__} instance."
        raise Exception(msg)


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

    >>> class MyModel(ConfigMixin):
    ...     config_name = "my_model_config.json"
    ...     ignore_for_config = ["dropout"]
    ...
    ...     @register_to_config
    ...     def __init__(self, hidden_size: int = 768, _num_layers: int = 12, dropout: float = 0.1):
    ...         self.hidden_size = hidden_size
    ...         self.num_layers = _num_layers
    ...         self.dropout = dropout  # This will be ignored because of the specification in `ignore_for_config`
    ...
    >>> model = MyModel(hidden_size=1024, _num_layers=20, dropout=0.2)
    >>> model.config
    FrozenDict([('_use_default_values', []), ('hidden_size', 1024)])
    >>> model.num_layers
    20
    >>> model.dropout
    0.2
    """

    # These argument names should be ignored when initializing the class from a config.
    _meta_names = ["_class_name", "_use_default_values", "_var_positional", "_var_keyword"]

    config_name = None
    ignore_for_config = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} {self.get_config_json()}"

    def __getattr__(self, name: str) -> Any:
        r"""Create a shortcut to access the config attributes."""

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(
            self.__dict__["_internal_dict"], name
        )
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            return self._internal_dict[name]

        msg = f"`{type(self).__name__}` object has no attribute `{name}`"
        raise AttributeError(msg)

    @property
    def config(self) -> FrozenDict:
        r"""Returns the config of the class as a frozen dictionary.

        Returns
        -------
        FrozenDict
            The config of the class as a frozen dictionary. This is a shortcut to access the config
            attributes of the class.
        """
        return self._internal_dict

    def register_to_config(self, **kwargs) -> None:
        r"""Register keyword arguments to the configuration.

        There are two ways to register keyword arguments to the configuration:

        - By explicitly calling `register_to_config` in the ``__init__`` method of the subclass.
        - By using the `@register_to_config` decorator (for the ``__init__`` method of the subclass).

        It is recommended to use the ``@register_to_config`` decorator to register keyword arguments
        to automatically register keyword arguments to the configuration.

        Note that, multiple calls to ``register_to_config`` will raise an error to prevent updating
        the config after the class has been instantiated since it may cause unexpected inconsistencies
        between the config and the class attributes.

        Please refer to the documentation of ``register_to_config`` decorator for usage examples.
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

    def save_config(
        self, save_directory: str | PathLike, overwrite: bool = False
    ) -> None:
        r"""Save a configuration object to the directory specified in ``save_directory``.

        The configuration is saved as a JSON file named as ``self.config_name`` in the directory
        specified in ``save_directory``.

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
            writer.write(self.get_config_json())

    @classmethod
    def from_config(
        cls: type[_Self],
        save_directory: str | PathLike = None,
        config: dict[str, Any] = None,
        runtime_kwargs: dict[str, Any] = None,
    ) -> _Self:
        r"""Instantiate the current class from a config dictionary.

        Parameters
        ----------
        save_directory : str or PathLike, default=None
            Directory where the configuration JSON file, named as ``self.config_name``, is saved.
        config : dict[str, Any], default=None
            A dictionary of the config parameters. If provided, the config will be loaded from the dictionary
            instead of the JSON file.
        runtime_kwargs : dict[str, Any], default=None
            A dictionary of the runtime kwargs. These are usually non-serializable parameters that need to be
            determined/initialized at runtime, such as the model object of a trainer class.

        Returns
        -------
        An instance of the class.
        """
        if config is None:
            if save_directory is None:
                msg = "Either `save_directory` or `config` must be provided"
                raise ValueError(msg)

            dest = pathlib.Path(save_directory)
            if dest.is_file():
                msg = f"Provided path ({save_directory}) should be a directory, not a file"
                raise AssertionError(msg)

            file = dest / cls.config_name
            if not file.is_file():
                msg = f"Provided path ({save_directory}) does not contain a file named {cls.config_name}"
                raise FileNotFoundError(msg)

            with open(file, encoding="utf-8") as reader:
                config = json.load(reader)

        if config.get("_class_name") != cls.__name__:
            msg = f"Config {cls.config_name} is not a config for {cls.__name__}."
            raise ValueError(msg)

        signature = inspect.signature(cls)
        expected_names, default_names = set(), set()
        for name, param in signature.parameters.items():
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if param.default is not param.empty:
                default_names.add(name)
                continue
            expected_names.add(name)

        missing_names = expected_names - set(config.keys())
        if missing_names:
            msg = f"Config is missing required parameter(s): {', '.join(missing_names)}"
            raise ValueError(msg)

        # Initialize with default values. However, note that, if the config file is serialized from
        # a ConfigMixin instance, theoretically, it should already contain all the default values.
        init_kwargs = {
            name: signature.parameters[name].default for name in default_names
        }
        for name, value in config.items():
            if name in cls._meta_names:
                continue
            if name not in expected_names and name not in init_kwargs:
                msg = f"Config contains unexpected parameter: {name}."
                raise ValueError(msg)
            init_kwargs[name] = value

        return cls(**init_kwargs, **(runtime_kwargs or {}))

    def get_config_json(self) -> str:
        r"""Serializes the configurations to a JSON string.

        In addition to the config parameters, the JSON string also includes a few metadata such as the class
        name and which argument values were registered from default values. Metadata will have a leading
        underscore to indicate that they are not part of the class initialization parameters.

        Returns
        -------
        str
            String containing all the attributes that make up the configuration instance in JSON format.
            Note that ignored config parameters and private attributes are not included in the JSON string.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict = dict(config_dict)

        def cast(value):
            if isinstance(value, pathlib.Path):
                value = value.as_posix()
            elif hasattr(value, "to_dict") and callable(value.to_dict):
                value = value.to_dict()
            elif isinstance(value, list):
                value = [cast(v) for v in value]
            return value

        return json.dumps(
            {k: cast(v) for k, v in config_dict.items()},
            indent=2,
            sort_keys=True,
        )


def register_to_config(init):
    r"""Decorator for the init of classes inheriting from `ConfigMixin` for auto argument-registration.

    Users should apply this decorator to the ``__init__(self, ...)`` method of the subclass so that all
    the arguments are automatically sent to ``self.register_to_config``. To ignore a specific argument
    accepted by the init but that shouldn't be registered in the config, use the ``ignore_for_config``
    class variable. **Note that**, once decorated, all private arguments (beginning with an underscore)
    are trashed and not sent to the init!

    Examples
    --------
    In this example, we have a model with 3 arguments:

    - ``hidden_size``: The hidden size of the model.
    - ``_num_layers``: The number of layers in the model.
    - ``dropout``: The dropout rate of the model.

    Among the three arguments, the number of layers is implicitly ignored by the decorator because of the leading
    underscore; the ``dropout`` argument is explicitly based on the specification in ``ignore_for_config`` class
    variable. The ``hidden_size`` argument is registered to the config.

    >>> class MyModel(ConfigMixin):
    ...     config_name = "my_model_config.json"
    ...     ignore_for_config = ["dropout"]
    ...
    ...     @register_to_config
    ...     def __init__(self, hidden_size: int = 768, _num_layers: int = 12, dropout: float = 0.1):
    ...         self.hidden_size = hidden_size
    ...         self.num_layers = _num_layers
    ...         self.dropout = dropout  # This will be ignored because of the specification in `ignore_for_config`
    ...
    >>> model = MyModel(hidden_size=1024, _num_layers=20, dropout=0.2)
    >>> model.config
    FrozenDict([('_use_default_values', []), ('hidden_size', 1024)])
    >>> model.num_layers
    20
    >>> model.dropout
    0.2
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        if not isinstance(self, ConfigMixin):
            msg = (
                f"`@register_to_config` was applied to {self.__class__.__name__} init method, "
                "but this class does not inherit from `ConfigMixin`."
            )
            raise RuntimeError(msg)

        signature = inspect.signature(self.__class__)

        ignore_for_config = set(getattr(self, "ignore_for_config", []))
        registered_kwargs = {
            "_class_name": self.__class__.__name__,
            "_use_default_values": [],
            "_var_positional": args[_num_non_var_positional(signature):],
            "_var_keyword": {
                name: param
                for name, param in kwargs.items()
                if name not in signature.parameters
            },
        }

        # Obtain the names corresponding to positional arguments.
        #
        # Note that, if the number of provided positional argument is greater than the number
        # of non-var positional arguments, while no var positional argument is present in the
        # init signature, the extra positional arguments could be incorrectly associated with
        # the var keyword arguments. But the instantiation of the class will fail anyway.
        for name, param in zip(signature.parameters.keys(), args):
            if signature.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL:
                break
            if name in ignore_for_config or name.startswith("_"):
                continue
            registered_kwargs[name] = param

        # Fill in the default values for the remaining positional arguments.
        #
        # Note that positional arguments of the init method may also be passed in as keyword
        # arguments, which will be captured by the `kwargs` argument. In this cases, default
        # values will not be used.
        for name, param in filter(
            lambda i: i[0] not in registered_kwargs,
            signature.parameters.items()
        ):
            if name in ignore_for_config or name.startswith("_"):
                continue
            if name in kwargs:
                registered_kwargs[name] = kwargs[name]
            else:
                registered_kwargs[name] = param.default
                registered_kwargs["_use_default_values"].append(name)

        getattr(self, "register_to_config")(**registered_kwargs)
        init(self, *args, **kwargs)

    return inner_init


def _num_non_var_positional(signature: inspect.Signature) -> int:
    return sum(
        param.kind is not inspect.Parameter.VAR_POSITIONAL
        for param in signature.parameters.values()
    )
