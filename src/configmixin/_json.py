import os

import orjson

option = (
    orjson.OPT_INDENT_2
    | orjson.OPT_SORT_KEYS
    | orjson.OPT_APPEND_NEWLINE
    | orjson.OPT_SERIALIZE_NUMPY
)


def default(obj):
    if isinstance(obj, os.PathLike):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
