"""Model configs are plain dataclasses (nested ones allowed). A run's config is the dataclass
defaults, then a JSON or TOML file, then `key=value` overrides with dotted keys (`ce.lr=3e-4`)."""

import json
import tomllib
import types
import typing
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any


def read_file(path: Path) -> dict:
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text())
    return json.loads(path.read_text())


def merge(base: dict, update: dict, prefix: str = "") -> dict:
    for key, value in update.items():
        if key not in base:
            raise KeyError(f"unknown config key {prefix + key!r}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value
    return base


def parse_override(item: str) -> dict:
    """`a.b=v` -> {"a": {"b": v}}. The value is read as JSON when it parses, else kept as a string."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"override {item!r} is not key=value")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    for part in reversed(key.split(".")):
        value = {part: value}
    return value


def coerce(hint: Any, value: Any) -> Any:
    if is_dataclass(hint):
        return from_dict(hint, value)
    if hint is Path:
        return Path(value)
    if hint is float and isinstance(value, int):
        return float(value)
    if typing.get_origin(hint) is list:
        # `datasets=sst2,anli` on the command line
        items = value.split(",") if isinstance(value, str) else value
        (item_hint,) = typing.get_args(hint) or (Any,)
        return [coerce(item_hint, item) for item in items if item != ""]
    if isinstance(hint, types.UnionType) and value is not None:
        non_none = [h for h in typing.get_args(hint) if h is not type(None)]
        return coerce(non_none[0], value) if len(non_none) == 1 else value
    return value


def from_dict[C](cls: type[C], values: dict) -> C:
    hints = typing.get_type_hints(cls)
    return cls(**{f.name: coerce(hints[f.name], values[f.name]) for f in fields(cls) if f.name in values})


def load_config[C](cls: type[C], base: dict | None = None, path: Path | None = None, overrides: list[str] = ()) -> C:
    """`base` (for example a previous run's config) replaces the defaults it names."""
    values = asdict(cls())
    if base:
        merge(values, base)
    if path:
        merge(values, read_file(path))
    for item in overrides:
        merge(values, parse_override(item))
    return from_dict(cls, values)


def dump(config: Any) -> str:
    return json.dumps(asdict(config), indent=2, default=str)
