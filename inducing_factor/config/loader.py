from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Type

import yaml

from inducing_factor.config.models import AppConfig


def _coerce_dataclass(model_type: Type, payload: dict[str, Any]):
    kwargs = {}
    for field in fields(model_type):
        value = payload.get(field.name)
        if value is None:
            continue
        if is_dataclass(field.type):
            kwargs[field.name] = _coerce_dataclass(field.type, value)
        else:
            kwargs[field.name] = value
    return model_type(**kwargs)


def load_config(config_path: str | None = None) -> AppConfig:
    if not config_path:
        return AppConfig()
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return _coerce_dataclass(AppConfig, payload)
