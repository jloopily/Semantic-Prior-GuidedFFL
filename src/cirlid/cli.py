"""Shared CLI helpers for thin `python -m src.<command>` wrappers."""

from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

from .config import Config, load_config, require_protocol


def parser(description: str) -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=description)
    command.add_argument("--config", required=True, help="Frozen experiment YAML")
    if description == "extract_features":
        command.add_argument("--subset", required=True, choices=("gallery", "validation", "test"))
    if description == "decide_gates":
        command.add_argument("--through", required=True, choices=("A", "B", "D"))
    return command


def _parse(argv: list[str] | None, description: str) -> tuple[argparse.Namespace, Config]:
    args = parser(description).parse_args(argv)
    config = load_config(args.config)
    require_protocol(config)
    return args, config


def get_config(argv: list[str] | None, description: str) -> Config:
    """Load and validate a frozen config for callers that do not need options."""

    _, config = _parse(argv, description)
    return config


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def command_main(description: str, action: Any, argv: list[str] | None = None) -> int:
    args, config = _parse(argv, description)
    kwargs = {name: getattr(args, name) for name in ("subset", "through") if hasattr(args, name)}
    parameters = inspect.signature(action).parameters
    accepted = {name: value for name, value in kwargs.items() if name in parameters}
    result = action(config, **accepted)
    if result is not None:
        print_json(result)
    return 0
