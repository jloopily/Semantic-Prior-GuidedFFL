"""CLI entry point: build_gallery."""
from __future__ import annotations
import sys
from src.cirlid.cli import command_main
from src.cirlid import commands

if __name__ == "__main__":
    name = "build_gallery"
    action = getattr(commands, name + "_command", None) or getattr(commands, name, None)
    if action is None:
        action = commands.pending_experiment(name)
    raise SystemExit(command_main(name, action, sys.argv[1:]))
