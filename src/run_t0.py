"""Isolated exploratory T0 CLI. Never dispatches an official-test command."""
from __future__ import annotations

import argparse
import json

from .cirlid.t0 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='T0 YAML, not the frozen base YAML')
    parser.add_argument('--stage', required=True,
                        choices=('prepare', 'gallery', 'crops', 'features', 'scores', 'evaluate'))
    args = parser.parse_args()
    print(json.dumps(run_stage(args.config, args.stage), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
