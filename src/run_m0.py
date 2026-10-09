"""Independent real-only M0-v1 audit; never dispatches fake/test scoring."""
from __future__ import annotations

import argparse
import json

from .cirlid.m0 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='M0 YAML, not frozen base YAML')
    parser.add_argument('--stage', required=True, choices=(
        'prepare', 'audit', 'crops', 'features', 'baseline', 'resampling',
        'compression', 'null', 'finalize'))
    args = parser.parse_args()
    print(json.dumps(run_stage(args.config, args.stage), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
