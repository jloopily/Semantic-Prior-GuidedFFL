"""Run the preregistered, validation-only limited M1 static localization pilot."""
from __future__ import annotations

import argparse
import json

from .cirlid.m1 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='M1 frozen YAML configuration')
    parser.add_argument('--stage', required=True, choices=(
        'prepare', 'crops', 'features', 'scores', 'calibrate', 'evaluate',
        'bootstrap', 'diagnostics', 'finalize'))
    args = parser.parse_args()
    result = run_stage(args.config, args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
