"""Run the frozen validation-only M1.1 C2 deconfounding audit on the server."""
from __future__ import annotations

import argparse
import json

from .cirlid.m1_1 import run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=('prepare', 'scores', 'evaluate', 'bootstrap', 'finalize'))
    args = parser.parse_args()
    print(json.dumps(run_stage(args.config, args.stage), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
