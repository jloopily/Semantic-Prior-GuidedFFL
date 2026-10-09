"""Independent staged R1 runner; no work on the 0.5-core/2-GiB host.

Run only on the experiment machine (not the local editing environment):
    python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage prepare
    python -m src.run_r1 --config configs/r1_non_oracle_semantic_relation_v1.yaml --stage all --resume --allow-heavy
"""
from __future__ import annotations

import argparse
import os

STAGES = ('prepare', 'crops', 'geometry', 'features', 'gallery', 'calibrate', 'score', 'evaluate', 'finalize')


def main() -> None:
    parser = argparse.ArgumentParser(description='R1 train-only non-oracle semantic localization')
    parser.add_argument('--config', default='configs/r1_non_oracle_semantic_relation_v1.yaml')
    parser.add_argument('--stage', choices=(*STAGES, 'all'), required=True)
    parser.add_argument('--resume', action='store_true', help='Verify/skip committed stages; resume uncommitted stages')
    parser.add_argument('--allow-heavy', action='store_true', help='Explicit permission on an adequately provisioned host')
    args = parser.parse_args()
    # Cap numerical libraries before any numerical import.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    from src.cirlid.config import ConfigError
    from src.r1_non_oracle_semantic_relation.protocol import ProtocolError, Run
    try:
        from src.r1_non_oracle_semantic_relation.preprocessing import crops, features, geometry, resource_guard
        from src.r1_non_oracle_semantic_relation.dataset import prepare
        from src.r1_non_oracle_semantic_relation.reference import calibrate, gallery, score
        from src.r1_non_oracle_semantic_relation.evaluation import evaluate
        from src.r1_non_oracle_semantic_relation.reporting import finalize
        run = Run.open(args.config, resume=args.resume)
        stages = STAGES if args.stage == 'all' else (args.stage,)
        functions = {'prepare': prepare, 'crops': crops, 'geometry': geometry, 'features': features,
                     'gallery': gallery, 'calibrate': calibrate, 'score': score,
                     'evaluate': evaluate, 'finalize': finalize}
        # Guard all before probing or hashing assets when heavy work is requested.
        if args.stage == 'all':
            resource_guard(run, allow_heavy=args.allow_heavy)
        for name in stages:
            if args.resume and run.path(f'{name}/complete.json').exists():
                run.read_stage(name)
                print(f'[R1] {name}: verified, already complete', flush=True)
                continue
            if name not in ('prepare', 'finalize'):
                resource_guard(run, allow_heavy=args.allow_heavy)
            print(f'[R1] {name}: starting', flush=True)
            functions[name](run)
            print(f'[R1] {name}: complete', flush=True)
    except (ProtocolError, ConfigError) as exc:
        # Never overwrite an existing decision with a synthetic failure result.
        parser.exit(2, f'[R1 INVALID_PROTOCOL / safety stop] {exc}\n')
    except (ImportError, OSError, RuntimeError) as exc:
        parser.exit(2, f'[R1 execution stopped; no scientific decision] {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
