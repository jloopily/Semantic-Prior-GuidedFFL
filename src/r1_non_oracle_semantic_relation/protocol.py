"""R1 safety checks and append-only run provenance (no dependency execution here)."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
from typing import Iterator

from src.cirlid.config import Config, ConfigError, config_digest, load_config, resolve_config_path
from src.cirlid.io import read_json, sha256_file, write_json, write_jsonl
from . import SEMANTIC_CONTROL, methods_for


class ProtocolError(ValueError):
    """An integrity/split violation, not a sample-level detection failure."""


FIXED = {
    'protocol.split': 'train', 'protocol.manipulation': 'Face2Face',
    'protocol.compressions': ['c23', 'c40'],
    'protocol.official_test_locked': True, 'protocol.oracle_paired_target_real': False,
    'protocol.deployable_detector': False,
    'input.size': 448, 'input.color': 'RGB',
    'input.imagenet_mean': [0.485, 0.456, 0.406],
    'input.imagenet_std': [0.229, 0.224, 0.225], 'input.test_time_augmentation': False,
    'face_crop.detector': 'dlib_hog',
    'face_crop.detector_source': 'each_input_frame_independently',
    'face_crop.upsample_primary': 1, 'face_crop.upsample_retry': 2,
    'face_crop.square_scale': 1.3, 'face_crop.padding': 'reflection',
    'face_crop.image_resize': 'bilinear_antialias', 'face_crop.mask_resize': 'nearest',
    'face_crop.mask_binarization': 'nonzero',
    'backbone.implementation': 'official_facebookresearch_dinov3',
    'backbone.architecture': 'dinov3_vitb16', 'backbone.weights': 'LVD-1689M',
    'backbone.block_one_based': 6, 'backbone.api_n': [5],
    'backbone.norm': True, 'backbone.reshape': False, 'backbone.frozen': True,
    'backbone.extra_l2_normalization': False, 'backbone.feature_dtype_on_disk': 'float32',
    'features.expected_patch_grid': [28, 28],
    'features.expected_tokens_per_frame': 784, 'features.expected_dimension': 768,
    'sampling.frame_fractions': [0.25, 0.50, 0.75],
    'gallery.source': 'original_train_only',
    'gallery.correspondence': 'same_normalized_independent_crop_token_index',
    'gallery.aggregation': 'mean_frame_distances_then_equal_video_mean',
    'gallery.missing_policy': 'record_no_replacement',
    'calibration.source': 'component_disjoint_train_real_c23',
    'calibration.quantile': 0.99, 'calibration.numpy_method': 'linear',
    'calibration.transfer_to_c40': True,
    'mediapipe.num_faces': 2, 'mediapipe.multi_face': 'fail_no_fallback',
    'mediapipe.min_face_detection_confidence': 0.5,
    'mediapipe.min_face_presence_confidence': 0.5,
    'mediapipe.min_tracking_confidence': 0.5,
    'regions.perioral_ring_radius': 12,
    'regions.geometry_source': 'each_input_frame_independently',
    'regions.empty_region': 'record_failure_do_not_substitute_full_face',
    'statistics.primary_domain': 'full_face_valid_domain',
    'statistics.secondary_domains': ['mouth_core', 'perioral_ring', 'cheek_left', 'cheek_right'],
    'statistics.grouping': 'source_video_connected_component',
    'statistics.bootstrap_replicates': 2000,
    'statistics.paired_real_comparison': 'warp_independent_real_score_to_fake_source_coordinates',
}


def validate(config: Config) -> None:
    methods = methods_for(config)
    problems = [f'{key}: expected {value!r}, got {config.get(key)!r}'
                for key, value in FIXED.items()
                if type(config.get(key)) is not type(value) or config.get(key) != value]
    for key in ('protocol.seed', 'protocol.bootstrap_seed', 'holdout.seed', 'calibration.seed'):
        if type(config.get(key)) is not int or config.get(key) < 0:
            problems.append(f'{key}: nonnegative integer required')
    for key in ('gallery.videos', 'gallery.min_valid_videos', 'calibration.videos',
                'calibration.min_valid_videos', 'quality.min_independent_components',
                'runtime.batch_size', 'runtime.threads', 'runtime.progress_every'):
        if type(config.get(key)) is not int or config.get(key) < 1:
            problems.append(f'{key}: positive integer required')
    seeds = config.get('gallery.seeds')
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(v) is not int or v < 0 for v in seeds) or len(set(seeds)) != len(seeds)):
        problems.append('gallery.seeds: at least two distinct nonnegative integers required')
    for key in ('gallery.min_valid_frames_per_video', 'quality.min_evaluable_frames_per_video'):
        if config.get(key) != 3 or type(config.get(key)) is not int:
            problems.append(f'{key}: this three-frame pilot requires 3')
    for key in ('holdout.fraction', 'quality.min_evaluable_frame_fraction'):
        value = config.get(key)
        if type(value) not in (float, int) or not 0 < value < 1:
            problems.append(f'{key}: a fraction in (0,1) required')
    margin = config.get('statistics.practical_margin_ap')
    if type(margin) not in (float, int) or not 0 < margin < 1:
        problems.append('statistics.practical_margin_ap: a fraction in (0,1) required')
    for key in ('runtime.minimum_cpu_cores', 'runtime.minimum_memory_gib'):
        value = config.get(key)
        if type(value) not in (float, int) or value < (2 if key.endswith('cores') else 4):
            problems.append(f'{key}: cannot lower the low-resource execution safeguard')
    if config.get('runtime.device') not in ('cuda', 'cpu'):
        problems.append('runtime.device: cuda or cpu required')
    for key in ('paths.ffpp_root', 'paths.split_files.train', 'paths.output_dir',
                'paths.dinov3_repo', 'paths.dinov3_weights', 'mediapipe.asset_path',
                'mediapipe.asset_sha256'):
        if not isinstance(config.get(key), str) or not config.get(key):
            problems.append(f'{key}: nonempty path/hash required')
    if problems:
        raise ConfigError('R1 protocol mismatch:\n' + '\n'.join(problems))
    if config.require('gallery.min_valid_videos') > config.require('gallery.videos'):
        raise ConfigError('minimum valid gallery videos exceeds selected gallery size')
    if config.require('calibration.min_valid_videos') > config.require('calibration.videos'):
        raise ConfigError('minimum valid calibration videos exceeds selected calibration size')
    split_path = resolve_config_path(config, config.require('paths.split_files.train'))
    if split_path.name != 'train.json':
        raise ConfigError('R1 requires the authorized train.json, never validation/test split files')
    root = resolve_config_path(config, config.require('paths.output_dir')).resolve()
    if not root.name.startswith('r1_non_oracle_') or root.is_symlink():
        raise ConfigError('R1 output must be a dedicated r1_non_oracle_* directory')
    if SEMANTIC_CONTROL in methods and not root.name.startswith(
            'r1_non_oracle_semantic_relation_semantic_control_'):
        raise ConfigError('semantic-control supplement requires its own output namespace')
    project = Path(__file__).resolve().parents[2]
    for protected in (project / 'src', project / 'configs',
                      resolve_config_path(config, config.require('paths.ffpp_root'))):
        if root == protected.resolve() or protected.resolve() in root.parents:
            raise ConfigError(f'R1 output cannot be inside protected inputs: {protected}')


def source_hashes() -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    own = [project / 'src/run_r1.py', *sorted((project / 'src/r1_non_oracle_semantic_relation').glob('*.py'))]
    shared = ['config', 'io', 'ffpp', 'geometry', 'types', 'crops', 'dinov3', 'features', 'manifest']
    own += [project / f'src/cirlid/{name}.py' for name in shared]
    own += [project / f'src/r0_semantic_relation/{name}.py'
            for name in ('geometry', 'scoring', 'metrics', 'bootstrap', 'mediapipe_adapter')]
    return {str(path.relative_to(project)).replace('\\', '/'): sha256_file(path) for path in own}


def environment_versions() -> dict[str, str]:
    versions = {'python': platform.python_version(), 'platform': platform.platform()}
    for package in ('numpy', 'PyYAML', 'torch', 'torchvision', 'dlib',
                    'mediapipe', 'opencv-python', 'opencv-python-headless'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = 'not_installed_as_distribution'
    return versions


def key(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@dataclass
class Run:
    config: Config
    root: Path
    resume: bool = False

    @property
    def methods(self) -> tuple[str, ...]:
        return methods_for(self.config)

    @classmethod
    def open(cls, path: str | Path, *, resume: bool = False) -> 'Run':
        config = load_config(path)
        validate(config)
        return cls(config, resolve_config_path(config, config.require('paths.output_dir')).resolve(), resume)

    def path(self, name: str) -> Path:
        destination = (self.root / name).resolve()
        if self.root not in destination.parents:
            raise ProtocolError('artifact path escaped R1 output root')
        return destination

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def freeze(self, payload: dict) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            raise ProtocolError('prepare requires an empty/new output directory; existing runs are immutable')
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.path('freeze_receipt.json'), {
            **payload, 'config': self.config.data, 'config_digest': config_digest(self.config),
            'source_hashes': source_hashes(), 'git_commit': None,
            'methods': list(self.methods),
            'semantic_control_origin': self.config.get('scoring.semantic_control_origin'),
            'environment_versions': environment_versions(),
            'git_commit_status': 'unavailable_not_inferred', 'official_test_locked': True,
            'paired_real_used_for_gallery_or_fake_scoring': False,
            'mask_access_before_evaluate': False,
            'coverage_threshold_origin': self.config.get('quality.coverage_origin'),
            'practical_margin_origin': self.config.get('statistics.practical_margin_origin'),
        })

    def frozen(self) -> dict:
        receipt = read_json(self.path('freeze_receipt.json'))
        if receipt['config_digest'] != config_digest(self.config):
            raise ProtocolError('configuration changed after prepare; use a new R1 output directory')
        if receipt['source_hashes'] != source_hashes():
            raise ProtocolError('source code changed after prepare; use a new R1 output directory')
        if receipt['environment_versions'] != environment_versions():
            raise ProtocolError('runtime package versions changed after prepare; use a new run')
        split = resolve_config_path(self.config, self.config.require('paths.split_files.train'))
        if sha256_file(split) != receipt['train_split_sha256']:
            raise ProtocolError('train split changed since freeze')
        return receipt

    def read_stage(self, name: str, _checked: set[str] | None = None,
                   _active: set[str] | None = None) -> list[dict]:
        self.frozen()
        from src.cirlid.io import read_jsonl
        checked = set() if _checked is None else _checked
        active = set() if _active is None else _active
        if name in active:
            raise ProtocolError('cyclic stage dependencies')
        if name in checked:
            return read_jsonl(self.path(f'{name}/rows.jsonl'))
        active.add(name)
        receipt = read_json(self.path(f'{name}/complete.json'))
        if receipt['freeze_sha256'] != sha256_file(self.path('freeze_receipt.json')):
            raise ProtocolError(f'{name}: freeze receipt mismatch')
        for relative, digest in receipt['files'].items():
            if sha256_file(self.path(relative)) != digest:
                raise ProtocolError(f'{name}: modified/missing artifact {relative}')
        for dependency, digest in receipt['dependencies'].items():
            self.read_stage(dependency, checked, active)
            if sha256_file(self.path(f'{dependency}/complete.json')) != digest:
                raise ProtocolError(f'{name}: upstream stage changed')
        active.remove(name)
        checked.add(name)
        return read_jsonl(self.path(f'{name}/rows.jsonl'))

    @contextmanager
    def stage(self, name: str, dependencies: tuple[str, ...]) -> Iterator['Stage']:
        self.frozen()
        for dependency in dependencies:
            self.read_stage(dependency)
        directory = self.path(name)
        if (directory / 'complete.json').exists():
            if not self.resume:
                raise ProtocolError(f'{name} already complete; --resume verifies and skips it')
            self.read_stage(name)
            yield Stage(self, name, dependencies, complete=True)
            return
        if directory.exists() and not self.resume:
            raise ProtocolError(f'{name} is incomplete; use --resume to continue without deletion')
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / 'active.lock'
        try:
            handle = lock.open('x', encoding='utf-8')
        except FileExistsError as exc:
            raise ProtocolError(f'{name}: active/stale lock {lock}; no concurrent writers allowed') from exc
        try:
            handle.write('R1 stage is active. A crashed-process stale lock requires manual inspection.\n')
            handle.close()
            yield Stage(self, name, dependencies)
        finally:
            handle.close()
            # Only remove our ephemeral lock, never a dataset/result artifact.
            lock.unlink(missing_ok=True)


@dataclass
class Stage:
    run: Run
    name: str
    dependencies: tuple[str, ...]
    complete: bool = False

    def cached(self, sample_id: str) -> dict | None:
        path = self.run.path(f'{self.name}/records/{key(sample_id)}.json')
        if not path.exists():
            return None
        value = read_json(path)
        if value['record']['sample_id'] != sample_id:
            raise ProtocolError('checkpoint identifier mismatch')
        for relative, digest in value['files'].items():
            if sha256_file(self.run.path(relative)) != digest:
                raise ProtocolError(f'checkpoint artifact changed: {relative}')
        return value['record']

    def record(self, row: dict, artifacts: list[Path] | None = None) -> dict:
        destination = self.run.path(f'{self.name}/records/{key(row["sample_id"])}.json')
        if destination.exists():
            raise ProtocolError('attempt to overwrite a completed sample checkpoint')
        write_json(destination, {
            'record': row, 'files': {self.run.relative(p): sha256_file(p) for p in artifacts or []},
        })
        return row

    def finish(self, rows: list[dict]) -> None:
        write_jsonl(self.run.path(f'{self.name}/rows.jsonl'), rows)
        directory = self.run.path(self.name)
        files = {self.run.relative(p): sha256_file(p) for p in sorted(directory.rglob('*'))
                 if p.is_file() and p.name not in ('active.lock', 'complete.json') and not p.name.endswith('.tmp')}
        write_json(directory / 'complete.json', {
            'freeze_sha256': sha256_file(self.run.path('freeze_receipt.json')),
            'dependencies': {d: sha256_file(self.run.path(f'{d}/complete.json')) for d in self.dependencies},
            'files': files, 'rows': len(rows),
        })
