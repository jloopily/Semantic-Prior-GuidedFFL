"""Regression checks for the R0 freeze, feature identity, and AP semantics."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.cirlid.config import Config, ConfigError, config_digest
from src.cirlid.features import feature_artifact_path, save_features
from src.cirlid.io import sha256_file
from src.r0_semantic_relation.metrics import average_precision
from src.r0_semantic_relation.protocol import R0_FIXED_VALUES, require_r0_protocol, r0_source_hashes
from src.run_r0 import R0ProvenanceError, _assert_frozen, _context_derangement, _full_video_coverage, _load_verified_feature


def frozen_config() -> Config:
    data = {}
    for key, value in R0_FIXED_VALUES.items():
        node = data
        for part in key.split('.')[:-1]:
            node = node.setdefault(part, {})
        node[key.split('.')[-1]] = value
    data['protocol'].update(seed=2, bootstrap_seed=3)
    data['holdout']['seed'] = 4
    data['paths'] = dict(ffpp_root='data', split_files={'train': 'train.json'},
                         dinov3_repo='repo', dinov3_weights='model.pth', crops='crops',
                         masks='masks', features='features', output_dir='results')
    data['mediapipe'].update(asset_path='asset.task', asset_sha256='1' * 64)
    return Config(data)


class RepairTests(unittest.TestCase):
    def test_r0_freeze_rejects_protocol_drift_and_unset_keys(self):
        c = frozen_config()
        require_r0_protocol(c)
        c.data['protocol']['split'] = 'test'
        with self.assertRaises(ConfigError):
            require_r0_protocol(c)
        c.data['protocol']['split'] = 'train'
        c.data['quality']['min_evaluable_frames_per_video'] = 2
        with self.assertRaises(ConfigError):
            require_r0_protocol(c)

    def test_source_manifest_includes_runner_and_scoring_and_is_stable(self):
        hashes = r0_source_hashes()
        self.assertIn('src/run_r0.py', hashes)
        self.assertIn('src/r0_semantic_relation/scoring.py', hashes)
        self.assertEqual(hashes, r0_source_hashes())

    def test_freeze_rejects_source_and_checkpoint_mutation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            c = frozen_config()
            c = Config(c.data, root / 'config.yaml')
            (root / 'model.pth').write_bytes(b'original')
            (root / 'asset.task').write_bytes(b'asset')
            receipt = {'status': 'frozen', 'config_digest': config_digest(c),
                       'source_sha256': r0_source_hashes(),
                       'dinov3_checkpoint_sha256': sha256_file(root / 'model.pth'),
                       'mediapipe_asset_sha256': sha256_file(root / 'asset.task')}
            (root / 'freeze_receipt.json').write_text(json.dumps(receipt))
            with patch('src.run_r0._verify_mediapipe_asset', return_value=(root / 'asset.task', receipt['mediapipe_asset_sha256'])):
                _assert_frozen(c, root)
                receipt['source_sha256']['src/run_r0.py'] = 'bad'
                (root / 'freeze_receipt.json').write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'source files changed'):
                    _assert_frozen(c, root)
                receipt['source_sha256'] = r0_source_hashes()
                (root / 'freeze_receipt.json').write_text(json.dumps(receipt))
                (root / 'model.pth').write_bytes(b'mutated')
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    _assert_frozen(c, root)

    def test_context_shuffle_is_seeded_and_has_no_fixed_positions(self):
        first = _context_derangement(20260930)
        self.assertTrue(np.array_equal(first, _context_derangement(20260930)))
        self.assertFalse(np.any(first == np.arange(784)))
        self.assertTrue(np.array_equal(np.sort(first), np.arange(784)))
        self.assertFalse(np.array_equal(first, _context_derangement(20260931)))

    def test_feature_identity_rejects_checkpoint_change_and_crop_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / 'crop.npy'
            crop.write_bytes(b'one')
            c = Config({'features': {'expected_tokens_per_frame': 784, 'expected_dimension': 768}})
            row = {'sample_id': 'sample', 'crop_path': str(crop)}
            artifact = feature_artifact_path(root, 'sample')
            save_features(artifact, np.ones((1, 784, 768), np.float32),
                          metadata={'sample_id': 'sample', 'crop_path': str(crop),
                                    'crop_sha256': sha256_file(crop), 'r0_feature_role': 'paired_frame',
                                    'checkpoint_sha256': 'hash-A'}, config=c)
            self.assertEqual(_load_verified_feature(row, root, c, 'hash-A').shape, (784, 768))
            with self.assertRaises(R0ProvenanceError):
                _load_verified_feature(row, root, c, 'hash-B')
            crop.write_bytes(b'two')
            with self.assertRaises(R0ProvenanceError):
                _load_verified_feature(row, root, c, 'hash-A')

    def test_features_stage_uses_official_loader_signature_and_binds_checkpoint(self):
        from types import SimpleNamespace
        from src.run_r0 import features
        with TemporaryDirectory() as directory:
            root = Path(directory)
            c = frozen_config()
            c = Config(c.data, root / 'config.yaml')
            crop = root / 'sample_crop.npy'
            np.save(crop, np.zeros((448, 448, 3), dtype=np.uint8))
            row = dict(sample_id='sample', is_real=True, frame_count=1, frame_index=0,
                       frame_fraction=.5, crop_path=str(crop), crop_source_frame=0, exclusion_reason=None)
            (root / 'freeze_receipt.json').write_text(json.dumps({'dinov3_checkpoint_sha256': 'hash-A'}))
            spec = SimpleNamespace(expected_tokens=784, expected_dimension=768)
            with patch('src.run_r0.ctx', return_value=(c, root)), \
                 patch('src.run_r0._assert_frozen'), \
                 patch('src.run_r0.read_manifest', return_value=[row]), \
                 patch('src.run_r0.spec_from_config', return_value=spec), \
                 patch('src.run_r0.load_official_model', return_value=object()) as load_model, \
                 patch('src.run_r0.preprocess_rgb', return_value=object()), \
                 patch('src.run_r0.extract_patch_tokens', return_value=np.zeros((1, 784, 768), np.float32)):
                self.assertEqual(features(root / 'config.yaml')['features'], 1)
            load_model.assert_called_once_with((root / 'repo').resolve(), (root / 'model.pth').resolve(), spec, 'cuda')
            artifact = feature_artifact_path(root / 'features', 'sample')
            self.assertEqual(_load_verified_feature(row, root / 'features', c, 'hash-A').shape, (784, 768))

    def test_video_requires_every_sampling_fraction_and_all_geometry_frames(self):
        c = frozen_config()
        frames = [{'frame_fraction': fraction} for fraction in (.25, .5, .75)]
        self.assertTrue(_full_video_coverage(frames, frames, c))
        self.assertFalse(_full_video_coverage(frames, frames[:2], c))
        duplicated = [{'frame_fraction': fraction} for fraction in (.25, .5, .5)]
        self.assertFalse(_full_video_coverage(duplicated, duplicated, c))

    def test_zero_mask_frame_undefined_but_video_pooled_ap_defined(self):
        zero = np.zeros(4, dtype=np.uint8)
        positive = np.array([0, 1, 0, 0], dtype=np.uint8)
        scores = np.array([0.0, 1.0, 0.1, 0.2])
        self.assertIsNone(average_precision(scores, zero))
        pooled = average_precision(np.tile(scores, 3), np.r_[zero, positive, zero])
        self.assertIsNotNone(pooled)
        self.assertGreater(pooled, 0)


if __name__ == '__main__':
    unittest.main()
