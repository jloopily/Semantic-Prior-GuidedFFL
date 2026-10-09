import pytest

from src.cirlid.manifest import normalize_record, read_manifest, validate_records, write_manifest


def _row(**updates):
    row = {
        "sample_id": "train:c23:original:000:2",
        "split": "train",
        "compression": "c23",
        "manipulation": "original",
        "is_real": True,
        "target_id": "000",
        "source_id": None,
        "video_name": "000",
        "video_path": "/data/000.mp4",
        "mask_path": None,
        "frame_count": 5,
        "frame_fraction": 0.5,
        "frame_index": 2,
        "mask_frame_index": None,
        "crop_source_video": "/data/000.mp4",
        "crop_source_frame": 2,
    }
    row.update(updates)
    return row


def test_missing_required_fields_are_reported_before_scalar_coercion():
    row = _row()
    del row["frame_count"]
    with pytest.raises(ValueError, match="missing fields.*frame_count"):
        validate_records([row])


def test_csv_style_scalars_are_normalized():
    row = _row(is_real="true", frame_count="5", frame_index="2", frame_fraction="0.5")
    validate_records([row])
    value = normalize_record(row)
    assert value["is_real"] is True
    assert value["frame_count"] == 5
    assert value["frame_fraction"] == 0.5


def test_fake_requires_mask_unless_excluded():
    row = _row(
        sample_id="validation:c23:Deepfakes:000_001:2",
        split="validation",
        manipulation="Deepfakes",
        is_real=False,
        target_id="000",
        source_id="001",
        video_name="000_001",
    )
    with pytest.raises(ValueError, match="must have a mask"):
        validate_records([row])
    row["exclusion_reason"] = "missing_mask"
    validate_records([row])


def test_frame_fraction_must_match_frame_index():
    row = _row(frame_index=1)
    with pytest.raises(ValueError, match="frame_index/frame_fraction disagree"):
        validate_records([row])


def test_crop_artifact_paths_survive_manifest_round_trip(tmp_path):
    row = _row(crop_path="/cache/crop.npy")
    path = tmp_path / "manifest.jsonl"
    write_manifest([row], path)
    loaded = read_manifest(path)
    assert loaded[0]["crop_path"] == "/cache/crop.npy"
    assert loaded[0]["mask_cache_path"] is None


def test_make_frame_records_remaps_mask_frame_by_normalized_position():
    from src.cirlid.manifest import make_frame_records

    rows = make_frame_records(
        video_name="000_001",
        split="validation",
        compression="c23",
        manipulation="Deepfakes",
        is_real=False,
        target_id="000",
        source_id="001",
        video_path="/data/000_001.mp4",
        mask_path="/data/000_001_mask.mp4",
        frame_count=10,
        fractions=[0.5],
        crop_source_video="/data/000.mp4",
        crop_source_frame=5,
        mask_frame_count=5,
    )
    assert rows[0].frame_index == 5
    assert rows[0].mask_frame_index == 2
    assert rows[0].mask_frame_remapped is True
    validate_records(rows)


def test_nonexcluded_fake_requires_mask_frame_count():
    row = _row(
        sample_id="validation:c23:Deepfakes:000_001:2",
        split="validation",
        manipulation="Deepfakes",
        is_real=False,
        target_id="000",
        source_id="001",
        video_name="000_001",
        mask_path="/data/000_001_mask.mp4",
        mask_frame_index=2,
    )
    with pytest.raises(ValueError, match="mask_frame_count"):
        validate_records([row])


def test_mask_frame_index_must_match_declared_remapping():
    row = _row(
        sample_id="validation:c23:Deepfakes:000_001:5",
        split="validation",
        manipulation="Deepfakes",
        is_real=False,
        target_id="000",
        source_id="001",
        video_name="000_001",
        mask_path="/data/000_001_mask.mp4",
        frame_count=10,
        frame_fraction=0.5,
        frame_index=5,
        mask_frame_index=1,
        mask_frame_count=5,
        mask_frame_remapped=True,
    )
    with pytest.raises(ValueError, match="mask_frame_index/frame_count disagree"):
        validate_records([row])


def test_real_row_cannot_carry_source_or_mask_metadata():
    row = _row(source_id="001")
    with pytest.raises(ValueError, match="Real manifest row"):
        validate_records([row])
