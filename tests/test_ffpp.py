import json

from src.cirlid.ffpp import directed_fake_names, read_split_pairs, split_original_ids


def test_official_pair_split_expands_ids_and_both_directions(tmp_path):
    split = tmp_path / "val.json"
    split.write_text(json.dumps([["071", "054"], ["002", "001"]]), encoding="utf-8")
    assert read_split_pairs(split) == [("071", "054"), ("002", "001")]
    assert split_original_ids(split) == ["001", "002", "054", "071"]
    assert directed_fake_names(split) == [
        ("001_002", "002", "001"),
        ("002_001", "001", "002"),
        ("054_071", "071", "054"),
        ("071_054", "054", "071"),
    ]


def test_audit_detects_split_overlap_and_reports_pair_accounting(tmp_path):
    from src.cirlid.ffpp import FFPPPaths, audit_ffpp

    root = tmp_path / "ffpp"
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    train = split_dir / "train.json"
    validation = split_dir / "val.json"
    test = split_dir / "test.json"
    train.write_text(json.dumps([["000", "001"]]), encoding="utf-8")
    validation.write_text(json.dumps([["000", "002"]]), encoding="utf-8")
    test.write_text(json.dumps([["003", "004"]]), encoding="utf-8")
    paths = FFPPPaths.from_root(
        root,
        {"train": train, "validation": validation, "test": test},
    )

    report = audit_ffpp(paths, compressions=(), manipulations=())

    assert report["status"] == "invalid_splits"
    assert report["splits"]["train"]["reused_original_id_count"] == 0
    assert any("train/validation: overlapping original IDs" in issue for issue in report["split_issues"])
