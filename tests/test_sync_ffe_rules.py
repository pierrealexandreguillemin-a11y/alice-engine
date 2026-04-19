"""Tests pour scripts/sync_ffe_rules.py."""

from pathlib import Path

from scripts.sync_ffe_rules import compute_file_sha256, detect_drift, sync_rules


def test_sha256_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "test.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    h1 = compute_file_sha256(f)
    h2 = compute_file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64


def test_detect_drift_true_when_contents_differ(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text('{"v": 1}', encoding="utf-8")
    target.write_text('{"v": 2}', encoding="utf-8")
    assert detect_drift(source, target) is True


def test_sync_copies_source_to_target(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target" / "a02.json"
    source.write_text('{"rules": []}', encoding="utf-8")
    sync_rules(source, target)
    assert target.exists()
    assert target.read_text(encoding="utf-8") == '{"rules": []}'
