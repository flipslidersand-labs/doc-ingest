"""Tests for core/path_guard.py — path traversal prevention."""

import pytest

from core.path_guard import assert_safe_path
from tests.conftest import override_settings


class TestAssertSafePath:
    def test_file_inside_base_is_allowed(self, tmp_path):
        allowed = tmp_path / "data"
        allowed.mkdir()
        target = allowed / "notes.md"
        target.write_text("ok")
        result = assert_safe_path(target, base=allowed)
        assert result == target.resolve()

    def test_path_traversal_raises_permission_error(self, tmp_path):
        allowed = tmp_path / "data"
        allowed.mkdir()
        traversal = allowed / ".." / ".." / "etc" / "passwd"
        with pytest.raises(PermissionError, match="outside the allowed base"):
            assert_safe_path(traversal, base=allowed)

    def test_dotdot_string_raises(self, tmp_path):
        allowed = tmp_path / "safe"
        allowed.mkdir()
        with pytest.raises(PermissionError):
            assert_safe_path(str(allowed / "../../../etc/passwd"), base=allowed)

    def test_settings_override_base(self, tmp_path):
        custom_base = tmp_path / "custom"
        custom_base.mkdir()
        target = custom_base / "file.pdf"
        target.write_text("data")
        override_settings(doc_ingest_base_dir=str(custom_base))
        result = assert_safe_path(target)
        assert result == target.resolve()

    def test_settings_base_rejects_outside(self, tmp_path):
        custom_base = tmp_path / "custom"
        custom_base.mkdir()
        outside = tmp_path / "outside.pdf"
        outside.write_text("data")
        override_settings(doc_ingest_base_dir=str(custom_base))
        with pytest.raises(PermissionError):
            assert_safe_path(outside)

    def test_empty_setting_falls_back_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        override_settings(doc_ingest_base_dir="")
        inside = tmp_path / "a.md"
        inside.write_text("x")
        assert assert_safe_path(inside) == inside.resolve()
        with pytest.raises(PermissionError):
            assert_safe_path(tmp_path.parent / "outside.md")

    def test_relative_base_is_resolved_at_call_time(self, tmp_path, monkeypatch):
        (tmp_path / "rel").mkdir()
        monkeypatch.chdir(tmp_path)
        override_settings(doc_ingest_base_dir="rel")
        target = tmp_path / "rel" / "f.md"
        target.write_text("x")
        assert assert_safe_path(target) == target.resolve()
        with pytest.raises(PermissionError):
            assert_safe_path(tmp_path / "other.md")

    def test_returns_resolved_path(self, tmp_path):
        allowed = tmp_path
        target = allowed / "a" / ".." / "b.txt"
        (allowed / "b.txt").write_text("x")
        result = assert_safe_path(target, base=allowed)
        assert result == (allowed / "b.txt").resolve()
