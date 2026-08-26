"""Tests for core/path_guard.py — path traversal prevention."""
import pytest

from core.path_guard import assert_https_url, assert_safe_path


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

    def test_env_var_overrides_base(self, tmp_path, monkeypatch):
        custom_base = tmp_path / "custom"
        custom_base.mkdir()
        target = custom_base / "file.pdf"
        target.write_text("data")
        monkeypatch.setenv("DOC_INGEST_BASE_DIR", str(custom_base))
        result = assert_safe_path(target)
        assert result == target.resolve()

    def test_env_var_rejects_outside(self, tmp_path, monkeypatch):
        custom_base = tmp_path / "custom"
        custom_base.mkdir()
        outside = tmp_path / "outside.pdf"
        outside.write_text("data")
        monkeypatch.setenv("DOC_INGEST_BASE_DIR", str(custom_base))
        with pytest.raises(PermissionError):
            assert_safe_path(outside)

    def test_returns_resolved_path(self, tmp_path):
        allowed = tmp_path
        target = allowed / "a" / ".." / "b.txt"
        (allowed / "b.txt").write_text("x")
        result = assert_safe_path(target, base=allowed)
        assert result == (allowed / "b.txt").resolve()


class TestAssertHttpsUrl:
    def test_https_is_accepted(self):
        # Should not raise
        assert_https_url("https://example.com/paper.pdf")

    def test_http_raises_value_error(self):
        with pytest.raises(ValueError, match="https://"):
            assert_https_url("http://example.com/paper.pdf")

    def test_ftp_raises_value_error(self):
        with pytest.raises(ValueError):
            assert_https_url("ftp://example.com/paper.pdf")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            assert_https_url("")
