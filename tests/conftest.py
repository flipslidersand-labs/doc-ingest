"""Shared pytest fixtures for doc-ingest tests."""
import pytest


@pytest.fixture(autouse=True)
def _allow_any_base_dir(monkeypatch):
    """Set DOC_INGEST_BASE_DIR=/ so path_guard allows tmp_path and other
    arbitrary paths used in tests without requiring real project files."""
    monkeypatch.setenv("DOC_INGEST_BASE_DIR", "/")
