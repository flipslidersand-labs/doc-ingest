"""Shared pytest fixtures for doc-ingest tests."""

import pytest


@pytest.fixture(autouse=True)
def _allow_any_base_dir(monkeypatch):
    """Set DOC_INGEST_BASE_DIR=/ so path_guard allows tmp_path and other
    arbitrary paths used in tests without requiring real project files."""
    monkeypatch.setenv("DOC_INGEST_BASE_DIR", "/")


@pytest.fixture(autouse=True)
def _reset_settings():
    """Drop any cached/overridden Settings so tests never leak config into each other."""
    from core import config

    config._settings = None
    yield
    config._settings = None


def override_settings(**changes):
    """Replace the cached Settings with a copy that has `changes` applied (reset per test)."""
    import dataclasses

    from core import config

    config._settings = dataclasses.replace(config.get_settings(), **changes)
