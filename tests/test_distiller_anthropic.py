"""Tests that execute the body of core.distiller._anthropic().

The ``anthropic.Anthropic`` class is replaced by a MagicMock, so the SDK never
creates an HTTP client and no network access to the Anthropic API can happen.
The API key is a dummy value.
"""

from types import SimpleNamespace

import pytest

from core import distiller

DUMMY_KEY = "sk-ant-dummy-key-for-tests"


def _msg(*texts):
    return SimpleNamespace(content=[SimpleNamespace(text=t) for t in texts])


@pytest.fixture(autouse=True)
def _dummy_config(mocker):
    mocker.patch.object(distiller, "ANTHROPIC_API_KEY", DUMMY_KEY)
    mocker.patch.object(distiller, "ANTHROPIC_MODEL", "claude-test-model")


@pytest.fixture
def sdk(mocker):
    """Patch anthropic.Anthropic; returns (class_mock, messages.create mock)."""
    cls = mocker.patch("anthropic.Anthropic")
    return cls, cls.return_value.messages.create


class TestAnthropic:
    def test_extracts_and_strips_text(self, sdk):
        sdk[1].return_value = _msg("  要点  \n")
        assert distiller._anthropic("hello") == "要点"

    def test_client_built_with_api_key(self, sdk):
        sdk[1].return_value = _msg("x")
        distiller._anthropic("p")
        sdk[0].assert_called_once_with(api_key=DUMMY_KEY)

    def test_request_shape(self, sdk):
        sdk[1].return_value = _msg("x")
        distiller._anthropic("my prompt")
        sdk[1].assert_called_once_with(
            model="claude-test-model",
            max_tokens=300,
            messages=[{"role": "user", "content": "my prompt"}],
        )

    def test_only_first_content_block_is_used(self, sdk):
        sdk[1].return_value = _msg("first", "second")
        assert distiller._anthropic("p") == "first"

    def test_empty_content_raises_index_error(self, sdk):
        sdk[1].return_value = _msg()
        with pytest.raises(IndexError):
            distiller._anthropic("p")

    def test_api_error_propagates(self, sdk):
        sdk[1].side_effect = RuntimeError("api down")
        with pytest.raises(RuntimeError, match="api down"):
            distiller._anthropic("p")


class TestGenerateFallsBackFromAnthropic:
    @pytest.fixture(autouse=True)
    def _ollama_down(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))

    def test_generate_uses_anthropic_text(self, sdk):
        sdk[1].return_value = _msg("distilled")
        assert distiller._generate("p") == "distilled"

    def test_empty_content_raises_distill_error(self, sdk):
        sdk[1].return_value = _msg()
        with pytest.raises(distiller.DistillError):
            distiller._generate("p")

    def test_api_error_raises_distill_error(self, sdk):
        sdk[1].side_effect = RuntimeError("api down")
        with pytest.raises(distiller.DistillError):
            distiller._generate("p")
