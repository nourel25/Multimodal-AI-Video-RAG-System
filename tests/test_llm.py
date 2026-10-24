import pytest

from helpers.config import Settings
from stores.llm import LLMError
from stores.llm.FallbackLLMProvider import FallbackLLMProvider
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.retry import call_with_retry
from tests.fakes import FakeLLM

MESSAGES = [{"role": "user", "content": "hi"}]


class HalfStreamLLM(FakeLLM):
    def stream(self, messages, **kwargs):
        yield "partial"
        raise LLMError("connection dropped", provider="half")


def test_fallback_used_when_primary_fails():
    provider = FallbackLLMProvider(FakeLLM(fail=True), FakeLLM(answer="from fallback"))
    assert provider.generate(MESSAGES) == "from fallback"
    assert list(provider.stream(MESSAGES)) == ["from", "fallback"]


def test_fallback_not_used_after_stream_started():
    fallback = FakeLLM(answer="should not appear")
    provider = FallbackLLMProvider(HalfStreamLLM(), fallback)
    tokens = []
    with pytest.raises(LLMError):
        for token in provider.stream(MESSAGES):
            tokens.append(token)
    assert tokens == ["partial"]
    assert fallback.calls == []


def test_retry_recovers_from_transient_errors(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("slow")
        return "ok"

    assert call_with_retry(flaky, provider="t", is_transient=lambda e: True,
                           max_attempts=3) == "ok"
    assert len(attempts) == 3


def test_retry_does_not_retry_permanent_errors():
    attempts = []

    def broken():
        attempts.append(1)
        raise ValueError("bad request")

    with pytest.raises(LLMError) as info:
        call_with_retry(broken, provider="t", is_transient=lambda e: False, max_attempts=3)
    assert len(attempts) == 1
    assert info.value.transient is False


def test_factory_requires_api_keys_and_known_backends():
    factory = LLMProviderFactory(Settings(_env_file=None, GROQ_API_KEY="", GEMINI_API_KEY=""))
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        factory.create("GROQ")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        factory.create("gemini")
    with pytest.raises(ValueError, match="Unsupported"):
        factory.create("OPENAI")


def test_factory_wraps_generation_client_with_fallback():
    settings = Settings(_env_file=None, GENERATION_BACKEND="GROQ", GROQ_API_KEY="k",
                        GENERATION_FALLBACK_BACKEND="OLLAMA")
    client = LLMProviderFactory(settings).create_generation_client()
    assert isinstance(client, FallbackLLMProvider)
    assert client.name == "groq->ollama"
    assert LLMProviderFactory(settings).create_vision_client() is None
