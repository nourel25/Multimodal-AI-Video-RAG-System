import pytest

from helpers.config import Settings
from stores.llm import LLMError
from stores.llm.FallbackLLMProvider import FallbackLLMProvider
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.rate_limit import RateLimiter
from stores.llm.retry import call_with_retry, retry_after_seconds
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
                        GENERATION_FALLBACK_BACKEND="OLLAMA", VISION_BACKEND="")
    client = LLMProviderFactory(settings).create_generation_client()
    assert isinstance(client, FallbackLLMProvider)
    assert client.name == "groq->ollama"
    assert LLMProviderFactory(settings).create_vision_client() is None


def test_rate_limiter_spaces_calls_evenly():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)  # one call per second
    for _ in range(3):
        limiter.acquire()
    assert slept == [1.0, 1.0]
    now[0] += 5  # idle time is not "saved up" into a burst
    assert limiter.acquire() == 0
    assert RateLimiter(0).acquire() == 0


class _RateLimited(Exception):
    def __init__(self, message, headers=None):
        super().__init__(message)
        self.response = type("Response", (), {"headers": headers or {}})()


def test_retry_after_is_read_from_header_or_body():
    assert retry_after_seconds(_RateLimited("429", {"retry-after": "7"})) == 7.0
    assert retry_after_seconds(_RateLimited('quota exceeded "retryDelay": "17s"')) == 17.0
    assert retry_after_seconds(_RateLimited("429")) is None


def test_retry_waits_what_the_provider_asks(monkeypatch):
    waits = []
    monkeypatch.setattr("tenacity.nap.time.sleep", waits.append)
    attempts = []

    def limited():
        attempts.append(1)
        if len(attempts) == 1:
            raise _RateLimited("slow down", {"retry-after": "3"})
        return "ok"

    assert call_with_retry(limited, provider="t", is_transient=lambda e: True) == "ok"
    assert waits == [3.0]
