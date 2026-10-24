from enum import StrEnum


class LLMBackend(StrEnum):
    OLLAMA = "OLLAMA"
    GROQ = "GROQ"
    GEMINI = "GEMINI"


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
