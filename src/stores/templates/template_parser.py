import importlib
import os


class TemplateParser:
    def __init__(self, language: str | None = None, default_language: str = "en"):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_language = default_language
        self.language = default_language
        self.set_language(language)

    def _language_exists(self, language: str) -> bool:
        return os.path.isdir(os.path.join(self.current_path, "locales", language))

    def set_language(self, language: str | None):
        self.language = (
            language if language and self._language_exists(language) else self.default_language
        )

    def resolve_language(self, language: str | None) -> str:
        """Return `language` if templates exist for it, else the default."""
        return language if language and self._language_exists(language) else self.default_language

    def get(self, group: str, key: str, vars: dict | None = None,
            language: str | None = None) -> str | None:
        if not group or not key:
            return None

        for candidate in (self.resolve_language(language or self.language),
                          self.default_language):
            group_path = os.path.join(self.current_path, "locales", candidate, f"{group}.py")
            if not os.path.exists(group_path):
                continue
            module = importlib.import_module(f"stores.templates.locales.{candidate}.{group}")
            template = getattr(module, key, None)
            if template is not None:
                return template.substitute(vars or {})
        return None
