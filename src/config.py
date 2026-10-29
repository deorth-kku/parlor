"""JSON-file based configuration for the Parlor application.

The application reads its settings from a nested JSON file (``config.json``)
instead of environment variables. Built-in defaults live in :data:`DEFAULTS`
so the app keeps working when the file — or individual keys — are missing.

Only settings that this project defines directly are modelled here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default location: alongside this module (src/config.json).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

# Built-in defaults, mirroring the previous environment variable defaults.
DEFAULTS: dict[str, Any] = {
    "server": {
        "port": 8000,
    },
    "openai": {
        "base_url": "http://127.0.0.1:8080/v1",
        "api_key": "",
        "model": "",
        "timeout": 120.0,
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "tts": {
        "base_url": "http://127.0.0.1:9931",
        "model": "",
        "languages": [],
        "default_language": "en",
        "default_voice": "",
        "voices": {},
        "timeout": 120.0,
        "params": {},
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` and return a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class Config:
    """Read-only access to the merged application configuration."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def replace(self, data: dict[str, Any]) -> None:
        """Replace the underlying data in place (used by :func:`reload`)."""
        self._data = data

    @property
    def data(self) -> dict[str, Any]:
        """The full merged configuration as a plain dict."""
        return self._data

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level section as a dict (empty dict if absent)."""
        value = self._data.get(name)
        return value if isinstance(value, dict) else {}

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. ``"openai.base_url"``."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None) -> Config:
    """Load the JSON config file and merge it over the built-in defaults.

    A missing file is not an error: the built-in defaults are used instead.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    file_data: dict[str, Any] = {}
    if config_path.is_file():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a JSON object: {config_path}")
        file_data = loaded
    return Config(_deep_merge(DEFAULTS, file_data))


# Module-level singleton, loaded once at import time.
config = load_config()


def reload(path: str | Path | None = None) -> Config:
    """Re-read the config file and update the module-level singleton in place.

    Returns the same :class:`Config` instance so existing references (e.g.
    ``from config import config``) keep pointing at the refreshed data.
    """
    config.replace(load_config(path).data)
    return config
