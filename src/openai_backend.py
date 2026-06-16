"""OpenAI-compatible multimodal chat client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class OpenAIBackendConfig:
    base_url: str
    api_key: str | None
    model: str | None
    timeout: float
    temperature: float
    max_tokens: int | None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


class OpenAICompatibleBackend:
    def __init__(self, config: OpenAIBackendConfig):
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout,
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleBackend":
        base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
        model = os.environ.get("OPENAI_MODEL", "").strip() or None
        timeout = _env_float("OPENAI_TIMEOUT", 120.0)
        temperature = _env_float("OPENAI_TEMPERATURE", 0.2)
        max_tokens = _env_int("OPENAI_MAX_TOKENS", 4096)

        return cls(
            OpenAIBackendConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _schema() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "parlor_turn",
                "schema": {
                    "type": "object",
                    "properties": {
                        "transcription": {"type": "string"},
                        "response": {"type": "string"},
                    },
                    "required": ["transcription", "response"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @staticmethod
    def _join_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part.strip())
        return str(content)

    async def complete_turn(
        self,
        history: list[dict[str, Any]],
        *,
        user_content: str | list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [*history, {"role": "user", "content": user_content}],
            "temperature": self._config.temperature,
            "reasoning_format": "none",
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        if self._config.model:
            payload["model"] = self._config.model
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
        payload["response_format"] = self._schema()

        response = await self._client.post("chat/completions", json=payload)
        if response.status_code >= 400:
            body = response.text.lower()
            if "response_format" in body or "json_schema" in body:
                fallback = dict(payload)
                fallback.pop("response_format", None)
                response = await self._client.post("chat/completions", json=fallback)
            response.raise_for_status()

        data = response.json()
        choice = data["choices"][0]["message"]
        content = self._join_message_content(choice.get("content"))
        parsed = _extract_json_payload(content)
        parsed.setdefault("transcription", "")
        parsed.setdefault("response", "")
        parsed["_raw_content"] = content
        parsed["_usage"] = data.get("usage", {})
        return parsed
