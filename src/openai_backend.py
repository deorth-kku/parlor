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


def _extract_json_string_field(text: str, field_name: str) -> tuple[str | None, bool]:
    """Return the current value of a JSON string field if present.

    The second return value indicates whether the closing quote has been seen.
    """

    marker = f'"{field_name}"'
    start = text.find(marker)
    if start == -1:
        return None, False

    colon = text.find(":", start + len(marker))
    if colon == -1:
        return None, False

    i = colon + 1
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != '"':
        return None, False

    i += 1
    out: list[str] = []
    escape = False
    unicode_digits = ""

    while i < len(text):
        ch = text[i]
        if unicode_digits:
            if ch.lower() in "0123456789abcdef":
                unicode_digits += ch
                if len(unicode_digits) == 4:
                    out.append(chr(int(unicode_digits, 16)))
                    unicode_digits = ""
                    escape = False
                i += 1
                continue
            return "".join(out), False

        if escape:
            mapping = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if ch == "u":
                unicode_digits = ""
                i += 1
                continue
            if ch not in mapping:
                return "".join(out), False
            out.append(mapping[ch])
            escape = False
            i += 1
            continue

        if ch == "\\":
            escape = True
            i += 1
            continue
        if ch == '"':
            return "".join(out), True
        out.append(ch)
        i += 1

    return "".join(out), False


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

    async def resolve_model(self) -> str | None:
        """Fetch the first model ID from the API."""
        try:
            response = await self._client.get("models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return models[0].get("id")
        except Exception:
            pass
        return None

    @classmethod
    async def from_env(cls) -> "OpenAICompatibleBackend":
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

    async def stream_turn(
        self,
        history: list[dict[str, Any]],
        *,
        user_content: str | list[dict[str, Any]],
    ):
        payload: dict[str, Any] = {
            "messages": [*history, {"role": "user", "content": user_content}],
            "temperature": self._config.temperature,
            "reasoning_format": "none",
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
            "stream": True,
        }
        if self._config.model:
            payload["model"] = self._config.model
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
        payload["response_format"] = self._schema()

        async def emit_structured_stream(lines):
            raw_content = ""
            last_response_text = ""
            last_transcription_text = ""

            async for line in lines:
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue

                raw_content += delta

                transcription_text, transcription_complete = _extract_json_string_field(raw_content, "transcription")
                if transcription_text is not None and transcription_text != last_transcription_text:
                    last_transcription_text = transcription_text
                    yield {
                        "type": "transcription",
                        "text": transcription_text,
                        "complete": transcription_complete,
                    }

                response_text, response_complete = _extract_json_string_field(raw_content, "response")
                if response_text is not None:
                    if response_text.startswith(last_response_text):
                        new_text = response_text[len(last_response_text):]
                    else:
                        new_text = response_text
                    if new_text:
                        last_response_text = response_text
                        yield {"type": "delta", "delta": new_text, "complete": response_complete}

            parsed = _extract_json_payload(raw_content)
            parsed.setdefault("transcription", "")
            parsed.setdefault("response", "")
            parsed["_raw_content"] = raw_content
            parsed["_usage"] = {}
            yield {"type": "done", "parsed": parsed}

        async with self._client.stream("POST", "chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                text = body.decode("utf-8", errors="ignore").lower()
                if "response_format" in text or "json_schema" in text:
                    fallback = dict(payload)
                    fallback.pop("response_format", None)
                    async with self._client.stream("POST", "chat/completions", json=fallback) as fallback_response:
                        fallback_response.raise_for_status()
                        async for event in emit_structured_stream(fallback_response.aiter_lines()):
                            yield event
                        return
                response.raise_for_status()

            async for event in emit_structured_stream(response.aiter_lines()):
                yield event
