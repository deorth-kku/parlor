"""OpenAI-compatible multimodal chat client."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

STRUCTURED_OUTPUT_PROMPT = (
    "Return exactly one JSON object with two string fields: "
    '"transcription" and "response". '
    "Use an empty string for transcription when there is no spoken audio to transcribe. "
    "Do not include markdown fences, commentary, role tags, or any extra keys."
)

PLAIN_RESPONSE_PROMPT = (
    "Answer the user's latest request directly in 1-4 short sentences. "
    "Do not emit JSON, markdown fences, thinking tags, or role labels."
)

ROLE_LABEL_PATTERN = r"(?:assistant|user|system|tool)"


@dataclass(slots=True)
class OpenAIBackendConfig:
    base_url: str
    api_key: str | None
    model: str | None
    timeout: float
    temperature: float
    max_tokens: int | None
    structured_output_mode: str


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_structured_output_mode(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"auto", "schema", "prompt"}:
        return raw
    return default


def _cleanup_stream_text(text: str) -> str:
    cleaned = text.replace("<|im_end|>", "").replace('<|"|>', "").replace("<|eot_id|>", "")
    cleaned = re.sub(rf"<\|im_start\|>{ROLE_LABEL_PATTERN}\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        rf"<\|start_header_id\|>{ROLE_LABEL_PATTERN}<\|end_header_id\|>\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"<think>\s*</think>\s*", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL)
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].lstrip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].lstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned


def _cleanup_model_output(text: str) -> str:
    cleaned = _cleanup_stream_text(text)
    cleaned = re.sub(rf"^\s*{ROLE_LABEL_PATTERN}\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(rf"(^|\n)\s*{ROLE_LABEL_PATTERN}\s*(?=\n|$)", r"\1", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = _cleanup_model_output(text)
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


def _parse_turn_output(text: str) -> dict[str, Any]:
    cleaned = _cleanup_model_output(text)
    try:
        parsed = _extract_json_payload(cleaned)
    except json.JSONDecodeError:
        return {
            "transcription": "",
            "response": cleaned,
        }

    transcription = parsed.get("transcription", "")
    response = parsed.get("response", "")
    return {
        "transcription": transcription if isinstance(transcription, str) else str(transcription),
        "response": response if isinstance(response, str) else str(response),
    }


def _append_instruction_to_user_content(
    user_content: str | list[dict[str, Any]],
    instruction: str,
) -> str | list[dict[str, Any]]:
    if isinstance(user_content, str):
        return f"{user_content.rstrip()}\n\n{instruction}"
    if isinstance(user_content, list):
        updated = list(user_content)
        updated.append({"type": "text", "text": instruction})
        return updated
    return user_content


def _model_prefers_prompt_structured_output(model_name: str | None) -> bool:
    if not model_name:
        return False
    lowered = model_name.lower()
    return "qwen" in lowered or "omni" in lowered


def _should_switch_to_raw_text(cleaned_text: str) -> bool:
    compact = cleaned_text.lstrip()
    if len(compact) < 24:
        return False
    if compact.startswith("{") or compact.startswith("```"):
        return False
    if '"transcription"' in compact or '"response"' in compact:
        return False
    return True


def _is_degenerate_model_output(text: str) -> bool:
    cleaned = _cleanup_model_output(text)
    if not cleaned:
        return True
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return True
    return all(re.fullmatch(ROLE_LABEL_PATTERN, line, flags=re.IGNORECASE) for line in lines)


class OpenAICompatibleBackend:
    def __init__(self, config: OpenAIBackendConfig):
        headers = {}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._config = config
        self._resolved_model: str | None = config.model
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
                    model_id = models[0].get("id")
                    if isinstance(model_id, str) and model_id:
                        self._resolved_model = model_id
                    return model_id
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
        structured_output_mode = _env_structured_output_mode("OPENAI_STRUCTURED_OUTPUT_MODE", "auto")

        return cls(
            OpenAIBackendConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
                structured_output_mode=structured_output_mode,
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

    def _model_name_hint(self) -> str | None:
        return self._config.model or self._resolved_model

    def _preferred_structured_output_mode(self) -> str:
        mode = self._config.structured_output_mode
        if mode != "auto":
            return mode
        if _model_prefers_prompt_structured_output(self._model_name_hint()):
            return "prompt"
        return "schema"

    def _request_modes(self) -> list[str]:
        preferred = self._preferred_structured_output_mode()
        if self._config.structured_output_mode == "auto" and preferred == "schema":
            return ["schema", "prompt"]
        return [preferred]

    @staticmethod
    def _should_retry_with_prompt(mode: str, status_code: int, body_text: str) -> bool:
        if mode != "schema":
            return False
        text = body_text.lower()
        indicators = (
            "response_format",
            "json_schema",
            "grammar",
            "sampler",
            "structured output",
            "<think>",
            "thinking",
        )
        return status_code >= 500 or any(indicator in text for indicator in indicators)

    def _build_messages(
        self,
        history: list[dict[str, Any]],
        user_content: str | list[dict[str, Any]],
        *,
        structured_mode: str,
        prompt_variant: str,
    ) -> list[dict[str, Any]]:
        messages = [*history]
        if structured_mode == "prompt":
            instruction = STRUCTURED_OUTPUT_PROMPT if prompt_variant == "json" else PLAIN_RESPONSE_PROMPT
            user_content = _append_instruction_to_user_content(user_content, instruction)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_payload(
        self,
        history: list[dict[str, Any]],
        *,
        user_content: str | list[dict[str, Any]],
        stream: bool,
        structured_mode: str,
        prompt_variant: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": self._build_messages(
                history,
                user_content,
                structured_mode=structured_mode,
                prompt_variant=prompt_variant,
            ),
            "temperature": self._config.temperature,
            "reasoning_format": "none",
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }
        if stream:
            payload["stream"] = True
        if self._config.model:
            payload["model"] = self._config.model
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
        if structured_mode == "schema":
            payload["response_format"] = self._schema()
        else:
            payload["stop"] = [
                "<|im_start|>",
                "<|start_header_id|>",
                "\nuser\n",
                "\nassistant\n",
                "\nsystem\n",
                "\ntool\n",
            ]
        return payload

    def _request_attempts(self) -> list[tuple[str, str]]:
        attempts: list[tuple[str, str]] = []
        for mode in self._request_modes():
            if mode == "schema":
                attempts.append(("schema", "json"))
            else:
                attempts.append(("prompt", "json"))
        if not any(mode == "prompt" and variant == "plain" for mode, variant in attempts):
            attempts.append(("prompt", "plain"))
        return attempts

    async def complete_turn(
        self,
        history: list[dict[str, Any]],
        *,
        user_content: str | list[dict[str, Any]],
    ) -> dict[str, Any]:
        last_response: httpx.Response | None = None
        last_mode = self._preferred_structured_output_mode()
        last_variant = "json"

        for structured_mode, prompt_variant in self._request_attempts():
            last_mode = structured_mode
            last_variant = prompt_variant
            payload = self._build_payload(
                history,
                user_content=user_content,
                stream=False,
                structured_mode=structured_mode,
                prompt_variant=prompt_variant,
            )
            response = await self._client.post("chat/completions", json=payload)
            last_response = response
            if response.status_code >= 400:
                if self._should_retry_with_prompt(structured_mode, response.status_code, response.text):
                    continue
                response.raise_for_status()

            data = response.json()
            choice = data["choices"][0]["message"]
            content = self._join_message_content(choice.get("content"))
            if structured_mode == "prompt" and _is_degenerate_model_output(content):
                continue
            parsed = _parse_turn_output(content)
            parsed["_raw_content"] = content
            parsed["_usage"] = data.get("usage", {})
            parsed["_structured_output_mode"] = f"{structured_mode}:{prompt_variant}"
            return parsed

        if last_response is None:
            raise RuntimeError(f"Failed to submit completion request in mode={last_mode}:{last_variant}")
        last_response.raise_for_status()
        raise RuntimeError("Completion request failed unexpectedly")

    async def stream_turn(
        self,
        history: list[dict[str, Any]],
        *,
        user_content: str | list[dict[str, Any]],
    ):
        async def emit_structured_stream(lines, structured_mode: str, prompt_variant: str):
            raw_content = ""
            last_response_text = ""
            last_transcription_text = ""
            last_clean_text = ""
            raw_text_mode = False

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
                cleaned = _cleanup_stream_text(raw_content)

                transcription_text, transcription_complete = _extract_json_string_field(cleaned, "transcription")
                response_text, response_complete = _extract_json_string_field(cleaned, "response")

                if transcription_text is not None and transcription_text != last_transcription_text:
                    last_transcription_text = transcription_text
                    yield {
                        "type": "transcription",
                        "text": transcription_text,
                        "complete": transcription_complete,
                    }

                if response_text is not None:
                    if response_text.startswith(last_response_text):
                        new_text = response_text[len(last_response_text):]
                    else:
                        new_text = response_text
                    if new_text:
                        last_response_text = response_text
                        yield {"type": "delta", "delta": new_text, "complete": response_complete}
                    continue

                if structured_mode == "prompt" and not raw_text_mode and _should_switch_to_raw_text(cleaned):
                    raw_text_mode = True

                if raw_text_mode:
                    if cleaned.startswith(last_clean_text):
                        new_text = cleaned[len(last_clean_text) :]
                    else:
                        new_text = cleaned
                    if new_text:
                        last_clean_text = cleaned
                        yield {"type": "delta", "delta": new_text, "complete": False}

            parsed = _parse_turn_output(raw_content)
            parsed["_raw_content"] = raw_content
            parsed["_usage"] = {}
            parsed["_structured_output_mode"] = f"{structured_mode}:{prompt_variant}"
            yield {"type": "done", "parsed": parsed}

        last_error_text = ""
        last_fallback_event: dict[str, Any] | None = None
        for structured_mode, prompt_variant in self._request_attempts():
            payload = self._build_payload(
                history,
                user_content=user_content,
                stream=True,
                structured_mode=structured_mode,
                prompt_variant=prompt_variant,
            )
            saw_meaningful_text = False
            last_done_event: dict[str, Any] | None = None
            async with self._client.stream("POST", "chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    last_error_text = body.decode("utf-8", errors="ignore")
                    if self._should_retry_with_prompt(structured_mode, response.status_code, last_error_text):
                        continue
                    response.raise_for_status()

                async for event in emit_structured_stream(response.aiter_lines(), structured_mode, prompt_variant):
                    if event["type"] == "delta" and str(event.get("delta", "")).strip():
                        saw_meaningful_text = True
                        yield event
                        continue
                    if event["type"] == "transcription" and str(event.get("text", "")).strip():
                        yield event
                        continue
                    if event["type"] == "done":
                        last_done_event = event
                        continue
                if last_done_event is None:
                    return
                parsed = last_done_event["parsed"]
                last_fallback_event = last_done_event
                if structured_mode == "prompt" and not saw_meaningful_text and _is_degenerate_model_output(
                    str(parsed.get("_raw_content", ""))
                ):
                    continue
                yield last_done_event
                return

        if last_fallback_event is not None:
            yield last_fallback_event
            return

        fallback_note = last_error_text.strip() or "all structured-output attempts returned degenerate content"
        yield {
            "type": "done",
            "parsed": {
                "transcription": "",
                "response": "",
                "_raw_content": "",
                "_usage": {},
                "_structured_output_mode": "failed",
                "_error": fallback_note,
            },
        }
