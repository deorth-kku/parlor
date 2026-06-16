"""Parlor - real-time multimodal AI (voice + vision)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts
from openai_backend import OpenAICompatibleBackend
from voices_catalog import VOICE_CATALOG
from tts_prompt import build_language_instruction

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful real-time multimodal AI assistant. "
    "You can understand speech, images, and text. "
    "Return concise answers and keep responses to 1-4 short sentences. "
    "Always transcribe the user's speech faithfully before responding."
)

SENTENCE_END_RE = re.compile(r"[.!?。！？]")

backend: OpenAICompatibleBackend | None = None
tts_backend = None


def get_tts_defaults() -> dict[str, str]:
    language = VOICE_CATALOG.default_language
    return {
        "tts_language": language,
        "tts_voice": VOICE_CATALOG.default_voices[language],
    }


def resolve_tts_selection(msg: dict[str, object]) -> tuple[str, str]:
    defaults = get_tts_defaults()
    language = str(msg.get("tts_language") or defaults["tts_language"])
    if language not in VOICE_CATALOG.languages:
        language = defaults["tts_language"]
    voice = str(msg.get("tts_voice") or VOICE_CATALOG.languages[language]["default_voice"])
    valid_voices = VOICE_CATALOG.languages[language]["voices"]
    if voice not in valid_voices:
        voice = str(VOICE_CATALOG.languages[language]["default_voice"])
    return language, voice


def load_models():
    global backend, tts_backend
    backend = OpenAICompatibleBackend.from_env()
    print(f"OpenAI-compatible backend loaded: {backend._config.base_url} model={backend._config.model}")
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    try:
        yield
    finally:
        if backend is not None:
            await backend.aclose()


app = FastAPI(lifespan=lifespan)


def build_user_content(msg: dict[str, object]) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    tts_language, tts_voice = resolve_tts_selection(msg)

    if msg.get("audio"):
        parts.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": msg["audio"],
                    "format": "wav",
                },
            }
        )

    if msg.get("image"):
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{msg['image']}",
                },
            }
        )

    if msg.get("text"):
        parts.append({"type": "text", "text": str(msg["text"])})
    elif msg.get("audio") and msg.get("image"):
        parts.append(
            {
                "type": "text",
                "text": "The user is speaking while showing an image. Transcribe the speech and answer using the image if relevant.",
            }
        )
    elif msg.get("audio"):
        parts.append({"type": "text", "text": "Transcribe the user's speech and respond briefly."})
    elif msg.get("image"):
        parts.append({"type": "text", "text": "Describe the image briefly and naturally."})
    else:
        parts.append({"type": "text", "text": "Hello!"})

    parts.append({"type": "text", "text": build_language_instruction(tts_language, tts_voice)})
    return parts


def append_sentence_buffer(buffer: str, chunk: str) -> tuple[str, list[str]]:
    text = buffer + chunk
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return text[start:], sentences


def json_string_field_is_closed(text: str, field_name: str) -> tuple[str | None, bool]:
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


async def speak_sentence(ws: WebSocket, sentence: str, index: int, language: str, voice: str) -> None:
    pcm = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda s=sentence, lang=language, selected_voice=voice: tts_backend.generate(s, lang, selected_voice),
    )
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    await ws.send_text(
        json.dumps(
            {
                "type": "audio_chunk",
                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                "index": index,
            }
        )
    )


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.get("/api/tts/options")
async def tts_options():
    return {
        "languages": [
            {
                "code": code,
                "label": data["label"],
                "voices": data["voices"],
                "default_voice": data["default_voice"],
            }
            for code, data in VOICE_CATALOG.languages.items()
        ],
        "default_language": VOICE_CATALOG.default_language,
        "default_voice": VOICE_CATALOG.default_voices[VOICE_CATALOG.default_language],
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if backend is None:
        await ws.close(code=1011, reason="Backend not initialized")
        return

    interrupted = asyncio.Event()
    msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    history: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()
            tts_language, tts_voice = resolve_tts_selection(msg)
            user_content = build_user_content(msg)
            history.append({"role": "user", "content": user_content})

            t0 = time.time()
            stream = backend.stream_turn(history[:-1], user_content=user_content)

            assistant_text = ""
            assistant_transcription = ""
            pending_text = ""
            pending_sentences: list[str] = []
            tts_started = False
            sentence_index = 0

            async def flush_pending_sentences() -> None:
                nonlocal sentence_index, tts_started
                while pending_sentences and not interrupted.is_set():
                    if not tts_started:
                        tts_started = True
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "audio_start",
                                    "sample_rate": tts_backend.sample_rate,
                                    "sentence_count": 0,
                                }
                            )
                        )
                    sentence = pending_sentences.pop(0)
                    await speak_sentence(ws, sentence, sentence_index, tts_language, tts_voice)
                    sentence_index += 1

            try:
                await ws.send_text(json.dumps({"type": "text_start"}))
                async for event in stream:
                    if interrupted.is_set():
                        break

                    if event["type"] == "delta":
                        delta = str(event["delta"])
                        assistant_text += delta
                        await ws.send_text(json.dumps({"type": "text_delta", "delta": delta}))

                        pending_text, new_sentences = append_sentence_buffer(pending_text, delta)
                        if new_sentences:
                            pending_sentences.extend(new_sentences)
                            await flush_pending_sentences()

                    elif event["type"] == "transcription":
                        transcription = str(event["text"]).replace('<|"|>', "").strip()
                        if transcription:
                            assistant_transcription = transcription
                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": "transcription_delta",
                                        "text": transcription,
                                        "complete": bool(event.get("complete", False)),
                                    }
                                )
                            )

                    elif event["type"] == "done":
                        parsed = event["parsed"]
                        assistant_transcription = str(parsed.get("transcription", "")).strip()
                        final_response = str(parsed.get("response", "")).strip()
                        if final_response and final_response != assistant_text:
                            assistant_text = final_response
                        if assistant_transcription:
                            assistant_transcription = assistant_transcription.replace('<|"|>', "").strip()

                if pending_text.strip():
                    pending_sentences.append(pending_text.strip())
                    pending_text = ""
                if pending_sentences and not interrupted.is_set():
                    await flush_pending_sentences()

            finally:
                llm_time = time.time() - t0

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            history.append({"role": "assistant", "content": assistant_text or "Okay."})

            print(f"LLM ({llm_time:.2f}s) [stream]: {assistant_transcription!r} -> {assistant_text}")

            await ws.send_text(
                json.dumps(
                    {
                        "type": "text_end",
                        "text": assistant_text or "Okay.",
                        "llm_time": round(llm_time, 2),
                        **({"transcription": assistant_transcription} if assistant_transcription else {}),
                    }
                )
            )

            if not tts_started:
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_start",
                            "sample_rate": tts_backend.sample_rate,
                            "sentence_count": 0,
                        }
                    )
                )
                tts_started = True

            if not interrupted.is_set():
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_end",
                            "tts_time": 0 if sentence_index == 0 else round(time.time() - t0, 2),
                        }
                    )
                )

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
