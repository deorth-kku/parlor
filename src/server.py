"""Parlor - real-time multimodal AI (voice + vision)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

import tts
from openai_backend import OpenAICompatibleBackend, response_text_from_value
from voices_catalog import VOICE_CATALOG
from tts_prompt import build_language_instruction

load_dotenv()

SYSTEM_PROMPT = (
    "You are a real-time multimodal AI assistant. "
    "You can understand speech, images, and text. "
    "Return concise answers and keep responses to 1-4 sentences. "
    "Always transcribe the user's speech faithfully before responding. "
    "When producing structured JSON, set 'response' to an array of short spoken chunks, "
    "with one sentence or speakable clause per item."
)

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


async def load_models():
    global backend, tts_backend
    loop = asyncio.get_event_loop()
    backend = await OpenAICompatibleBackend.from_env()
    print(f"OpenAI-compatible backend loaded: {backend._config.base_url} model={backend._config.model}")
    tts_backend = await loop.run_in_executor(None, tts.load)


@asynccontextmanager
async def lifespan(app):
    await load_models()
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


async def speak_sentence(
    ws: WebSocket,
    sentence: str,
    index: int,
    language: str,
    voice: str,
    turn_id: str,
    should_send: Callable[[], bool],
) -> bool:
    pcm = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda s=sentence, lang=language, selected_voice=voice: tts_backend.generate(s, lang, selected_voice),
    )
    if not should_send():
        return False
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    await ws.send_text(
        json.dumps(
            {
                "type": "audio_chunk",
                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                "index": index,
                "turn_id": turn_id,
            }
        )
    )
    return True


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


@app.get("/api/model")
async def model_info():
    if backend is None:
        return {"model": "auto"}
    if backend._config.model is not None:
        return {"model": backend._config.model}
    resolved = await backend.resolve_model()
    return {"model": resolved or "auto"}


# ── OpenAI-compatible TTS endpoints ──────────────────────────────────

_TTS_MODELS = [
    {"id": "kokoro", "object": "model", "created": 1687584915, "owned_by": "hexgrad"},
]

_AUDIO_FORMATS = {
    "mp3": {"container": "mp3", "codec": "mp3", "media_type": "audio/mpeg", "extension": "mp3"},
    "wav": {"container": None, "codec": None, "media_type": "audio/wav", "extension": "wav"},
    "opus": {"container": "ogg", "codec": "libopus", "media_type": "audio/ogg; codecs=opus", "extension": "opus"},
    "flac": {"container": "flac", "codec": "flac", "media_type": "audio/flac", "extension": "flac"},
    "pcm": {"container": None, "codec": None, "media_type": "application/octet-stream", "extension": "pcm"},
    "aac": {"container": "adts", "codec": "aac", "media_type": "audio/aac", "extension": "aac"},
}

_SUPPORTED_FORMATS = tuple(_AUDIO_FORMATS)


def _detect_language(text: str) -> str:
    """Detect language from text content: ja, zh, or en."""
    has_kana = any(0x3040 <= ord(ch) <= 0x30FF for ch in text)
    has_cjk = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
    if has_kana:
        return "ja"
    if has_cjk:
        return "zh"
    return "en"


class TTSRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0


def _write_wav(pcm: np.ndarray, sr: int) -> bytes:
    """Encode PCM float32 to a proper WAV file."""
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    data = pcm_int16.tobytes()
    import struct
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # fmt chunk size
    buf.write(struct.pack("<H", 1))   # PCM
    buf.write(struct.pack("<H", 1))   # mono
    buf.write(struct.pack("<I", sr))  # sample rate
    buf.write(struct.pack("<I", sr * 2))  # byte rate
    buf.write(struct.pack("<H", 2))   # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


async def _encode_audio(pcm: np.ndarray, fmt: str, sr: int) -> bytes:
    """Encode PCM float32 [-1,1] to the requested format."""
    fmt = fmt.lower()
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if fmt not in _AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {fmt}")
    if fmt in {"wav", "pcm"}:
        if fmt == "wav":
            return _write_wav(pcm, sr)
        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
        return pcm_int16.tobytes()
    return _encode_audio_with_pyav(pcm, fmt, sr)


def _encode_audio_with_pyav(pcm: np.ndarray, fmt: str, sr: int) -> bytes:
    fmt_config = _AUDIO_FORMATS[fmt]
    import av

    arr = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    out = io.BytesIO()
    with av.open(out, mode="w", format=fmt_config["container"]) as out_ctx:
        stream = out_ctx.add_stream(fmt_config["codec"], rate=sr)
        stream.layout = "mono"
        chunk_size = stream.frame_size or 1024
        for i in range(0, len(arr), chunk_size):
            chunk = arr[i : i + chunk_size]
            if len(chunk) == 0:
                break
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), layout="mono", format="s16"
            )
            frame.sample_rate = sr
            for packet in stream.encode(frame):
                out_ctx.mux(packet)
        for packet in stream.encode(None):
            out_ctx.mux(packet)
    return out.getvalue()


def _error_response(message: str, error_type: str, param: str | None, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": param, "code": status_code}},
    )



@app.get("/v1/audio/models")
async def tts_model_list():
    return {"object": "list", "data": _TTS_MODELS}


@app.post("/v1/audio/speech")
async def tts_speech(req: TTSRequest):
    if tts_backend is None:
        return _error_response("TTS backend not loaded", "server_error", None, 500)

    response_format = req.response_format.lower()
    if response_format not in _SUPPORTED_FORMATS:
        return _error_response(
            f"Invalid response format: {req.response_format}",
            "invalid_request_error",
            "response_format",
            400,
        )

    try:
        language = _detect_language(req.input)
        voice = req.voice if req.voice else resolve_tts_selection({"tts_language":language})[1]
        pcm = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: tts_backend.generate(req.input, language, voice, speed=req.speed),
        )
        audio_bytes = await _encode_audio(pcm, response_format, tts_backend.sample_rate)
        format_config = _AUDIO_FORMATS[response_format]
        return Response(
            content=audio_bytes,
            media_type=format_config["media_type"],
            headers={"Content-Disposition": f'attachment; filename="speech.{format_config["extension"]}"'}
        )
    except ValueError as exc:
        return _error_response(str(exc), "invalid_request_error", "voice", 400)
    except Exception as exc:
        print(exc)
        return _error_response(str(exc), "server_error", None, 500)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if backend is None:
        await ws.close(code=1011, reason="Backend not initialized")
        return

    interrupted = asyncio.Event()
    msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    history: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn_counter = 0
    current_turn_id = ""

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
            turn_counter += 1
            turn_id = f"turn-{turn_counter}"
            current_turn_id = turn_id
            tts_language, tts_voice = resolve_tts_selection(msg)
            user_content = build_user_content(msg)
            history.append({"role": "user", "content": user_content})

            t0 = time.time()
            stream = backend.stream_turn(history[:-1], user_content=user_content)

            assistant_text = ""
            assistant_transcription = ""
            streamed_response_text = ""
            tts_started = False
            sentence_index = 0
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            turn_failed = False
            final_text = "Okay."
            final_transcription = ""
            text_end_sent = False
            audio_end_sent = False

            def is_turn_active() -> bool:
                return current_turn_id == turn_id and not interrupted.is_set()

            async def enqueue_sentences(sentences: list[str]) -> None:
                for sentence in sentences:
                    if sentence:
                        await sentence_queue.put(sentence)

            async def tts_worker() -> None:
                nonlocal sentence_index, tts_started
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        break
                    if not is_turn_active():
                        continue
                    if not tts_started:
                        tts_started = True
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "audio_start",
                                    "sample_rate": tts_backend.sample_rate,
                                    "sentence_count": 0,
                                    "turn_id": turn_id,
                                }
                            )
                        )
                    sent = await speak_sentence(
                        ws,
                        sentence,
                        sentence_index,
                        tts_language,
                        tts_voice,
                        turn_id,
                        is_turn_active,
                    )
                    if not sent:
                        break
                    sentence_index += 1

            tts_task = asyncio.create_task(tts_worker())

            try:
                await ws.send_text(json.dumps({"type": "text_start", "turn_id": turn_id}))
                async for event in stream:
                    if interrupted.is_set():
                        break

                    if event["type"] == "delta":
                        delta = str(event["delta"])
                        assistant_text += delta
                        streamed_response_text = assistant_text
                        await ws.send_text(json.dumps({"type": "text_delta", "delta": delta, "turn_id": turn_id}))
                    elif event["type"] == "replace":
                        assistant_text = str(event["text"])
                        streamed_response_text = assistant_text
                        await ws.send_text(
                            json.dumps({"type": "text_replace", "text": assistant_text, "turn_id": turn_id})
                        )
                    elif event["type"] == "response_item":
                        sentence = str(event["text"]).strip()
                        if sentence:
                            await enqueue_sentences([sentence])

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
                                        "turn_id": turn_id,
                                    }
                                )
                            )

                    elif event["type"] == "done":
                        parsed = event["parsed"]
                        assistant_transcription = str(parsed.get("transcription", "")).strip()
                        final_response = response_text_from_value(parsed.get("response", []))
                        if final_response.strip():
                            if final_response != assistant_text:
                                assistant_text = final_response
                        elif streamed_response_text.strip():
                            assistant_text = streamed_response_text
                        else:
                            turn_failed = True
                        if assistant_transcription:
                            assistant_transcription = assistant_transcription.replace('<|"|>', "").strip()

            except Exception as exc:
                turn_failed = True
                print(f"LLM turn failed: {exc!r}")

            finally:
                llm_time = time.time() - t0
                final_transcription = assistant_transcription
                final_text = assistant_text.strip() or "Okay."
                await sentence_queue.put(None)
                try:
                    await tts_task
                except Exception as exc:
                    print(f"TTS turn failed: {exc!r}")

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            history.append({"role": "assistant", "content": final_text})

            print(f"LLM ({llm_time:.2f}s) [stream]: {final_transcription!r} -> {final_text}")

            if not text_end_sent:
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "text_end",
                            "text": final_text,
                            "llm_time": round(llm_time, 2),
                            "turn_id": turn_id,
                            **({"transcription": final_transcription} if final_transcription else {}),
                        }
                    )
                )
                text_end_sent = True

            if is_turn_active():
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_end",
                            "tts_time": 0 if sentence_index == 0 else round(time.time() - t0, 2),
                            "turn_id": turn_id,
                        }
                    )
                )
                audio_end_sent = True

            if turn_failed and not audio_end_sent and is_turn_active():
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_end",
                            "tts_time": 0,
                            "turn_id": turn_id,
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
