"""Parlor - real-time multimodal AI (voice + vision)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from config import DEFAULT_CONFIG_PATH, config, reload as reload_config
from config_reload import ConfigManager
from openai_backend import OpenAICompatibleBackend, response_text_from_value
from tts_remote import RemoteTTS
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
tts_remote: RemoteTTS | None = None
manager: ConfigManager | None = None
config_path = DEFAULT_CONFIG_PATH
_reload_task: asyncio.Task[None] | None = None


def get_tts_defaults() -> dict[str, str]:
    if tts_remote is None:
        return {"tts_language": "en", "tts_voice": ""}
    return {
        "tts_language": tts_remote.default_language,
        "tts_voice": tts_remote.default_voice,
    }


def resolve_tts_selection(msg: dict[str, object]) -> tuple[str, str]:
    defaults = get_tts_defaults()
    language = str(msg.get("tts_language") or defaults["tts_language"])
    valid_languages = {item["code"] for item in tts_remote.languages()} if tts_remote else set()
    if language not in valid_languages:
        language = defaults["tts_language"]
    voice = str(msg.get("tts_voice") or defaults["tts_voice"])
    valid_voices = set(tts_remote.voice_names()) if tts_remote else set()
    if voice not in valid_voices:
        voice = defaults["tts_voice"]
    return language, voice


async def load_models():
    global backend, tts_remote
    backend = OpenAICompatibleBackend.from_config(config)
    print(f"OpenAI-compatible backend loaded: {backend.base_url} model={backend.model}")
    tts_remote = RemoteTTS(config)
    tts_remote.validate()
    print(f"Remote TTS loaded: {tts_remote.base_url}")
    if manager is not None:
        manager.set_current(backend, tts_remote, config)


def on_sighup() -> None:
    """SIGHUP callback: hot-reload the config file (no restart needed)."""
    global _reload_task
    if manager is None:
        return
    if _reload_task is not None and not _reload_task.done():
        # A reload is already in flight; the lock in ConfigManager would
        # serialize them, but the in-flight one reads the same file state.
        return
    print("SIGHUP received: reloading config")
    _reload_task = asyncio.create_task(apply_reload())


async def apply_reload() -> None:
    """Reload the config and refresh every component that reads it.

    The manager swaps in fresh backend/TTS clients; the module-level
    globals (used by the WebSocket handlers) and the config singleton are
    updated to point at the new ones.
    """
    global backend, tts_remote
    result = await manager.apply()
    if result.ok:
        backend = manager.backend
        tts_remote = manager.tts
        if result.config is not None:
            config.replace(result.config.data)
        print(
            f"Reloaded: backend={backend.base_url if backend else None} "
            f"model={backend.model if backend else None} "
            f"tts={tts_remote.base_url if tts_remote else None}"
        )
    else:
        print(result.message)


@asynccontextmanager
async def lifespan(app):
    global manager
    manager = ConfigManager(config_path)
    await load_models()
    await manager.start()
    loop = asyncio.get_running_loop()
    signal_installed = False
    try:
        loop.add_signal_handler(signal.SIGHUP, on_sighup)
        signal_installed = True
    except NotImplementedError:
        # Platforms without loop signal support (e.g. Windows).
        print("SIGHUP reload not supported on this platform")
    try:
        yield
    finally:
        if signal_installed:
            loop.remove_signal_handler(signal.SIGHUP)
        if _reload_task is not None and not _reload_task.done():
            await _reload_task
        await manager.stop()
        if backend is not None:
            await backend.aclose()
        if tts_remote is not None:
            await tts_remote.aclose()


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
    on_sample_rate: Callable[[int], Any],
) -> bool:
    """Stream one sentence's PCM from the remote TTS server to the client.

    Raw float32 PCM is forwarded as binary WebSocket frames. The remote's
    sample rate (reported on the first chunk) is surfaced via ``on_sample_rate``
    so the caller can emit ``audio_start``.
    """
    if tts_remote is None:
        return False
    async for sample_rate, chunk in tts_remote.stream_pcm(sentence, language, voice):
        if not should_send():
            return False
        if sample_rate is not None:
            await on_sample_rate(sample_rate)
        await ws.send_bytes(chunk)
    return True


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent.parent / "public" / "index.html").read_text(encoding="utf-8"))

@app.get("/styles.css")
async def styles_css():
    return Response(content=(Path(__file__).parent.parent / "public" / "styles.css").read_text(encoding="utf-8"), media_type="text/css")

@app.get("/app.js")
async def app_js():
    return Response(content=(Path(__file__).parent.parent / "public" / "app.js").read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/favicon.png")
async def favicon():
    favicon_path = Path(__file__).parent.parent / "public" / "favicon.png"
    return Response(content=favicon_path.read_bytes(), media_type="image/png")


@app.get("/api/tts/options")
async def tts_options():
    if tts_remote is None:
        return {"languages": [], "voices": [], "default_language": "en", "default_voice": ""}
    return {
        "languages": tts_remote.languages(),
        "voices": tts_remote.voice_names(),
        "default_language": tts_remote.default_language,
        "default_voice": tts_remote.default_voice,
    }


@app.get("/api/model")
async def model_info():
    if backend is None:
        return {"model": "auto"}
    if backend._config.model is not None:
        return {"model": backend._config.model}
    resolved = await backend.resolve_model()
    return {"model": resolved or "auto"}


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
            stream = backend.stream_turn(
                history[:-1],
                user_content=user_content,
                should_continue=lambda: not interrupted.is_set(),
            )

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

            async def on_sample_rate(sample_rate: int) -> None:
                nonlocal tts_started
                if not tts_started:
                    tts_started = True
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "audio_start",
                                "sample_rate": sample_rate,
                                "turn_id": turn_id,
                            }
                        )
                    )

            async def tts_worker() -> None:
                nonlocal sentence_index, tts_started
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        break
                    if not is_turn_active():
                        continue

                    sent = await speak_sentence(
                        ws,
                        sentence,
                        sentence_index,
                        tts_language,
                        tts_voice,
                        turn_id,
                        is_turn_active,
                        on_sample_rate,
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
                history.append(
                    {
                        "role": "system",
                        "content": "The user cancelled (interrupted) their previous query; no assistant response was generated for it.",
                    }
                )
                try:
                    await ws.send_text(
                        json.dumps({"type": "turn_interrupted", "turn_id": turn_id})
                    )
                except Exception as exc:
                    print(f"Failed to send turn_interrupted: {exc!r}")
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
    parser = argparse.ArgumentParser(description="Parlor real-time multimodal AI server")
    parser.add_argument(
        "--config",
        "-c",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    reload_config(args.config)
    port = int(config.get("server.port", 8000))
    print(f"Send SIGHUP to {os.getpid()} to reload the config file")
    uvicorn.run(app, host="0.0.0.0", port=port)
