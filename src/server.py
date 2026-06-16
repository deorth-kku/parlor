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

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts
from openai_backend import OpenAICompatibleBackend

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful real-time multimodal AI assistant. "
    "You can understand speech, images, and text. "
    "Return concise answers and keep responses to 1-4 short sentences."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

backend: OpenAICompatibleBackend | None = None
tts_backend = None


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


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def build_user_content(msg: dict[str, object]) -> list[dict[str, object]] | str:
    parts: list[dict[str, object]] = []

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

    return parts if (msg.get("audio") or msg.get("image")) else parts[0]["text"]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    if backend is None:
        await ws.close(code=1011, reason="Backend not initialized")
        return

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()
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
            user_content = build_user_content(msg)

            t0 = time.time()
            response = await backend.complete_turn(history, user_content=user_content)
            llm_time = time.time() - t0

            transcription = str(response.get("transcription", "")).strip()
            text_response = str(response.get("response", "")).strip()
            if not text_response:
                text_response = str(response.get("_raw_content", "")).strip() or "Okay."

            if transcription:
                transcription = transcription.replace('<|"|>', "").strip()
                print(f"LLM ({llm_time:.2f}s) [json] heard: {transcription!r} -> {text_response}")
            else:
                print(f"LLM ({llm_time:.2f}s) [json]: {text_response}")

            history.append({"role": "user", "content": user_content})
            history.append({"role": "assistant", "content": text_response})

            if interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
            if transcription:
                reply["transcription"] = transcription
            await ws.send_text(json.dumps(reply))

            if interrupted.is_set():
                print("Interrupted before TTS, skipping audio")
                continue

            sentences = split_sentences(text_response)
            if not sentences:
                sentences = [text_response]

            tts_start = time.time()
            await ws.send_text(
                json.dumps(
                    {
                        "type": "audio_start",
                        "sample_rate": tts_backend.sample_rate,
                        "sentence_count": len(sentences),
                    }
                )
            )

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i + 1}/{len(sentences)})")
                    break

                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s)
                )

                if interrupted.is_set():
                    break

                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                            "index": i,
                        }
                    )
                )

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_end",
                            "tts_time": round(tts_time, 2),
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
