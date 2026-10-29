# Parlor

Real-time multimodal AI. Have natural voice and vision conversations with an AI that runs through an OpenAI-compatible backend.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

## How it works

```
Browser (mic + camera)
    → WebSocket (audio PCM + JPEG frames)
    → FastAPI server
        → OpenAI-compatible multimodal API (audio + vision + text)
        → Kokoro TTS (MLX on Mac, ONNX on Linux)
    → WebSocket (streamed audio chunks)
Browser (playback + transcript)
```

- Voice Activity Detection in the browser.
- Barge-in support. Interrupt the AI mid-sentence by speaking.
- Sentence-level TTS streaming. Audio starts playing before the full response is generated.

## Requirements

- Python 3.12+
- An OpenAI-compatible multimodal backend that supports `audio`, `image_url`, and structured output
- macOS with Apple Silicon, or Linux with a supported GPU for TTS

## Quick start

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

cd src
uv sync
uv run server.py
```

Open `http://localhost:8000`, grant camera and microphone access, and start talking.

## Configuration

Settings are read from a nested JSON config file (default: `src/config.json`).
Copy `src/config.example.json` to `src/config.json` and edit it. Point the
server at a different file with the `--config` / `-c` flag:

```bash
uv run server.py --config /path/to/my-config.json
```

| Path | Default | Description |
| --- | --- | --- |
| `server.port` | `8000` | Server port |
| `openai.base_url` | `http://127.0.0.1:8080/v1` | OpenAI-compatible API base URL |
| `openai.api_key` | empty | API key if your backend requires one |
| `openai.model` | optional | Multimodal model name; omit to use the endpoint default |
| `openai.timeout` | `120` | Request timeout in seconds |
| `openai.temperature` | `0.2` | Sampling temperature |
| `openai.max_tokens` | `4096` | Max output tokens |

> TTS / ONNX Runtime options (e.g. `ONNX_PROVIDER`, `ORT_CUDA_*`) are still
> read from environment variables and are intentionally not part of the JSON
> config.

## Project structure

```
src/
├── server.py          # FastAPI WebSocket server + OpenAI-compatible backend
├── openai_backend.py   # HTTP client and response parsing
├── tts.py              # Platform-aware TTS
├── index.html          # Frontend UI
├── pyproject.toml      # Dependencies
└── benchmarks/
    └── bench.py       # End-to-end WebSocket benchmark
```

## Acknowledgments

- Kokoro TTS by Hexgrad
- Silero VAD for browser voice activity detection

## License

[Apache 2.0](LICENSE)
