"""kokoro-onnx TTS backend with configurable language and voice."""

from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from voices_catalog import VOICE_CATALOG


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, language: str, voice: str, speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class _LanguageRuntime:
    language: str
    voices: set[str]
    model: object
    g2p: object


def _parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed


def _provider_name_from_env() -> str:
    provider = os.getenv("ONNX_PROVIDER", "").strip()
    if provider:
        return provider
    return "CPUExecutionProvider"


def _should_use_gpu(provider_name: str) -> bool:
    return provider_name in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}


def _memory_arena_shrink_target(provider_name: str) -> str:
    return "gpu:0" if _should_use_gpu(provider_name) else "cpu:0"


def _build_session_options(ort: Any, provider_name: str) -> Any:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_mem_reuse = True
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("memory.enable_memory_arena_shrinkage", _memory_arena_shrink_target(provider_name))
    return options


def _build_provider_config(provider_name: str) -> list[str | tuple[str, dict[str, Any]]]:
    if provider_name != "CUDAExecutionProvider":
        return [provider_name]

    provider_options: dict[str, Any] = {
        "arena_extend_strategy": os.getenv("ORT_CUDA_ARENA_EXTEND_STRATEGY", "kSameAsRequested"),
    }

    gpu_mem_limit = os.getenv("ORT_CUDA_GPU_MEM_LIMIT", "").strip()
    if gpu_mem_limit:
        provider_options["gpu_mem_limit"] = gpu_mem_limit

    use_ep_level_unified_stream = os.getenv("ORT_CUDA_USE_EP_LEVEL_UNIFIED_STREAM", "").strip()
    if use_ep_level_unified_stream:
        provider_options["use_ep_level_unified_stream"] = use_ep_level_unified_stream

    return [(provider_name, provider_options)]


def _create_kokoro_session(
    ort: Any,
    kokoro_onnx: Any,
    model_path: str,
    voices_path: str,
    config_path: str,
) -> tuple[object, dict[str, Any]]:
    provider_name = _provider_name_from_env()
    session_options = _build_session_options(ort, provider_name)
    providers = _build_provider_config(provider_name)
    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
    active_providers = session.get_providers()
    active_provider_name = next((name for name in active_providers if name != "CPUExecutionProvider"), active_providers[0])
    model = kokoro_onnx.Kokoro.from_session(session, voices_path, vocab_config=config_path)
    meta = {
        "provider_name": active_provider_name,
        "providers": active_providers,
        "provider_options": session.get_provider_options(),
        "gpu_enabled": any(_should_use_gpu(name) for name in active_providers),
    }
    return model, meta


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend with explicit ONNX Runtime session control."""

    def __init__(self):
        import kokoro_onnx
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from misaki import en, espeak, ja, zh

        self._kokoro_onnx = kokoro_onnx
        self._ort = ort
        self._model_path = hf_hub_download("xun/kokoro-v1.1-zh-onnx", "onnx/kokoro-v1.1-zh.onnx")
        self._voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")
        self._config_path = hf_hub_download("hexgrad/Kokoro-82M-v1.1-zh", "config.json")
        self._gpu_semaphore = threading.BoundedSemaphore(
            max(1, _parse_int_env("TTS_GPU_MAX_CONCURRENCY", 1))
        )
        self._session_lock = threading.RLock()
        self._run_options = ort.RunOptions()

        self._en_us = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
        self._en_uk = en.G2P(trf=False, british=True, fallback=espeak.EspeakFallback(british=True))
        self._ja = ja.JAG2P(version="pyopenjtalk")
        self._zh = zh.ZHG2P(version="1.1", en_callable=lambda text: self._en_us(text)[0])

        model, session_meta = _create_kokoro_session(
            self._ort,
            self._kokoro_onnx,
            self._model_path,
            self._voices_path,
            self._config_path,
        )
        self._run_options.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage",
            _memory_arena_shrink_target(session_meta["provider_name"]),
        )
        self._install_run_options(model)
        self._session_meta = session_meta

        self._runtimes = {
            "en": _LanguageRuntime(
                language="en",
                voices=set(VOICE_CATALOG.languages["en"]["voices"]),
                model=model,
                g2p=None,
            ),
            "ja": _LanguageRuntime(
                language="ja",
                voices=set(VOICE_CATALOG.languages["ja"]["voices"]),
                model=model,
                g2p=self._ja,
            ),
            "zh": _LanguageRuntime(
                language="zh",
                voices=set(VOICE_CATALOG.languages["zh"]["voices"]),
                model=model,
                g2p=self._zh,
            ),
        }
        self.sample_rate = 24000

    @staticmethod
    def _validate_voice(language: str, voice: str) -> None:
        if voice not in VOICE_CATALOG.languages[language]["voices"]:
            raise ValueError(f"Voice {voice} is not available for language {language}")

    @staticmethod
    def _en_lang_for_voice(voice: str) -> str:
        return "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "failed to allocate memory" in message or "cuda" in message and "out of memory" in message

    def _install_run_options(self, model: object) -> None:
        max_phoneme_length = self._kokoro_onnx.MAX_PHONEME_LENGTH
        run_options = self._run_options

        def _create_audio_with_run_options(phonemes: str, voice: np.ndarray, speed: float):
            phonemes = phonemes[:max_phoneme_length]
            tokens = np.array(model.tokenizer.tokenize(phonemes), dtype=np.int64)
            voice_slice = voice[len(tokens)]
            tokens_input = [[0, *tokens, 0]]
            if "input_ids" in [i.name for i in model.sess.get_inputs()]:
                inputs = {
                    "input_ids": tokens_input,
                    "style": np.array(voice_slice, dtype=np.float32),
                    "speed": np.array([speed], dtype=np.int32),
                }
            else:
                inputs = {
                    "tokens": tokens_input,
                    "style": voice_slice,
                    "speed": np.ones(1, dtype=np.float32) * speed,
                }
            audio = model.sess.run(None, inputs, run_options=run_options)[0]
            return audio, self.sample_rate

        model._create_audio = _create_audio_with_run_options

    def _rebuild_model_locked(self) -> None:
        model, session_meta = _create_kokoro_session(
            self._ort,
            self._kokoro_onnx,
            self._model_path,
            self._voices_path,
            self._config_path,
        )
        self._run_options = self._ort.RunOptions()
        self._run_options.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage",
            _memory_arena_shrink_target(session_meta["provider_name"]),
        )
        self._install_run_options(model)
        for runtime in self._runtimes.values():
            runtime.model = model
        self._session_meta = session_meta

    def _run_model_create(self, runtime: _LanguageRuntime, phonemes: str, voice: str, speed: float) -> np.ndarray:
        concurrency_guard = self._gpu_semaphore if self._session_meta.get("gpu_enabled") else nullcontext()
        with concurrency_guard:
            try:
                pcm, _sr = runtime.model.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
                return pcm
            except Exception as exc:
                if not self._session_meta.get("gpu_enabled") or not self._is_oom_error(exc):
                    raise
                print("TTS GPU OOM detected, rebuilding ONNX session and retrying once")
                with self._session_lock:
                    self._rebuild_model_locked()
                    runtime = self._runtimes[runtime.language]
                    pcm, _sr = runtime.model.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
                    return pcm

    def generate(self, text: str, language: str, voice: str, speed: float = 1.1) -> np.ndarray:
        if language not in self._runtimes:
            raise ValueError(f"Unsupported language: {language}")
        self._validate_voice(language, voice)

        runtime = self._runtimes[language]
        t0 = time.time()

        if language == "en":
            g2p = self._en_uk if self._en_lang_for_voice(voice) == "en-gb" else self._en_us
            phonemes, _ = g2p(text)
        elif language == "ja":
            phonemes, _ = runtime.g2p(text)
            phonemes = zipja(phonemes)
        else:
            phonemes, _ = runtime.g2p(text)

        pcm = self._run_model_create(runtime, phonemes, voice, speed)
        print(
            f"tts provider={self._session_meta['provider_name']} providers={self._session_meta['providers']} "
            f"voice={voice} lang={language} elapsed={time.time() - t0:.2f}s text=\"{text}\" pho={phonemes}"
        )
        return pcm


def zipja(s: str):
    half = len(s) // 2
    return s[:half]
    return "".join(a + b for a, b in zip(s[:half], s[half:]))


def load() -> TTSBackend:
    """Load the TTS backend."""
    backend = ONNXBackend()
    print(
        "TTS: kokoro-onnx "
        f"(provider={backend._session_meta['provider_name']}, providers={backend._session_meta['providers']}, "
        f"sample_rate={backend.sample_rate}, gpu_max_concurrency={_parse_int_env('TTS_GPU_MAX_CONCURRENCY', 1)})"
    )
    if backend._session_meta["provider_options"]:
        print(f"TTS provider options: {backend._session_meta['provider_options']}")
    return backend
