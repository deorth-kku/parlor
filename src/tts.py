"""Platform-aware Kokoro TTS: mlx-audio on Apple Silicon, kokoro-onnx elsewhere."""

import os
import platform
import sys
from pathlib import Path
import time
import numpy as np


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "jf_alpha", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class MLXBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model("mlx-community/Kokoro-82M-bf16")
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice="jf_alpha", speed=1.0))

    def generate(self, text: str, voice: str = "jf_alpha", speed: float = 1.1) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, speed=speed))
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def _to_hiragana(self, text: str) -> str:
        kks = kakasi()
        return "".join(item["hira"] for item in kks.convert(text))

    def __init__(self):
        import kokoro_onnx
        import torch
        print(f"torch:{torch.cuda.is_available()}")
        from misaki import ja
        self.g2p = ja.JAG2P()
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download("xun/kokoro-v1.1-zh-onnx", "onnx/kokoro-v1.1-zh.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")
        config_path = hf_hub_download("hexgrad/Kokoro-82M-v1.1-zh", "config.json")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path, vocab_config= config_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "jf_alpha", speed: float = 1.1) -> np.ndarray:
        t0 = time.time()
        phonemes, _ = self.g2p(text)
        pho_time = time.time()
        pcm, _sr = self._model.create(phonemes, voice=voice, speed=speed,is_phonemes=True)
        pcm_time = time.time()
        p1=pho_time-t0
        p2=pcm_time-pho_time
        print(f"pho:{p1},pcm:{p2},provider:{self._model.sess.get_providers()}")
        return pcm


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend
