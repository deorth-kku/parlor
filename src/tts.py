"""kokoro-onnx TTS backend with configurable language and voice."""

from __future__ import annotations

import time
from dataclasses import dataclass

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


class ONNXBackend(TTSBackend):
    """kokoro-onnx backend (ONNX Runtime, CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download
        from misaki import en, espeak, ja, zh

        model_path = hf_hub_download("xun/kokoro-v1.1-zh-onnx", "onnx/kokoro-v1.1-zh.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")
        config_path = hf_hub_download("hexgrad/Kokoro-82M-v1.1-zh", "config.json")

        model = kokoro_onnx.Kokoro(model_path, voices_path, vocab_config=config_path)

        self._en_us = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
        self._en_uk = en.G2P(trf=False, british=True, fallback=espeak.EspeakFallback(british=True))
        self._ja = ja.JAG2P(version='pyopenjtalk')
        self._zh = zh.ZHG2P(version="1.1", en_callable=lambda text: self._en_us(text)[0])

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

    def generate(self, text: str, language: str, voice: str, speed: float = 1.1) -> np.ndarray:
        if language not in self._runtimes:
            raise ValueError(f"Unsupported language: {language}")
        self._validate_voice(language, voice)

        runtime = self._runtimes[language]
        t0 = time.time()

        if language == "en":
            g2p = self._en_uk if self._en_lang_for_voice(voice) == "en-gb" else self._en_us
            phonemes, _ = g2p(text)
            pcm, _sr = runtime.model.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
        elif language == "ja":
            phonemes, _ = runtime.g2p(text)
            phonemes = zipja(phonemes)
            pcm, _sr = runtime.model.create(phonemes, voice=voice, speed=speed, is_phonemes=True)
        else:
            phonemes, _ = runtime.g2p(text)
            pcm, _sr = runtime.model.create(phonemes, voice=voice, speed=speed, is_phonemes=True)

        print(f"tts language={language} voice={voice} elapsed={time.time() - t0:.2f}s text=\"{text}\" pho={phonemes}")
        return pcm
    
def zipja(s: str):
    half = len(s) // 2
    return s[:half]
    return "".join(a + b for a, b in zip(s[:half], s[half:]))

def load() -> TTSBackend:
    """Load the TTS backend."""
    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend
