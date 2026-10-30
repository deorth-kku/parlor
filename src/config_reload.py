"""Online config reload: hot-swap the LLM backend and TTS client.

The server registers a ``SIGHUP`` handler (see ``server.py``). On signal,
:meth:`ConfigManager.apply` re-reads the config file, validates it, and
swaps in fresh backend/TTS clients. If the new config is invalid the
previous clients keep running and the error is reported.

Retired clients are closed after a grace period so in-flight turns that
still hold references to them can finish without "closed client" errors.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Config, load_config
from openai_backend import OpenAICompatibleBackend
from tts_remote import RemoteTTS

# Grace period before retired clients are closed, so in-flight turns can finish.
DEFAULT_GRACE_SECONDS = 30.0


@dataclass
class ReloadResult:
    """Outcome of a single reload attempt."""

    ok: bool
    message: str
    config: Config | None = None


class ConfigManager:
    """Owns the current backend/TTS clients and hot-swaps them on reload."""

    def __init__(
        self,
        path: str | Path,
        *,
        backend: OpenAICompatibleBackend | None = None,
        tts: RemoteTTS | None = None,
        config: Config | None = None,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
    ):
        self._path = Path(path)
        self._backend = backend
        self._tts = tts
        self._config = config
        self._grace_seconds = grace_seconds
        self._lock = asyncio.Lock()
        self._retired: list[tuple[float, Any]] = []
        self._retire_task: asyncio.Task[None] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def backend(self) -> OpenAICompatibleBackend | None:
        return self._backend

    @property
    def tts(self) -> RemoteTTS | None:
        return self._tts

    @property
    def config(self) -> Config | None:
        return self._config

    def set_current(
        self,
        backend: OpenAICompatibleBackend | None,
        tts: RemoteTTS | None,
        config: Config | None = None,
    ) -> None:
        """Record the clients created at startup as the current ones."""
        self._backend = backend
        self._tts = tts
        self._config = config

    async def apply(self) -> ReloadResult:
        """Re-read the config file and hot-swap the clients if valid.

        On any failure (unreadable file, bad JSON, missing TTS reference
        audio) the previous clients are kept and an error result is
        returned.
        """
        async with self._lock:
            try:
                new_config = load_config(self._path)
            except (ValueError, OSError) as exc:
                return ReloadResult(False, f"Failed to load config {self._path}: {exc}")

            new_backend: OpenAICompatibleBackend | None = None
            new_tts: RemoteTTS | None = None
            try:
                new_backend = OpenAICompatibleBackend.from_config(new_config)
                new_tts = RemoteTTS(new_config)
                new_tts.validate()
            except Exception as exc:
                # Close whatever was created; the previous clients keep running.
                if new_backend is not None:
                    await new_backend.aclose()
                if new_tts is not None:
                    await new_tts.aclose()
                return ReloadResult(
                    False, f"Invalid config {self._path}, keeping previous: {exc}"
                )

            old_backend, old_tts = self._backend, self._tts
            # Freeze the old TTS's config so in-flight turns on the retired
            # client don't mix new config values (languages, defaults) with
            # old ones during the grace period.
            if old_tts is not None:
                old_tts.freeze_config()
            self._backend = new_backend
            self._tts = new_tts
            self._config = new_config
            self._retire(old_backend, old_tts)
            print(f"[config] reloaded {self._path}")
            return ReloadResult(True, f"Config reloaded from {self._path}", new_config)

    def _retire(self, *clients: Any) -> None:
        deadline = time.monotonic() + self._grace_seconds
        for client in clients:
            if client is not None:
                self._retired.append((deadline, client))

    async def start(self) -> None:
        """Start the background task that closes retired clients."""
        if self._retire_task is None:
            self._retire_task = asyncio.create_task(self._retire_loop(), name="config-retire")

    async def stop(self) -> None:
        """Cancel the retire loop (call before closing the current clients)."""
        if self._retire_task is not None:
            self._retire_task.cancel()
            try:
                await self._retire_task
            except asyncio.CancelledError:
                pass
            self._retire_task = None

    async def _retire_loop(self) -> None:
        while True:
            now = time.monotonic()
            due: list[Any] = []
            while self._retired and self._retired[0][0] <= now:
                due.append(self._retired.pop(0)[1])
            for client in due:
                try:
                    await client.aclose()
                except Exception as exc:
                    print(f"[config] failed to close retired client: {exc!r}")
            await asyncio.sleep(1.0)
