"""HTTP server for Koi webhook/channel integrations."""

from __future__ import annotations

from loguru import logger

from .channels.base import Channel
from .config import Config
from .sessions import SessionManager


def _import_starlette():
    """Lazy-import Starlette so the rest of koi works without the server extra."""
    try:
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        return Starlette, JSONResponse, Route
    except ImportError:
        raise ImportError("Server mode requires the 'server' extra. Install with: pip install 'koi[server]'")


class KoiServer:
    """Koi HTTP server that hosts channels and a health endpoint."""

    def __init__(self, config: Config, channels: list[Channel] | None = None):
        self.config = config
        self.channels: list[Channel] = channels or []
        self.session_manager = SessionManager(config)

        Starlette, JSONResponse, Route = _import_starlette()  # noqa: N806

        async def health(request):
            return JSONResponse(
                {
                    "status": "ok",
                    "sessions": self.session_manager.active_count,
                    "channels": [type(ch).__name__ for ch in self.channels],
                }
            )

        self.app = Starlette(
            routes=[Route("/health", health, methods=["GET"])],
            on_startup=[self._on_startup],
            on_shutdown=[self._on_shutdown],
        )

    async def _on_startup(self) -> None:
        """Start session manager and all channels."""
        await self.session_manager.start()
        for ch in self.channels:
            try:
                await ch.start()
                logger.info("Started channel: {}", type(ch).__name__)
            except Exception:
                logger.exception("Failed to start channel: {}", type(ch).__name__)

    async def _on_shutdown(self) -> None:
        """Gracefully stop channels and session manager."""
        for ch in self.channels:
            try:
                await ch.stop()
            except Exception:
                logger.exception("Error stopping channel: {}", type(ch).__name__)
        await self.session_manager.stop()

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the server with uvicorn (blocking)."""
        try:
            import uvicorn
        except ImportError:
            raise ImportError("Server mode requires uvicorn. Install with: pip install 'koi[server]'")
        uvicorn.run(self.app, host=host, port=port, log_level="info")
