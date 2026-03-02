"""Session manager for concurrent agent instances in server mode."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from .agent import Agent
from .channels.base import InboundMessage
from .config import Config

logger = logging.getLogger(__name__)


class Session:
    """Wraps a single Agent instance with metadata."""

    def __init__(self, key: str, agent: Agent):
        self.key = key
        self.agent = agent
        self.last_active: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def handle_message(self, text: str) -> str:
        """Send a message to the agent and return the response text.

        Serialises concurrent messages for the same session so context
        stays consistent.
        """
        async with self._lock:
            self.last_active = time.monotonic()

            self.agent.messages.append({"role": "user", "content": text})

            # Run the agent loop (non-interactive, no streaming)
            await self.agent._agent_loop(non_interactive=True)

            # Extract the last assistant response
            for msg in reversed(self.agent.messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return msg["content"]

            return "(no response)"


class SessionManager:
    """Manages concurrent agent sessions keyed by session_key."""

    def __init__(self, config: Config):
        self.config = config
        self._sessions: Dict[str, Session] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background idle-cleanup loop."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop cleanup and close all sessions."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for session in list(self._sessions.values()):
            await self._close_session(session)
        self._sessions.clear()

    def get_or_create(self, session_key: str) -> Session:
        """Return an existing session or create a new one."""
        if session_key not in self._sessions:
            agent = Agent(self.config, non_interactive=True)
            session = Session(session_key, agent)
            self._sessions[session_key] = session
            logger.info("Created session %s", session_key)
        return self._sessions[session_key]

    async def route_message(self, msg: InboundMessage) -> str:
        """Route an inbound message to the appropriate session and return the response."""
        session = self.get_or_create(msg.session_key)
        return await session.handle_message(msg.text)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Idle cleanup
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Periodically evict idle sessions."""
        max_idle = getattr(self.config, "_server_max_idle_minutes", 60) * 60
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            expired = [
                k
                for k, s in self._sessions.items()
                if (now - s.last_active) > max_idle
            ]
            for key in expired:
                session = self._sessions.pop(key, None)
                if session:
                    logger.info("Evicting idle session %s", key)
                    await self._close_session(session)

    async def _close_session(self, session: Session) -> None:
        """Gracefully close a session's agent."""
        try:
            await session.agent.llm_client.close()
        except Exception:
            logger.exception("Error closing session %s", session.key)
