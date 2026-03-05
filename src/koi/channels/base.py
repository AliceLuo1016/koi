"""Channel abstraction for messaging integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InboundMessage:
    """Normalized inbound message from any channel."""

    text: str
    user_id: str
    user_name: str
    channel_id: str
    session_key: str
    thread_id: str | None = None
    source: str = ""  # "slack", "webhook", etc.
    raw: dict = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """Message to send back through a channel."""

    text: str
    session_key: str
    thread_id: str | None = None
    reaction: str | None = None  # e.g. "eyes" for ack


class Channel(ABC):
    """Base class for messaging channels."""

    @abstractmethod
    async def start(self) -> None:
        """Start the channel connection (e.g. open WebSocket)."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""

    @abstractmethod
    async def send_message(self, message: OutboundMessage) -> None:
        """Send a message back to the channel."""
