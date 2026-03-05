"""Slack channel integration using Socket Mode."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .base import Channel, InboundMessage, OutboundMessage

if TYPE_CHECKING:
    from ..sessions import SessionManager

logger = logging.getLogger(__name__)


def _import_slack_sdk():
    """Lazy-import slack_sdk so the rest of koi works without the server extra."""
    try:
        from slack_sdk.socket_mode.aiohttp import SocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse
        from slack_sdk.web.async_client import AsyncWebClient

        return AsyncWebClient, SocketModeClient, SocketModeRequest, SocketModeResponse
    except ImportError:
        raise ImportError(
            "Slack integration requires the 'server' extra. "
            "Install with: pip install 'koi[server]'"
        )


class SlackChannel(Channel):
    """Slack channel using Socket Mode (no public URL needed).

    Behaviours:
    - Ack with eyes reaction while processing
    - Always reply in thread
    - Mention-gated in channels (@mention required), respond to all DMs
    - Show typing indicator while agent is working
    - Session key format:
        DM:      slack:dm:<user_id>
        Thread:  slack:channel:<channel_id>:thread:<thread_ts>
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        session_manager: SessionManager,
        *,
        mention_only_in_channels: bool = True,
        ack_reaction: str = "eyes",
    ):
        AsyncWebClient, SocketModeClient, _, _ = _import_slack_sdk()  # noqa: N806

        self._web = AsyncWebClient(token=bot_token)
        self._socket = SocketModeClient(app_token=app_token, web_client=self._web)
        self._session_manager = session_manager
        self._mention_only = mention_only_in_channels
        self._ack_reaction = ack_reaction
        self._bot_user_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        # Resolve our own bot user ID so we can detect @mentions
        resp = await self._web.auth_test()
        self._bot_user_id = resp["user_id"]
        logger.info("Slack bot user ID: %s", self._bot_user_id)

        # Register the event handler and connect
        self._socket.socket_mode_request_listeners.append(self._on_socket_event)
        await self._socket.connect()
        logger.info("Slack Socket Mode connected")

    async def stop(self) -> None:
        await self._socket.close()
        logger.info("Slack Socket Mode disconnected")

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(self, message: OutboundMessage) -> None:
        """Post a message to Slack (always in-thread)."""
        # Parse channel_id from session_key
        parts = message.session_key.split(":")
        if parts[1] == "dm":
            channel = parts[2]  # DM channel is the user_id; Slack opens a DM
        else:
            channel = parts[2]  # channel_id

        kwargs = {"channel": channel, "text": message.text}
        if message.thread_id:
            kwargs["thread_ts"] = message.thread_id

        await self._web.chat_postMessage(**kwargs)

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _on_socket_event(self, client, req) -> None:
        """Handle a raw Socket Mode request."""
        _, _, SocketModeRequest, SocketModeResponse = _import_slack_sdk()  # noqa: N806

        # Always ack immediately to avoid Slack retries
        response = SocketModeResponse(envelope_id=req.envelope_id)
        await client.send_socket_mode_response(response)

        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        await self._handle_event(event)

    async def _handle_event(self, event: dict) -> None:
        """Process a Slack event and route to the session manager."""
        event_type = event.get("type", "")

        # Ignore bot's own messages
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return

        # Ignore message subtypes (edits, deletes, etc.) except for normal messages
        if event.get("subtype"):
            return

        # Only handle message and app_mention events
        if event_type not in ("message", "app_mention"):
            return

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts") or event.get("ts", "")
        ts = event.get("ts", "")
        channel_type = event.get("channel_type", "")

        if not text or not user_id:
            return

        is_dm = channel_type == "im"
        is_mention = event_type == "app_mention" or (
            self._bot_user_id and f"<@{self._bot_user_id}>" in text
        )

        # Mention gating: in channels, only respond to @mentions or app_mention events
        if not is_dm and self._mention_only and not is_mention:
            return

        # Strip the bot mention from the text
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        if not text:
            return

        # Build session key
        if is_dm:
            session_key = f"slack:dm:{user_id}"
        else:
            session_key = f"slack:channel:{channel_id}:thread:{thread_ts}"

        inbound = InboundMessage(
            text=text,
            user_id=user_id,
            user_name=user_id,  # Could resolve via users.info, but keep simple
            channel_id=channel_id,
            session_key=session_key,
            thread_id=thread_ts,
            source="slack",
            raw=event,
        )

        # Process in a task so we don't block the socket listener
        asyncio.create_task(self._process_message(inbound, channel_id, ts, thread_ts))

    async def _process_message(
        self,
        msg: InboundMessage,
        channel_id: str,
        ts: str,
        thread_ts: str,
    ) -> None:
        """Ack, process, and reply to a message."""
        # Ack with reaction
        try:
            await self._web.reactions_add(
                channel=channel_id,
                timestamp=ts,
                name=self._ack_reaction,
            )
        except Exception:
            logger.debug("Could not add ack reaction", exc_info=True)

        # Route message to session manager
        try:
            response_text = await self._session_manager.route_message(msg)
        except Exception:
            logger.exception("Error processing message in session %s", msg.session_key)
            response_text = "Sorry, I encountered an error processing your message."

        # Remove ack reaction
        try:
            await self._web.reactions_remove(
                channel=channel_id,
                timestamp=ts,
                name=self._ack_reaction,
            )
        except Exception:
            logger.debug("Could not remove ack reaction", exc_info=True)

        # Reply in thread
        reply_thread = thread_ts or ts
        outbound = OutboundMessage(
            text=response_text,
            session_key=msg.session_key,
            thread_id=reply_thread,
        )
        try:
            await self.send_message(outbound)
        except Exception:
            logger.exception(
                "Failed to send Slack reply for session %s",
                msg.session_key,
            )
