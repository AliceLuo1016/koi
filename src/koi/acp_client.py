"""ACP client — spawn and communicate with ACP-compatible coding agents."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import acp
    from acp.client import ClientSideConnection
    from acp.schema import (
        AgentMessageChunk,
        AgentThoughtChunk,
        ToolCallStart,
        ToolCallProgress,
        AgentPlanUpdate,
        UsageUpdate,
        TextContentBlock,
        ClientCapabilities,
        Implementation,
        RequestPermissionResponse,
        PermissionOption,
        AllowedOutcome,
        DeniedOutcome,
        ToolCallUpdate,
        ReadTextFileResponse,
        WriteTextFileResponse,
        CreateTerminalResponse,
        ReleaseTerminalResponse,
        TerminalOutputResponse,
        WaitForTerminalExitResponse,
        KillTerminalCommandResponse,
        FileSystemCapability,
    )
    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False


@dataclass
class ACPResult:
    """Result from an ACP prompt turn."""

    content: str
    stop_reason: str = "end_turn"
    tool_calls: List[dict] = field(default_factory=list)
    thoughts: str = ""
    usage: Optional[dict] = None


class KoiACPClient:
    """ACP Client implementation that auto-approves permissions and collects responses."""

    def __init__(self, auto_approve: bool = True, cwd: Optional[str] = None):
        self.auto_approve = auto_approve
        self.cwd = cwd or os.getcwd()
        self._collected_text = ""
        self._collected_thoughts = ""
        self._tool_calls: List[dict] = []
        self._usage: Optional[dict] = None
        self._text_event = asyncio.Event()

    def reset(self):
        """Reset collected state for a new prompt turn."""
        self._collected_text = ""
        self._collected_thoughts = ""
        self._tool_calls = []
        self._usage = None

    # ── ACP Client Protocol methods ──

    async def request_permission(
        self,
        options: list,
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        """Handle permission requests from the agent."""
        if self.auto_approve and options:
            option_id = options[0].id if hasattr(options[0], 'id') else options[0].option_id if hasattr(options[0], 'option_id') else "allow"
            return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=option_id))
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        """Handle session updates (streamed text, tool calls, etc.)."""
        if isinstance(update, AgentMessageChunk):
            if hasattr(update, 'content') and update.content:
                content = update.content
                if hasattr(content, 'text'):
                    self._collected_text += content.text
        elif isinstance(update, AgentThoughtChunk):
            if hasattr(update, 'content') and update.content:
                content = update.content
                if hasattr(content, 'text'):
                    self._collected_thoughts += content.text
        elif isinstance(update, (ToolCallStart, ToolCallProgress)):
            tool_info = {
                "type": type(update).__name__,
                "tool_call_id": getattr(update, 'tool_call_id', ''),
                "title": getattr(update, 'title', ''),
                "status": getattr(update, 'status', ''),
            }
            self._tool_calls.append(tool_info)
        elif isinstance(update, UsageUpdate):
            if hasattr(update, 'usage'):
                self._usage = update.usage if isinstance(update.usage, dict) else {}

    async def read_text_file(
        self, path: str, session_id: str, **kwargs: Any
    ) -> ReadTextFileResponse:
        """Allow agent to read files."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return ReadTextFileResponse(content=content)
        except Exception as e:
            return ReadTextFileResponse(content=f"Error reading {path}: {e}")

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse:
        """Allow agent to write files."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return WriteTextFileResponse()
        except Exception as e:
            return WriteTextFileResponse()

    async def create_terminal(self, command: str, session_id: str, **kwargs: Any) -> CreateTerminalResponse:
        return CreateTerminalResponse(terminal_id="unsupported")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        return TerminalOutputResponse(output="", truncated=False)

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> ReleaseTerminalResponse:
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> WaitForTerminalExitResponse:
        return WaitForTerminalExitResponse(exit_code=1)

    async def kill_terminal_command(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalCommandResponse:
        return KillTerminalCommandResponse()


class ACPSession:
    """Manages a connection to an ACP agent subprocess."""

    def __init__(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        auto_approve: bool = True,
        env: Optional[Dict[str, str]] = None,
    ):
        self.command = command
        self.cwd = cwd or os.getcwd()
        self.auto_approve = auto_approve
        self.env = env
        self.session_id: Optional[str] = None
        self._conn: Optional[ClientSideConnection] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._client: Optional[KoiACPClient] = None
        self._ctx = None

    async def start(self) -> str:
        """Spawn the agent process, initialize, and create a session.

        Returns the session_id.
        """
        if not ACP_AVAILABLE:
            raise ImportError(
                "acp-sdk is required for ACP sub-agents. "
                "Install it with: pip install 'koi[acp]' or pip install acp-sdk"
            )
        self._client = KoiACPClient(auto_approve=self.auto_approve, cwd=self.cwd)

        # Use the SDK's spawn helper
        self._ctx = acp.spawn_agent_process(
            self._client,
            self.command[0],
            *self.command[1:],
            cwd=self.cwd,
            env=self.env,
        )
        self._conn, self._process = await self._ctx.__aenter__()

        # Initialize
        init_resp = await self._conn.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(
                    read_text_file=True,
                    write_text_file=True,
                ),
                terminal=True,
            ),
            client_info=Implementation(name="koi", version="0.1.0"),
        )

        # Create session
        session_resp = await self._conn.new_session(cwd=self.cwd)
        self.session_id = session_resp.session_id
        return self.session_id

    async def send(self, message: str, timeout: float = 300.0) -> ACPResult:
        """Send a prompt and wait for the complete response."""
        if not self._conn or not self.session_id:
            raise RuntimeError("Session not started. Call start() first.")

        self._client.reset()

        prompt_content = [TextContentBlock(type="text", text=message)]

        try:
            resp = await asyncio.wait_for(
                self._conn.prompt(prompt=prompt_content, session_id=self.session_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ACPResult(
                content="",
                stop_reason="timeout",
            )

        return ACPResult(
            content=self._client._collected_text,
            stop_reason=resp.stop_reason if hasattr(resp, 'stop_reason') else "end_turn",
            tool_calls=self._client._tool_calls,
            thoughts=self._client._collected_thoughts,
            usage=self._client._usage,
        )

    async def cancel(self):
        """Cancel the current prompt turn."""
        if self._conn and self.session_id:
            await self._conn.cancel(session_id=self.session_id)

    async def close(self):
        """Gracefully close the session and kill the process."""
        if self._ctx:
            try:
                await self._ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if self._process and self._process.returncode is None:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
        self._conn = None
        self._process = None

    async def kill(self):
        """Force kill without graceful shutdown."""
        if self._process and self._process.returncode is None:
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
        self._conn = None
        self._process = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.returncode is None
