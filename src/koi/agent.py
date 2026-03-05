"""Main agent implementation with conversation loop."""

import asyncio
import atexit
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PtStyle
from rich.console import Console
from rich.markdown import Markdown

from .compaction import ContextCompactor
from .config import Config
from .context_guard import enforce_context_budget
from .context_pruning import prune_context
from .errors import (
    KoiAPIError,
    KoiAuthError,
    KoiBillingError,
    KoiConnectionError,
    KoiContextOverflowError,
    KoiOverloadedError,
    KoiRateLimitError,
    KoiServerError,
)
from .llm import LLMClient
from .memory import Memory
from .memory_search import MemorySearchManager
from .prompts import build_system_prompt, build_tool_result_message
from .sandbox import Sandbox
from .session_manager import SessionManager
from .skills import SkillsManager
from .subagent import SubagentManager
from .tools import ToolExecutor, get_tool_definitions
from .transcript import TranscriptLogger
from .usage import estimate_cost, log_usage

console = Console()


def _fmt_num(n: int) -> str:
    """Format a number with human-friendly suffixes (k, M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def strip_thinking_tags(text: str) -> tuple[str, str]:
    """Strip <think>...</think> blocks and extract <final>...</final> content.

    Returns (visible_text, thinking_text) where:
    - visible_text: content to display to the user
    - thinking_text: concatenated content from all <think> blocks

    Logic:
    - Collect all <think>...</think> content into thinking_text
    - If <final>...</final> blocks exist, visible_text is their content
    - Otherwise, visible_text is everything outside <think> blocks
    """
    if not text:
        return ("", "")

    # Extract all <think>...</think> blocks (non-greedy, dotall)
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_matches = think_pattern.findall(text)
    thinking_text = "\n".join(m.strip() for m in think_matches if m.strip())

    # Extract all <final>...</final> blocks
    final_pattern = re.compile(r"<final>(.*?)</final>", re.DOTALL)
    final_matches = final_pattern.findall(text)

    if final_matches:
        # Use only <final> content
        visible_text = "\n".join(m.strip() for m in final_matches if m.strip())
    else:
        # No <final> tags — strip <think> blocks and show the rest
        visible_text = think_pattern.sub("", text).strip()

    return (visible_text, thinking_text)


class Agent:
    """Main agent class that handles conversation and tool execution."""

    def __init__(self, config: Config, non_interactive: bool = False):
        """Initialize agent with configuration."""
        self.config = config
        self.llm_client = LLMClient(config)
        self.memory = Memory()
        self.skills_manager = SkillsManager(config.skills_paths)
        self.sandbox = Sandbox()
        self.subagent_manager = SubagentManager(config)
        self.subagent_manager._on_complete = self._on_subagent_complete

        # Memory search (graceful if no API key — falls back to keyword-only)
        try:
            ms_provider = (
                getattr(config, "memory_search_provider", "openai") or "openai"
            )
            ms_model = (
                getattr(config, "memory_search_model", "text-embedding-3-small")
                or "text-embedding-3-small"
            )
            ms_api_key = getattr(config, "memory_search_api_key", "") or ""
            ms_api_base = getattr(config, "memory_search_api_base", "") or ""
            # Ensure values are actually strings (not Mock objects)
            if not isinstance(ms_provider, str):
                ms_provider = "openai"
            if not isinstance(ms_model, str):
                ms_model = "text-embedding-3-small"
            if not isinstance(ms_api_key, str):
                ms_api_key = ""
            if not isinstance(ms_api_base, str):
                ms_api_base = ""
            if not ms_api_key:
                ms_api_key = os.getenv("KOI_API_KEY", "")
            if not ms_api_key and ms_provider == "openai":
                api_key_val = getattr(config, "api_key", "")
                if isinstance(api_key_val, str):
                    ms_api_key = api_key_val
            koi_dir = Path.cwd() / ".koi"

            def _bool_attr(name: str, default: bool) -> bool:
                val = getattr(config, name, default)
                return val if isinstance(val, bool) else default

            def _float_attr(name: str, default: float) -> float:
                val = getattr(config, name, default)
                return val if isinstance(val, int | float) else default

            def _int_attr(name: str, default: int) -> int:
                val = getattr(config, name, default)
                return val if isinstance(val, int) else default

            self.memory_search_manager = MemorySearchManager(
                koi_dir=koi_dir,
                provider=ms_provider,
                model=ms_model,
                api_key=ms_api_key,
                api_base=ms_api_base,
                hybrid_enabled=_bool_attr("memory_search_hybrid_enabled", True),
                vector_weight=_float_attr("memory_search_hybrid_vector_weight", 0.7),
                text_weight=_float_attr("memory_search_hybrid_text_weight", 0.3),
                temporal_decay_enabled=_bool_attr(
                    "memory_search_temporal_decay_enabled", True
                ),
                temporal_decay_half_life_days=_int_attr(
                    "memory_search_temporal_decay_half_life_days", 30
                ),
                mmr_enabled=_bool_attr("memory_search_mmr_enabled", True),
                mmr_lambda=_float_attr("memory_search_mmr_lambda", 0.7),
                cache_enabled=_bool_attr("memory_search_cache_enabled", True),
                cache_max_entries=_int_attr("memory_search_cache_max_entries", 50000),
            )
        except Exception:
            self.memory_search_manager = None

        self.tool_executor = ToolExecutor(
            self.skills_manager,
            self.sandbox,
            self.subagent_manager,
            memory_search_manager=self.memory_search_manager,
        )
        self.compactor = ContextCompactor(self.llm_client, config.context_window)

        self.messages: list[dict[str, Any]] = []
        self.running = False
        self._current_task: asyncio.Task | None = None
        self._prompt_session = None
        self._memory_flushed = False

        if not non_interactive:
            # Set up prompt_toolkit session with custom key bindings
            # multiline=True enables multi-line paste support
            # Enter submits, Alt+Enter inserts a newline
            bindings = KeyBindings()

            @bindings.add("enter")
            def _(event):
                event.current_buffer.validate_and_handle()

            @bindings.add("escape", "enter")
            def _(event):
                event.current_buffer.newline()

            self._prompt_style = PtStyle.from_dict(
                {
                    "prompt": "#6cb6ff bold",  # soft blue prompt
                    "": "#e0e0e0",  # light gray user text
                }
            )
            self._prompt_session = PromptSession(
                key_bindings=bindings,
                multiline=True,
                style=self._prompt_style,
            )

        self._pending_subagent_results: list[dict[str, Any]] = []
        self._interrupted = False
        self._last_interrupt_time: float | None = None

        # System prompt stored separately — never in the messages array.
        # Injected into the API payload at call time by LLMClient.
        self.system_prompt = build_system_prompt(
            config,
            non_interactive=non_interactive,
            use_reasoning_tags=self.llm_client.use_reasoning_tags,
        )

        # Debug transcript logger
        import hashlib

        koi_dir = Path.cwd() / ".koi"
        self.transcript = TranscriptLogger(koi_dir, enabled=config.debug)
        self.transcript.log_session_start(
            {
                "model": config.model,
                "api_format": config.api_format,
                "system_prompt_hash": hashlib.sha256(
                    self.system_prompt.encode()
                ).hexdigest()[:16],
            }
        )

        # Session persistence (always on)
        self.session_manager = SessionManager(koi_dir)
        self._ephemeral = False  # Set True for --no-session mode

    async def run_interactive(self):
        """Run interactive agent session."""
        self.running = True

        # Use signal.signal (not loop.add_signal_handler) so that:
        # 1. raise KeyboardInterrupt properly interrupts prompt_toolkit
        # 2. task.cancel() fires immediately from the main thread
        prev_handler = signal.signal(signal.SIGINT, self._handle_sigint)

        # Register atexit handler for cleanup on force exit paths
        def _atexit_cleanup():
            self.subagent_manager.force_kill_all_sync()
            if self.llm_client.usage.total_requests > 0:
                log_usage(self.llm_client.usage, self.config.model, Path(".koi"))
            if not self._ephemeral:
                self.session_manager.close()

        atexit.register(_atexit_cleanup)

        self._print_session_header()
        if not self._ephemeral:
            self.session_manager.start_session(
                model=self.config.model,
                cwd=str(Path.cwd()),
            )
        console.print(
            "  Type [dim]/help[/dim] for commands,"
            " [dim]/exit[/dim] to quit,"
            " [dim]Alt+Enter[/dim] for newline"
        )
        console.print()

        try:
            while self.running:
                # Get user input
                try:
                    user_input = (
                        await self._prompt_session.prompt_async(
                            HTML("<prompt>koi&gt; </prompt>")
                        )
                    ).strip()
                except KeyboardInterrupt:
                    console.print("\n👋 Goodbye!", style="yellow")
                    break
                except EOFError:
                    console.print("\n👋 Goodbye!", style="yellow")
                    break

                if not user_input:
                    continue

                # Handle special commands (not file paths)
                _commands = {
                    "exit",
                    "quit",
                    "help",
                    "memory",
                    "remember",
                    "compact",
                    "skills",
                    "status",
                    "stats",
                    "usage",
                    "new",
                    "fork",
                }
                if user_input.startswith("/"):
                    cmd_word = user_input.split()[0][1:].lower()
                    if cmd_word in _commands:
                        await self._handle_command(user_input)
                        continue

                # Add user message
                user_msg = {"role": "user", "content": user_input}
                self.messages.append(user_msg)
                self.transcript.log_message("user_message", user_msg)
                if not self._ephemeral:
                    self.session_manager.save_message(user_msg)

                # Run agent loop as a cancellable task
                self._interrupted = False
                self._current_task = asyncio.create_task(self._agent_loop())
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    pass  # handled below via _interrupted flag
                finally:
                    # Ensure the background task is cleaned up
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    self._current_task = None

                if self._interrupted:
                    # Roll back partial messages from the interrupted iteration only;
                    # completed iterations are preserved.
                    snapshot = getattr(self, "_iter_msg_snapshot", len(self.messages))
                    self.messages = self.messages[:snapshot]
                    console.print("\n[dim]Operation cancelled.[/dim]")
                    self._interrupted = False

        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if self.llm_client.usage.total_requests > 0:
                console.print()
                console.print(
                    self.llm_client.usage.summary(self.config.model), style="dim"
                )
                log_usage(self.llm_client.usage, self.config.model, Path(".koi"))
            if not self._ephemeral:
                self.session_manager.close()
            await self.llm_client.close()

    async def run_task(self, task: str, non_interactive: bool = False):
        """Run a specific task (for cron jobs)."""
        if non_interactive:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'=' * 60}")
            print(f"[{timestamp}] Cron task started: {task}")
            print(f"{'=' * 60}")
        else:
            console.print(f"🐠 [bold cyan]Koi Agent[/bold cyan] - Running task: {task}")

        if not self._ephemeral:
            self.session_manager.start_session(
                model=self.config.model,
                cwd=str(Path.cwd()),
            )

        # Add task as user message
        task_msg = {"role": "user", "content": task}
        self.messages.append(task_msg)
        if not self._ephemeral:
            self.session_manager.save_message(task_msg)

        try:
            # Run agent loop as a cancellable task
            self._interrupted = False
            self._current_task = asyncio.create_task(
                self._agent_loop(non_interactive=non_interactive)
            )
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            finally:
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                self._current_task = None

            if self._interrupted:
                snapshot = getattr(self, "_iter_msg_snapshot", len(self.messages))
                self.messages = self.messages[:snapshot]
                if not non_interactive:
                    console.print("\n[dim]Operation cancelled.[/dim]")
                self._interrupted = False

        finally:
            if self.llm_client.usage.total_requests > 0:
                log_usage(self.llm_client.usage, self.config.model, Path(".koi"))
            await self.llm_client.close()

    async def run_pipe_mode(self):
        """Run in pipe mode: read JSON from stdin, write JSON to stdout.

        Used by persistent subagent sessions. The parent process communicates
        via newline-delimited JSON on stdin/stdout.

        Input:  {"type": "message", "content": "do X"}
        Output: {"type": "response", "content": "Done.", "usage": {...}}
        Shutdown: {"type": "shutdown"} or EOF on stdin
        """
        import sys

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break  # EOF

            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._pipe_write({"type": "error", "error": "Invalid JSON"})
                continue

            if msg.get("type") == "shutdown":
                break

            if msg.get("type") != "message":
                self._pipe_write(
                    {
                        "type": "error",
                        "error": f"Unknown type: {msg.get('type')}",
                    }
                )
                continue

            user_content = msg.get("content", "")
            if not user_content:
                self._pipe_write({"type": "error", "error": "Empty message"})
                continue

            # Add user message and run agent loop
            self.messages.append({"role": "user", "content": user_content})

            try:
                await self._agent_loop(non_interactive=True)
            except Exception as e:
                self._pipe_write({"type": "error", "error": str(e)})
                continue

            # Extract the last assistant message as the response
            response_text = ""
            for m in reversed(self.messages):
                if m.get("role") == "assistant" and m.get("content"):
                    response_text = m["content"]
                    break

            self._pipe_write(
                {
                    "type": "response",
                    "content": response_text,
                    "usage": self.llm_client.usage.to_dict(),
                }
            )

        # Cleanup
        if self.llm_client.usage.total_requests > 0:
            log_usage(self.llm_client.usage, self.config.model, Path(".koi"))
        await self.llm_client.close()

    def _pipe_write(self, data: dict):
        """Write a JSON line to stdout for pipe mode communication."""
        import sys

        sys.stdout.write(json.dumps(data) + "\n")
        sys.stdout.flush()

    async def _agent_loop(self, non_interactive: bool = False):
        """Main agent thinking loop."""
        tools = get_tool_definitions()
        if non_interactive:
            # Hide cron tools in non-interactive mode to prevent recursive scheduling
            cron_tool_names = {"add_cron_job", "list_cron_jobs", "remove_cron_job"}
            tools = [t for t in tools if t["function"]["name"] not in cron_tool_names]

        _overflow_retried = False

        while True:
            # Snapshot for fine-grained rollback on cancellation:
            # if cancelled mid-iteration, only the current iteration's
            # messages are rolled back (preserving completed iterations).
            self._iter_msg_snapshot = len(self.messages)

            # Inject pending sub-agent completion results
            if self._pending_subagent_results:
                self.messages.extend(self._pending_subagent_results)
                self._pending_subagent_results.clear()

            # Preemptive context pruning: trim/clear old tool results
            self.messages = prune_context(self.messages, self.config.context_window)

            # Check if compaction is needed
            if self.compactor.needs_compaction(self.messages):
                # Pre-compaction memory flush
                if not self._memory_flushed and getattr(
                    self.config, "compaction_memory_flush_enabled", True
                ):
                    await self._pre_compaction_memory_flush(tools)

                if not non_interactive:
                    console.print(
                        "🔄 Compacting conversation history...", style="yellow"
                    )

                try:
                    self.messages = await self.compactor.compact_messages(self.messages)
                    self._memory_flushed = False
                    if not self._ephemeral:
                        self.session_manager.save_compaction(
                            "Auto-compaction", tokens_before=0
                        )
                except asyncio.CancelledError:
                    # Compaction cancelled — keep original messages, re-raise
                    raise

            # Final context window guard before LLM call
            self.messages = enforce_context_budget(
                self.messages, self.config.context_window
            )

            # Get response from LLM
            try:
                if non_interactive:
                    # Non-interactive mode for cron jobs
                    response = await self.llm_client.chat(
                        self.messages,
                        tools=tools,
                        system_prompt=self.system_prompt,
                    )
                else:
                    # Interactive mode with streaming display
                    response = await self._stream_response(self.messages, tools)

                if not response.get("choices"):
                    console.print("No response from LLM", style="red")
                    break

                message = response["choices"][0]["message"]

                # Check if model wants to use tools
                if message.get("tool_calls"):
                    # Add assistant message with tool calls
                    self.messages.append(message)
                    if not self._ephemeral:
                        self.session_manager.save_message(message)

                    # Execute tools
                    for tool_call in message["tool_calls"]:
                        func_name = tool_call["function"]["name"]
                        if not non_interactive:
                            # Show tool call with styled name
                            try:
                                args = json.loads(tool_call["function"]["arguments"])
                                args_summary = ", ".join(
                                    f"{k}={repr(v)[:60]}" for k, v in args.items()
                                )
                                console.print(
                                    f"  [cyan]{func_name}[/cyan]"
                                    f"([dim]{args_summary}[/dim])"
                                )
                            except Exception:
                                console.print(f"  [cyan]{func_name}[/cyan]")

                        # Execute tool
                        if non_interactive:
                            result = await self.tool_executor.execute_tool(tool_call)
                        else:
                            with console.status(
                                f"  Running {func_name}...",
                                spinner="dots",
                            ):
                                result = await self.tool_executor.execute_tool(
                                    tool_call
                                )

                        # Show errors and key results
                        if not non_interactive:
                            if not result.get("success", True):
                                error_msg = result.get("error", "")
                                if not error_msg:
                                    error_msg = (
                                        result.get("stderr")
                                        or result.get("stdout")
                                        or ""
                                    ).strip()
                                    if not error_msg:
                                        error_msg = (
                                            f"{func_name} failed without details"
                                        )
                                if len(error_msg) > 300:
                                    error_msg = error_msg[:300] + "..."
                                console.print(
                                    f"    [red]x[/red] {error_msg}",
                                    style="red",
                                )
                            elif result.get("exit_code", 0) != 0:
                                console.print(
                                    f"    [yellow]![/yellow]"
                                    f" Exit code {result['exit_code']}",
                                    style="yellow",
                                )
                                if result.get("stderr"):
                                    console.print(
                                        f"    {result['stderr'][:200]}",
                                        style="dim red",
                                    )
                            else:
                                if func_name == "edit_file" and result.get("message"):
                                    msg_lines = result["message"].splitlines()
                                    if len(msg_lines) > 1:
                                        console.print(
                                            f"    [green]v[/green] {msg_lines[0]}"
                                        )
                                        for ln in msg_lines[1:]:
                                            if ln.startswith("-"):
                                                console.print(
                                                    f"      {ln}",
                                                    style="red",
                                                )
                                            elif ln.startswith("+"):
                                                console.print(
                                                    f"      {ln}",
                                                    style="green",
                                                )
                                            else:
                                                console.print(
                                                    f"      {ln}",
                                                    style="dim",
                                                )
                                    else:
                                        console.print(
                                            f"    [green]v[/green] {result['message']}"
                                        )
                                elif func_name in [
                                    "write_file",
                                    "create_alert",
                                    "update_memory",
                                ] and result.get("message"):
                                    console.print(
                                        f"    [green]v[/green] {result['message']}"
                                    )

                        # Add tool result message
                        tool_result_msg = build_tool_result_message(
                            tool_call,
                            result,
                            self.config.context_window,
                        )
                        self.messages.append(tool_result_msg)
                        if not self._ephemeral:
                            self.session_manager.save_message(tool_result_msg)

                    # Continue the loop for tool results
                    continue

                else:
                    # Final text response
                    content = message.get("content")
                    self.messages.append(message)
                    if not self._ephemeral:
                        self.session_manager.save_message(message)
                    if non_interactive and content:
                        # Non-interactive: display text (strip tags if needed)
                        if self.llm_client.use_reasoning_tags:
                            visible, _ = strip_thinking_tags(content)
                            if visible:
                                print(visible)
                        else:
                            print(content)
                    # Interactive display handled by _stream_response
                    break

            except asyncio.CancelledError:
                raise

            except KoiAuthError as e:
                console.print(
                    "Auth failed. Check your API key.",
                    style="red",
                )
                if not non_interactive:
                    console.print(f"   {e}", style="dim red")
                break

            except KoiBillingError as e:
                console.print(
                    "Billing issue. Check your account.",
                    style="red",
                )
                if not non_interactive:
                    console.print(f"   {e}", style="dim red")
                break

            except KoiContextOverflowError as e:
                if not _overflow_retried:
                    console.print(
                        "Context too long, auto-compacting...",
                        style="yellow",
                    )
                    self.messages = await self.compactor.compact_messages(self.messages)
                    if not self._ephemeral:
                        self.session_manager.save_compaction(
                            "Overflow auto-compaction",
                            tokens_before=0,
                        )
                    _overflow_retried = True
                    continue
                else:
                    console.print(
                        "Context too long. Try /compact or /new.",
                        style="yellow",
                    )
                    if not non_interactive:
                        console.print(f"   {e}", style="dim yellow")
                    break

            except KoiRateLimitError as e:
                if e.retry_after:
                    console.print(
                        f"Rate limited. Server wants {e.retry_after:.0f}s wait.",
                        style="yellow",
                    )
                else:
                    console.print(
                        "Rate limited after retries. Wait a moment and try again.",
                        style="yellow",
                    )
                break

            except (KoiServerError, KoiOverloadedError) as e:
                console.print(
                    f"Provider issues ({e}). Try again later.",
                    style="yellow",
                )
                break

            except KoiConnectionError:
                console.print(
                    "Connection failed after retries. Check your network.",
                    style="red",
                )
                break

            except KoiAPIError as e:
                console.print(f"❌ API error: {e}", style="red")
                break

            except Exception as e:
                error_msg = f"❌ Unexpected error: {e}"
                console.print(error_msg, style="red")
                if non_interactive:
                    print(error_msg)
                break

    async def _pre_compaction_memory_flush(self, tools: list[dict[str, Any]]) -> None:
        """Run one LLM turn to let the agent save durable memories before compaction."""
        flush_messages = list(self.messages)
        flush_messages.append(
            {
                "role": "system",
                "content": "Session nearing compaction. Store durable memories now.",
            }
        )
        flush_messages.append(
            {
                "role": "user",
                "content": (
                    "Write any lasting notes to the daily memory log. "
                    "Reply with NO_REPLY if nothing to store."
                ),
            }
        )

        try:
            response = await self.llm_client.chat(
                flush_messages,
                tools=tools,
                system_prompt=self.system_prompt,
            )
            self._memory_flushed = True

            if response.get("choices"):
                message = response["choices"][0]["message"]
                # Execute any tool calls (e.g. update_memory)
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        func_name = tool_call["function"]["name"]
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                        except (json.JSONDecodeError, KeyError):
                            continue
                        try:
                            await self.tool_executor.execute(func_name, args)
                        except Exception:
                            pass
        except Exception:
            # Flush is best-effort; don't block compaction
            self._memory_flushed = True

    async def _stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Stream response from LLM and display tokens progressively."""
        spinner = None
        try:
            collected_text = ""
            full_content = ""
            tool_calls = {}  # index/id → tool call dict
            first_token = True

            # Show spinner while waiting for first token
            spinner = console.status("  [dim]Thinking...[/dim]", spinner="dots")
            spinner.start()

            async for event in self.llm_client.stream_chat(
                messages, tools=tools, system_prompt=self.system_prompt
            ):
                if event.type == "text_delta":
                    collected_text += event.delta
                    full_content += event.delta
                    if not self.llm_client.use_reasoning_tags:
                        if first_token:
                            spinner.stop()
                            spinner = None
                            console.print()  # blank line before response
                            console.file.write("  ")  # indent agent response
                            first_token = False
                        console.file.write(event.delta)
                        console.file.flush()

                elif event.type == "toolcall_start":
                    if spinner is None:
                        spinner = console.status(
                            "  [dim]Preparing tool call...[/dim]", spinner="dots"
                        )
                        spinner.start()
                    else:
                        spinner.update("  [dim]Preparing tool call...[/dim]")
                    idx = event.content_index
                    tool_calls[idx] = {
                        "id": event.tool_call_id or str(idx),
                        "type": "function",
                        "function": {"name": event.tool_name, "arguments": ""},
                    }

                elif event.type == "toolcall_delta":
                    idx = event.content_index
                    if idx in tool_calls:
                        tool_calls[idx]["function"]["arguments"] += event.delta

                elif event.type == "usage":
                    self.llm_client._extract_usage_from_event(event)

                # thinking_delta, *_start, *_end, done — skip

            # Stop spinner in case no tokens were received, or reasoning-tags mode
            if spinner:
                spinner.stop()
                spinner = None

            # Build response from accumulated events
            message = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
            response = {"choices": [{"message": message, "finish_reason": "stop"}]}

            # For reasoning tags mode: display after stripping tags
            if self.llm_client.use_reasoning_tags and collected_text:
                display_text, _ = strip_thinking_tags(collected_text)
                if display_text:
                    console.print()
                    console.print(Markdown(display_text))
            elif not self.llm_client.use_reasoning_tags and collected_text:
                console.file.write("\n")  # final newline after streamed text
                console.file.flush()

            return response

        except asyncio.CancelledError:
            # Ensure spinner is stopped so terminal isn't left dirty
            if spinner:
                spinner.stop()
            raise

        except KoiAPIError:
            if spinner:
                spinner.stop()
            raise  # Already classified

        except Exception as e:
            if spinner:
                spinner.stop()
            raise KoiAPIError(f"LLM request failed: {e}", retryable=False)

    async def _handle_command(self, command: str):
        """Handle special commands."""
        cmd = command.lower()

        if cmd == "/exit" or cmd == "/quit":
            self.running = False
            console.print("Goodbye!", style="yellow")

        elif cmd == "/help":
            self._show_help()

        elif cmd == "/memory":
            memory_content = self.memory.load()
            if memory_content.strip():
                console.print("[bold blue]Current Memory:[/bold blue]")
                console.print(Markdown(memory_content))
            else:
                console.print("Memory is empty.", style="yellow")

        elif cmd.startswith("/remember "):
            text = command[10:]  # Remove '/remember '
            self.memory.append(text)
            console.print("Added to memory", style="green")

        elif cmd == "/compact":
            if len(self.messages) > 2:
                console.print(
                    "Compacting conversation...",
                    style="yellow",
                )
                self.messages = await self.compactor.compact_messages(self.messages)
                if not self._ephemeral:
                    self.session_manager.save_compaction(
                        "Manual compaction",
                        tokens_before=0,
                    )
                console.print("Conversation compacted", style="green")
            else:
                console.print(
                    "Not enough messages to compact.",
                    style="yellow",
                )

        elif cmd == "/skills":
            skills = self.skills_manager.list_skills()
            if skills:
                console.print("[bold blue]Available Skills:[/bold blue]")
                for skill in skills:
                    console.print(
                        f"- [cyan]{skill['name']}[/cyan]: {skill['description']}"
                    )
            else:
                console.print("No skills found.", style="yellow")

        elif cmd == "/status" or cmd == "/stats":
            self._show_status()

        elif cmd == "/usage":
            # Current session usage
            console.print(self.llm_client.usage.summary(self.config.model))
            console.print()  # Empty line

            # 7-day history from usage log
            from .usage import get_usage_history

            history = get_usage_history(Path(".koi"), days=7)
            console.print(history)

        elif cmd == "/new":
            self.messages.clear()
            self.compactor.compaction_count = 0
            if not self._ephemeral:
                self.session_manager.close()
                self.session_manager.start_session(
                    model=self.config.model,
                    cwd=str(Path.cwd()),
                )
            console.print(
                "New session started. Context cleared.",
                style="green",
            )

        elif cmd == "/fork":
            if self._ephemeral:
                console.print(
                    "Cannot fork in ephemeral mode.",
                    style="yellow",
                )
            else:
                self.session_manager.fork()
                console.print(
                    "Forked session. New messages will branch from here.",
                    style="green",
                )

        else:
            console.print(
                f"Unknown command: {command}",
                style="red",
            )

    def _print_session_header(self):
        """Print a bordered session header card."""
        model = self.config.model
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home) :]
        skills_count = len(self.skills_manager.list_skills())

        lines = [
            ("", ""),
            ("🐠", " [bold cyan]Koi[/bold cyan]"),
            ("", ""),
            ("  model:  ", f" {model}"),
            ("  dir:    ", f" {cwd}"),
            ("  skills: ", f" {skills_count} loaded"),
            ("", ""),
        ]

        from rich.text import Text as RichText

        max_width = 0
        for label, value in lines:
            t = RichText.from_markup(label + value)
            if len(t) > max_width:
                max_width = len(t)
        max_width = max(max_width, 30)
        box_width = max_width + 4

        console.print(f"  [dim]╭{'─' * (box_width - 2)}╮[/dim]")
        for label, value in lines:
            t = RichText.from_markup(label + value)
            padding = max_width - len(t)
            console.print(f"  [dim]|[/dim] {label}{value}{' ' * padding} [dim]|[/dim]")
        console.print(f"  [dim]╰{'─' * (box_width - 2)}╯[/dim]")

    def _show_help(self):
        """Show help information."""
        help_text = (
            "[bold blue]Koi Agent Commands:[/bold blue]\n"
            "\n"
            "[cyan]Chat Commands:[/cyan]\n"
            "- /exit, /quit  - Exit the agent\n"
            "- /help         - Show this help\n"
            "- /memory       - Show current memory\n"
            "- /remember TXT - Add text to memory\n"
            "- /skills       - List available skills\n"
            "- /compact      - Force compaction\n"
            "- /status       - Show status card\n"
            "- /stats        - Alias for /status\n"
            "- /usage        - Token usage + cost\n"
            "- /new          - New session\n"
            "- /fork         - Fork session\n"
            "\n"
            "[cyan]Usage:[/cyan]\n"
            "Type your requests and I'll help"
            " using available tools.\n"
        )
        console.print(help_text)

    def _show_status(self):
        """Show status card with model, tokens, cache, etc."""
        # Version
        console.print("\U0001f420 [bold cyan]Koi[/bold cyan] v0.1.0")

        # Model + masked key
        key = self.config.api_key
        if len(key) > 10:
            masked = key[:6] + "..." + key[-4:]
        else:
            masked = "***"
        model = self.config.model
        fmt = self.config.api_format
        console.print(f"\U0001f9e0 Model: {model} \u00b7 \U0001f511 {masked} ({fmt})")

        # Tokens + cost
        u = self.llm_client.usage
        cost = estimate_cost(
            self.config.model,
            u.input_tokens,
            u.output_tokens,
            u.cache_read_tokens,
            u.cache_creation_tokens,
        )
        cost_str = f" \u00b7 \U0001f4b0 ${cost:.4f}" if cost > 0 else ""
        inp = _fmt_num(u.input_tokens)
        out = _fmt_num(u.output_tokens)
        console.print(f"\U0001f9ee Tokens: {inp} in / {out} out{cost_str}")

        # Cache (only show if any cache activity)
        total_input = u.input_tokens + u.cache_read_tokens + u.cache_creation_tokens
        if u.cache_read_tokens or u.cache_creation_tokens:
            hit_pct = (
                int(u.cache_read_tokens / total_input * 100) if total_input > 0 else 0
            )
            cached = _fmt_num(u.cache_read_tokens)
            created = _fmt_num(u.cache_creation_tokens)
            console.print(
                f"\U0001f5c4\ufe0f  Cache: {hit_pct}% hit"
                f" \u00b7 {cached} cached,"
                f" {created} new"
            )

        # Context
        stats = self.compactor.get_context_stats(self.messages)
        ctx_tokens = stats["estimated_tokens"]
        ctx_max = self.config.context_window
        ctx_pct = stats["usage_percent"]
        compactions = self.compactor.compaction_count
        ctx_t = _fmt_num(ctx_tokens)
        ctx_m = _fmt_num(ctx_max)
        console.print(
            f"\U0001f4da Context: {ctx_t}/{ctx_m}"
            f" ({ctx_pct}%)"
            f" \u00b7 \U0001f9f9 Compactions: {compactions}"
        )

        # Runtime
        think = self.config.thinking_level
        cache_status = "on" if self.config.prompt_caching else "off"
        api_fmt = self.config.api_format
        console.print(
            f"\u2699\ufe0f  Runtime: {api_fmt}"
            f" \u00b7 Think: {think}"
            f" \u00b7 Prompt cache: {cache_status}"
        )

        # Sub-agents
        active = len(
            [r for r in self.subagent_manager.active_runs.values() if not r.completed]
        )
        if active > 0:
            console.print(f"\U0001f916 Sub-agents: {active} active")

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT (Ctrl+C).

        Flag-based approach: sets _interrupted and cancels the running task
        but does NOT raise KeyboardInterrupt.  This avoids the race between
        KeyboardInterrupt and CancelledError that caused flaky behavior
        (interrupt landing inside httpx cleanup, Rich spinner, etc.).

        Double Ctrl+C within 1.5 seconds triggers immediate force exit via
        os._exit() for cases where graceful cancellation is stuck.

        When no agent task is running (i.e. we're at the prompt), we raise
        KeyboardInterrupt so prompt_toolkit can handle it normally.
        """
        now = time.time()

        if self._current_task and not self._current_task.done():
            # Double Ctrl+C: force exit if second interrupt within 1.5s
            if (
                self._last_interrupt_time is not None
                and now - self._last_interrupt_time < 1.5
            ):
                self._last_interrupt_time = now
                self._force_exit()
                return

            self._last_interrupt_time = now
            self._interrupted = True
            self._current_task.cancel()
            # Also close any active LLM stream to unblock httpx immediately
            self.llm_client.abort_stream()
        else:
            # At the prompt — let prompt_toolkit handle it
            raise KeyboardInterrupt

    def _force_exit(self):
        """Immediate termination for double Ctrl+C.

        Uses sys.stderr.write (signal-handler safe) and os._exit
        (bypasses Python cleanup) for reliable shutdown.
        """
        sys.stderr.write("\n⚡ Force exit.\n")
        sys.stderr.flush()
        os._exit(1)

    async def _on_subagent_complete(self, run) -> None:
        """Callback invoked when a sub-agent finishes."""
        summary = ""
        if run.result:
            summary = run.result.get("summary", run.result.get("response", ""))
        if run.error:
            summary = f"Error: {run.error}"
        if not summary and run.stdout:
            summary = run.stdout[:500]

        label = run.label or run.task[:60]
        msg = {
            "role": "system",
            "content": (
                f"[Sub-agent '{label}'"
                f" (id={run.id}) completed]\n"
                f"Exit code: {run.exit_code}\n"
                f"Result: {summary[:1000]}"
            ),
        }
        self._pending_subagent_results.append(msg)
        # Notify user immediately
        status = "[green]v[/green]" if run.exit_code == 0 else "[red]x[/red]"
        console.print(
            f"\n  {status} Sub-agent [cyan]{label}[/cyan] (id={run.id}) completed"
        )
        if summary:
            for ln in summary[:500].splitlines():
                ln = ln.strip()
                if ln:
                    console.print(f"    {ln}", style="dim")

        # Reprint prompt so user knows they can type
        console.file.write("koi> ")
        console.file.flush()

    def resume_from_session(self, session_path: Path) -> None:
        """Load messages from a saved session file."""
        self.session_manager = SessionManager(
            Path.cwd() / ".koi",
            session_path=session_path,
        )
        data = self.session_manager.load_session()
        self.messages = data["messages"]

        # Show what was loaded
        msg_count = len(self.messages)
        model = data["header"].get("model", "unknown") if data["header"] else "unknown"
        console.print(
            f"📂 Resumed session: {msg_count} messages (model: {model})",
            style="dim cyan",
        )
