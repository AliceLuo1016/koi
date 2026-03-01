"""Main agent implementation with conversation loop."""

import asyncio
import json
import re
import signal
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.text import Text
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PtStyle

from .config import Config
from .llm import LLMClient
from .memory import Memory
from .skills import SkillsManager
from .sandbox import Sandbox
from .subagent import SubagentManager
from .tools import ToolExecutor, get_tool_definitions
from .prompts import build_system_prompt, build_tool_result_message
from .compaction import ContextCompactor
from .context_pruning import prune_context
from .context_guard import enforce_context_budget
from .usage import log_usage, estimate_cost

console = Console()


def _fmt_num(n: int) -> str:
    """Format a number with human-friendly suffixes (k, M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def strip_thinking_tags(text: str) -> Tuple[str, str]:
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
        self.tool_executor = ToolExecutor(
            self.skills_manager, self.sandbox, self.subagent_manager
        )
        self.compactor = ContextCompactor(self.llm_client, config.context_window)

        self.messages: List[Dict[str, Any]] = []
        self.running = False
        self._current_task: Optional[asyncio.Task] = None
        self._prompt_session = None

        if not non_interactive:
            # Set up prompt_toolkit session with custom key bindings
            # multiline=True enables multi-line paste support
            # Enter submits, Alt+Enter inserts a newline
            bindings = KeyBindings()

            @bindings.add('enter')
            def _(event):
                event.current_buffer.validate_and_handle()

            @bindings.add('escape', 'enter')
            def _(event):
                event.current_buffer.newline()

            self._prompt_style = PtStyle.from_dict({
                'prompt': '#6cb6ff bold',     # soft blue prompt
                '': '#e0e0e0',                # light gray user text
            })
            self._prompt_session = PromptSession(key_bindings=bindings, multiline=True, style=self._prompt_style)

        self._pending_subagent_results: List[Dict[str, Any]] = []

        # System prompt stored separately — never in the messages array.
        # Injected into the API payload at call time by LLMClient.
        self.system_prompt = build_system_prompt(
            config,
            non_interactive=non_interactive,
            use_reasoning_tags=self.llm_client.use_reasoning_tags,
        )
    
    async def run_interactive(self):
        """Run interactive agent session."""
        self.running = True

        # Use signal.signal (not loop.add_signal_handler) so that:
        # 1. raise KeyboardInterrupt properly interrupts prompt_toolkit
        # 2. task.cancel() fires immediately from the main thread
        prev_handler = signal.signal(signal.SIGINT, self._handle_sigint)

        console.print("🐠 [bold cyan]Koi Agent[/bold cyan] - Ready to help!", style="bold")
        console.print("Type '/exit' to quit, '/help' for commands, Option+Enter (Alt+Enter) for newline\n")

        try:
            while self.running:
                # Get user input
                try:
                    user_input = (await self._prompt_session.prompt_async(HTML('<prompt>koi&gt; </prompt>'))).strip()
                except KeyboardInterrupt:
                    console.print("\n👋 Goodbye!", style="yellow")
                    break
                except EOFError:
                    console.print("\n👋 Goodbye!", style="yellow")
                    break

                if not user_input:
                    continue

                # Handle special commands (but not file paths like /home/...)
                _COMMANDS = {'exit', 'quit', 'help', 'memory', 'remember', 'compact', 'skills', 'status', 'stats', 'usage', 'new', 'reset'}
                if user_input.startswith('/'):
                    cmd_word = user_input.split()[0][1:].lower()  # e.g. '/exit' -> 'exit'
                    if cmd_word in _COMMANDS:
                        await self._handle_command(user_input)
                        continue

                # Add user message
                self.messages.append({
                    "role": "user",
                    "content": user_input
                })

                # Run agent loop as a cancellable task
                self._current_task = asyncio.create_task(self._agent_loop())
                try:
                    await self._current_task
                except (asyncio.CancelledError, KeyboardInterrupt):
                    # Roll back partial messages from the interrupted iteration only;
                    # completed iterations are preserved.
                    snapshot = getattr(self, "_iter_msg_snapshot", len(self.messages))
                    self.messages = self.messages[:snapshot]
                    console.print("[dim]Operation cancelled.[/dim]")
                finally:
                    # Ensure the background task is cleaned up
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    self._current_task = None

        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if self.llm_client.usage.total_requests > 0:
                console.print()
                console.print(self.llm_client.usage.summary(self.config.model), style="dim")
                log_usage(self.llm_client.usage, self.config.model, Path(".koi"))
            await self.llm_client.close()
    
    async def run_task(self, task: str, non_interactive: bool = False):
        """Run a specific task (for cron jobs)."""
        if non_interactive:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*60}")
            print(f"[{timestamp}] Cron task started: {task}")
            print(f"{'='*60}")
        else:
            console.print(f"🐠 [bold cyan]Koi Agent[/bold cyan] - Running task: {task}")

        # Add task as user message
        self.messages.append({
            "role": "user",
            "content": task
        })

        try:
            # Run agent loop as a cancellable task
            self._current_task = asyncio.create_task(
                self._agent_loop(non_interactive=non_interactive)
            )
            try:
                await self._current_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                snapshot = getattr(self, "_iter_msg_snapshot", len(self.messages))
                self.messages = self.messages[:snapshot]
                if not non_interactive:
                    console.print("[dim]Operation cancelled.[/dim]")
            finally:
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                self._current_task = None

        finally:
            if self.llm_client.usage.total_requests > 0:
                log_usage(self.llm_client.usage, self.config.model, Path(".koi"))
            await self.llm_client.close()

    async def _agent_loop(self, non_interactive: bool = False):
        """Main agent thinking loop."""
        tools = get_tool_definitions()
        if non_interactive:
            # Hide cron tools in non-interactive mode to prevent recursive scheduling
            cron_tool_names = {"add_cron_job", "list_cron_jobs", "remove_cron_job"}
            tools = [t for t in tools if t["function"]["name"] not in cron_tool_names]
        
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
            self.messages = prune_context(
                self.messages, self.config.context_window
            )

            # Check if compaction is needed
            if self.compactor.needs_compaction(self.messages):
                if not non_interactive:
                    console.print("🔄 Compacting conversation history...", style="yellow")
                
                self.messages = await self.compactor.compact_messages(self.messages)

            # Final context window guard before LLM call
            self.messages = enforce_context_budget(
                self.messages, self.config.context_window
            )

            # Get response from LLM
            try:
                if non_interactive:
                    # Non-interactive mode for cron jobs
                    response = await self.llm_client.chat(
                        self.messages, tools=tools, system_prompt=self.system_prompt
                    )
                else:
                    # Interactive mode with streaming display
                    response = await self._stream_response(self.messages, tools)
                
                if not response.get("choices"):
                    console.print("❌ No response from LLM", style="red")
                    break
                
                message = response["choices"][0]["message"]
                
                # Check if model wants to use tools
                if message.get("tool_calls"):
                    # Add assistant message with tool calls
                    self.messages.append(message)
                    
                    # Execute tools
                    for tool_call in message["tool_calls"]:
                        func_name = tool_call["function"]["name"]
                        if not non_interactive:
                            # Show tool call with styled name
                            try:
                                args = json.loads(tool_call["function"]["arguments"])
                                args_summary = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
                                console.print(f"  ● [cyan]{func_name}[/cyan]([dim]{args_summary}[/dim])")
                            except Exception:
                                console.print(f"  ● [cyan]{func_name}[/cyan]")
                        
                        # Execute tool
                        if non_interactive:
                            result = await self.tool_executor.execute_tool(tool_call)
                        else:
                            with console.status(f"  Running {func_name}...", spinner="dots"):
                                result = await self.tool_executor.execute_tool(tool_call)
                        
                        # Show errors and key results to the user
                        if not non_interactive:
                            if not result.get("success", True):
                                error_msg = result.get("error", "")
                                if not error_msg:
                                    error_msg = (result.get("stderr") or result.get("stdout") or "").strip()
                                    if not error_msg:
                                        error_msg = f"{func_name} failed without providing details"
                                if len(error_msg) > 300:
                                    error_msg = error_msg[:300] + "..."
                                console.print(f"    [red]✗[/red] {error_msg}", style="red")
                            elif result.get("exit_code", 0) != 0:
                                console.print(f"    [yellow]⚠[/yellow] Exit code {result['exit_code']}", style="yellow")
                                if result.get("stderr"):
                                    console.print(f"    {result['stderr'][:200]}", style="dim red")
                            else:
                                if func_name == "edit_file" and result.get("message"):
                                    msg_lines = result["message"].splitlines()
                                    if len(msg_lines) > 1:
                                        console.print(f"    [green]✓[/green] {msg_lines[0]}")
                                        for line in msg_lines[1:]:
                                            if line.startswith("-"):
                                                console.print(f"      {line}", style="red")
                                            elif line.startswith("+"):
                                                console.print(f"      {line}", style="green")
                                            else:
                                                console.print(f"      {line}", style="dim")
                                    else:
                                        console.print(f"    [green]✓[/green] {result['message']}")
                                elif func_name in ["write_file", "create_alert", "update_memory"] and result.get("message"):
                                    console.print(f"    [green]✓[/green] {result['message']}")
                        
                        # Add tool result message
                        tool_result_msg = build_tool_result_message(tool_call, result, self.config.context_window)
                        self.messages.append(tool_result_msg)
                    
                    # Continue the loop to let model process tool results
                    continue
                
                else:
                    # Final text response
                    content = message.get("content")
                    self.messages.append(message)
                    if non_interactive and content:
                        # Non-interactive: display text (strip tags if needed)
                        if self.llm_client.use_reasoning_tags:
                            visible, _ = strip_thinking_tags(content)
                            if visible:
                                print(visible)
                        else:
                            print(content)
                    # Interactive display was already handled by _stream_response
                    break
            
            except asyncio.CancelledError:
                raise

            except Exception as e:
                error_msg = f"❌ Error: {e}"
                console.print(error_msg, style="red")
                if non_interactive:
                    print(error_msg)
                break
    
    async def _stream_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stream response from LLM and display tokens progressively."""
        try:
            collected_text = ""
            first_token = True

            async for token in self.llm_client.stream_chat(
                messages, tools=tools, system_prompt=self.system_prompt
            ):
                collected_text += token
                if not self.llm_client.use_reasoning_tags:
                    if first_token:
                        console.print()  # blank line before response
                        console.file.write("  ")  # indent agent response
                        first_token = False
                    console.file.write(token)
                    console.file.flush()

            response = self.llm_client._last_stream_response
            if response is None:
                response = {
                    "choices": [{"message": {"role": "assistant"}, "finish_reason": "stop"}]
                }

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
            raise

        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")
    
    async def _handle_command(self, command: str):
        """Handle special commands."""
        cmd = command.lower()
        
        if cmd == '/exit' or cmd == '/quit':
            self.running = False
            console.print("👋 Goodbye!", style="yellow")
        
        elif cmd == '/help':
            self._show_help()
        
        elif cmd == '/memory':
            memory_content = self.memory.load()
            if memory_content.strip():
                console.print("[bold blue]Current Memory:[/bold blue]")
                console.print(Markdown(memory_content))
            else:
                console.print("Memory is empty.", style="yellow")
        
        elif cmd.startswith('/remember '):
            text = command[10:]  # Remove '/remember '
            self.memory.append(text)
            console.print("✅ Added to memory", style="green")
        
        elif cmd == '/compact':
            if len(self.messages) > 2:
                console.print("🔄 Compacting conversation...", style="yellow")
                self.messages = await self.compactor.compact_messages(self.messages)
                console.print("✅ Conversation compacted", style="green")
            else:
                console.print("Not enough messages to compact.", style="yellow")
        
        elif cmd == '/skills':
            skills = self.skills_manager.list_skills()
            if skills:
                console.print("[bold blue]Available Skills:[/bold blue]")
                for skill in skills:
                    console.print(f"- [cyan]{skill['name']}[/cyan]: {skill['description']}")
            else:
                console.print("No skills found.", style="yellow")
        
        elif cmd == '/status' or cmd == '/stats':
            self._show_status()

        elif cmd == '/usage':
            console.print(self.llm_client.usage.summary(self.config.model))

        elif cmd == '/new' or cmd == '/reset':
            self.messages.clear()
            self.compactor.compaction_count = 0
            console.print("🆕 New session started. Context cleared.", style="green")

        else:
            console.print(f"Unknown command: {command}", style="red")
    
    def _show_help(self):
        """Show help information."""
        help_text = """[bold blue]Koi Agent Commands:[/bold blue]

[cyan]Chat Commands:[/cyan]
- /exit, /quit     - Exit the agent
- /help           - Show this help
- /memory         - Show current memory
- /remember TEXT  - Add text to memory
- /skills         - List available skills
- /compact        - Force conversation compaction
- /status         - Show status card (model, tokens, cache, context)
- /stats          - Alias for /status
- /usage          - Show detailed token usage and estimated cost
- /new, /reset    - Start a new session (clear context)

[cyan]Usage:[/cyan]
Just type your requests normally and I'll help you with tasks using available tools.
"""
        console.print(help_text)
    
    def _show_status(self):
        """Show rich status card with model, tokens, cache, context, and runtime info."""
        # Version
        console.print("\U0001f420 [bold cyan]Koi[/bold cyan] v0.1.0")

        # Model + masked key
        key = self.config.api_key
        if len(key) > 10:
            masked = key[:6] + "..." + key[-4:]
        else:
            masked = "***"
        console.print(
            f"\U0001f9e0 Model: {self.config.model} \u00b7 \U0001f511 {masked} ({self.config.api_format})"
        )

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
        console.print(
            f"\U0001f9ee Tokens: {_fmt_num(u.input_tokens)} in / {_fmt_num(u.output_tokens)} out{cost_str}"
        )

        # Cache (only show if any cache activity)
        total_input = u.input_tokens + u.cache_read_tokens + u.cache_creation_tokens
        if u.cache_read_tokens or u.cache_creation_tokens:
            hit_pct = (
                int(u.cache_read_tokens / total_input * 100) if total_input > 0 else 0
            )
            console.print(
                f"\U0001f5c4\ufe0f  Cache: {hit_pct}% hit \u00b7 {_fmt_num(u.cache_read_tokens)} cached, {_fmt_num(u.cache_creation_tokens)} new"
            )

        # Context
        stats = self.compactor.get_context_stats(self.messages)
        ctx_tokens = stats["estimated_tokens"]
        ctx_max = self.config.context_window
        ctx_pct = stats["usage_percent"]
        compactions = self.compactor.compaction_count
        console.print(
            f"\U0001f4da Context: {_fmt_num(ctx_tokens)}/{_fmt_num(ctx_max)} ({ctx_pct}%) \u00b7 \U0001f9f9 Compactions: {compactions}"
        )

        # Runtime
        think = self.config.thinking_level
        cache_status = "on" if self.config.prompt_caching else "off"
        console.print(
            f"\u2699\ufe0f  Runtime: {self.config.api_format} \u00b7 Think: {think} \u00b7 Prompt cache: {cache_status}"
        )

        # Sub-agents
        active = len(
            [r for r in self.subagent_manager.active_runs.values() if not r.completed]
        )
        if active > 0:
            console.print(f"\U0001f916 Sub-agents: {active} active")

    def _handle_sigint(self, signum, frame):
        """Handle SIGINT (Ctrl+C).

        Always raises KeyboardInterrupt for immediate stack unwinding.
        If an agent task is running, also cancel it so the background
        coroutine cleans up (subprocess termination, etc.).
        """
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        raise KeyboardInterrupt

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
                f"[Sub-agent '{label}' (id={run.id}) completed]\n"
                f"Exit code: {run.exit_code}\n"
                f"Result: {summary[:1000]}"
            ),
        }
        self._pending_subagent_results.append(msg)