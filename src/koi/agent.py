"""Main agent implementation with conversation loop."""

import json
import signal
from typing import List, Dict, Any
from rich.console import Console
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from .config import Config
from .llm import LLMClient
from .memory import Memory
from .skills import SkillsManager
from .sandbox import Sandbox
from .tools import ToolExecutor, get_tool_definitions
from .prompts import build_system_prompt, build_tool_result_message
from .compaction import ContextCompactor

console = Console()


class Agent:
    """Main agent class that handles conversation and tool execution."""
    
    def __init__(self, config: Config, non_interactive: bool = False):
        """Initialize agent with configuration."""
        self.config = config
        self.llm_client = LLMClient(config)
        self.memory = Memory()
        self.skills_manager = SkillsManager(config.skills_paths)
        self.sandbox = Sandbox()
        self.tool_executor = ToolExecutor(self.skills_manager, self.sandbox)
        self.compactor = ContextCompactor(self.llm_client, config.context_window)

        self.messages: List[Dict[str, Any]] = []
        self.running = False
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

            self._prompt_session = PromptSession(key_bindings=bindings, multiline=True)

        # Initialize with system prompt
        system_prompt = build_system_prompt(config, non_interactive=non_interactive)
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    async def run_interactive(self):
        """Run interactive agent session."""
        self.running = True
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        console.print("🐠 [bold cyan]Koi Agent[/bold cyan] - Ready to help!", style="bold")
        console.print("Type '/exit' to quit, '/help' for commands, Option+Enter (Alt+Enter) for newline\n")
        
        try:
            while self.running:
                # Get user input
                try:
                    user_input = (await self._prompt_session.prompt_async("koi> ")).strip()
                except KeyboardInterrupt:
                    console.print("\n👋 Goodbye!", style="yellow")
                    break
                except EOFError:
                    console.print("\n👋 Goodbye!", style="yellow")
                    break
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                    continue
                
                # Add user message
                self.messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Process with agent loop
                await self._agent_loop()
        
        finally:
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
            # Process with agent loop
            await self._agent_loop(non_interactive=non_interactive)
        
        finally:
            await self.llm_client.close()
    
    async def _agent_loop(self, non_interactive: bool = False):
        """Main agent thinking loop."""
        tools = get_tool_definitions()
        if non_interactive:
            # Hide cron tools in non-interactive mode to prevent recursive scheduling
            cron_tool_names = {"add_cron_job", "list_cron_jobs", "remove_cron_job"}
            tools = [t for t in tools if t["function"]["name"] not in cron_tool_names]
        
        while True:
            # Check if compaction is needed
            if self.compactor.needs_compaction(self.messages):
                if not non_interactive:
                    console.print("🔄 Compacting conversation history...", style="yellow")
                
                self.messages = await self.compactor.compact_messages(self.messages)
            
            # Get response from LLM
            try:
                if non_interactive:
                    # Non-interactive mode for cron jobs
                    response = await self.llm_client.chat(self.messages, tools=tools)
                else:
                    # Interactive mode with spinner
                    with console.status("Thinking...", spinner="dots"):
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
                            # Show tool name and arguments
                            try:
                                args = json.loads(tool_call["function"]["arguments"])
                                args_summary = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
                                console.print(f"🔧 {func_name}({args_summary})", style="dim cyan")
                            except Exception:
                                console.print(f"🔧 {func_name}...", style="dim cyan")
                        
                        # Execute tool
                        if non_interactive:
                            result = await self.tool_executor.execute_tool(tool_call)
                        else:
                            with console.status(f"Running {func_name}...", spinner="dots"):
                                result = await self.tool_executor.execute_tool(tool_call)
                        
                        # Show errors and key results to the user
                        if not non_interactive:
                            if not result.get("success", True):
                                error_msg = result.get("error", "")
                                if not error_msg:
                                    # Fall back to stderr/stdout for commands that failed
                                    error_msg = (result.get("stderr") or result.get("stdout") or "Unknown error").strip()
                                # Truncate long errors but show enough to be useful
                                if len(error_msg) > 300:
                                    error_msg = error_msg[:300] + "..."
                                console.print(f"  ❌ {error_msg}", style="red")
                            elif result.get("exit_code", 0) != 0:
                                console.print(f"  ⚠️ Exit code {result['exit_code']}", style="yellow")
                                if result.get("stderr"):
                                    console.print(f"  {result['stderr'][:200]}", style="dim red")
                        
                        # Add tool result message
                        tool_result_msg = build_tool_result_message(tool_call, result)
                        self.messages.append(tool_result_msg)
                    
                    # Continue the loop to let model process tool results
                    continue
                
                else:
                    # Final response
                    if message.get("content"):
                        if not non_interactive:
                            console.print()  # Add spacing
                        else:
                            # Print response for cron jobs
                            print(message["content"])
                    
                    # Add assistant message
                    self.messages.append(message)
                    break
            
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                console.print(error_msg, style="red")
                if non_interactive:
                    print(error_msg)
                break
    
    async def _stream_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get response from LLM and display it."""
        try:
            response = await self.llm_client.chat(messages, tools=tools, stream=False)

            # If it's a text response, print it nicely
            if response.get("choices"):
                msg = response["choices"][0]["message"]
                if msg.get("content") and not msg.get("tool_calls"):
                    console.print(Text(msg["content"]))

            return response

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
                console.print(memory_content)
            else:
                console.print("Memory is empty.", style="yellow")
        
        elif cmd.startswith('/remember '):
            text = command[10:]  # Remove '/remember '
            self.memory.append(text)
            console.print("✅ Added to memory", style="green")
        
        elif cmd == '/compact':
            if len(self.messages) > 3:
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
        
        elif cmd == '/stats':
            stats = self.compactor.get_context_stats(self.messages)
            console.print(f"[bold blue]Context Statistics:[/bold blue]")
            console.print(f"Messages: {stats['message_count']}")
            console.print(f"Estimated tokens: {stats['estimated_tokens']}")
            console.print(f"Context usage: {stats['usage_percent']}%")
            console.print(f"Needs compaction: {stats['needs_compaction']}")
        
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
- /stats          - Show context statistics

[cyan]Usage:[/cyan]
Just type your requests normally and I'll help you with tasks using available tools.
"""
        console.print(help_text)
    
    def _signal_handler(self, signum, frame):
        """Handle SIGINT (Ctrl+C). First press: graceful, second: force exit."""
        if not self.running:
            # Already shutting down — force exit immediately
            console.print("\n👋 Force quit.", style="yellow")
            raise SystemExit(0)
        console.print("\n🔄 Shutting down... (press Ctrl+C again to force quit)", style="yellow")
        self.running = False