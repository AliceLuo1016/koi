"""Sub-agent spawning and lifecycle management for koi."""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional
from uuid import uuid4

from .acp_client import ACPSession, ACPResult
from .acp_registry import get_agent


@dataclass
class SubagentRun:
    """State for a single sub-agent run."""

    id: str
    task: str
    label: Optional[str]
    process: asyncio.subprocess.Process
    result_file: Path
    started_at: datetime
    mode: str = "run"
    timeout_seconds: int = 0
    completed: bool = False
    last_activity: Optional[datetime] = None
    harness: str = "koi"         # "koi" (native) or "acp"
    agent_name: str = ""         # e.g. "claude-code", "codex"
    acp_session: Optional[Any] = None  # ACPSession instance for harness="acp"
    result: Optional[dict] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


class SubagentManager:
    """Spawn and manage isolated Koi sub-agent processes."""

    def __init__(
        self,
        config: Any,
        max_children: int = 5,
        max_depth: int = 3,
    ):
        self.config = config
        self.max_children = max_children
        self.max_depth = max_depth
        self.active_runs: Dict[str, SubagentRun] = {}
        self._depth = int(os.environ.get("KOI_SPAWN_DEPTH", "0"))
        self._on_complete: Optional[
            Callable[[SubagentRun], Coroutine[Any, Any, None]]
        ] = None

    # ── public API ──────────────────────────────────────────────

    async def spawn(
        self,
        task: str,
        label: Optional[str] = None,
        model: Optional[str] = None,
        thinking: Optional[str] = None,
        timeout_seconds: int = 0,
        cwd: Optional[str] = None,
    ) -> dict:
        """Spawn an isolated Koi sub-agent in the background.

        Returns a dict with ``status``, ``run_id``, and ``note`` on success,
        or ``status`` and ``error`` on failure.
        """
        # Depth guard
        if self._depth >= self.max_depth:
            return {
                "status": "error",
                "error": (
                    f"Max spawn depth reached ({self.max_depth}). "
                    "Sub-agents cannot spawn further."
                ),
            }

        # Children guard
        active_count = sum(
            1 for r in self.active_runs.values() if not r.completed
        )
        if active_count >= self.max_children:
            return {
                "status": "error",
                "error": f"Max children reached ({self.max_children})",
            }

        run_id = str(uuid4())[:8]

        # Result file lives under .koi/subagent-runs/
        result_file = Path(cwd or os.getcwd()) / ".koi" / "subagent-runs" / f"{run_id}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        # Build the command
        cmd: List[str] = [
            sys.executable,
            "-m",
            "koi",
            "run",
            "--task",
            task,
            "--non-interactive",
            "--result-file",
            str(result_file),
        ]
        if model:
            cmd.extend(["--model", model])
        if thinking:
            cmd.extend(["--thinking", thinking])

        env = {**os.environ, "KOI_SPAWN_DEPTH": str(self._depth + 1)}

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd or os.getcwd(),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        run = SubagentRun(
            id=run_id,
            task=task,
            label=label,
            process=process,
            result_file=result_file,
            started_at=datetime.now(),
            mode="run",
            timeout_seconds=timeout_seconds,
        )
        self.active_runs[run_id] = run

        # Fire-and-forget background waiter
        asyncio.create_task(self._wait_for_completion(run))

        return {
            "status": "accepted",
            "run_id": run_id,
            "note": "Sub-agent started. Result will be announced when done.",
        }

    async def spawn_acp_session(
        self,
        agent_name: str,
        label: str,
        cwd: Optional[str] = None,
        auto_approve: bool = True,
        idle_timeout: int = 1800,
    ) -> dict:
        """Spawn an ACP agent as a persistent session."""
        # Depth guard
        if self._depth >= self.max_depth:
            return {"status": "error", "error": f"Max spawn depth reached ({self.max_depth})."}

        active_count = sum(1 for r in self.active_runs.values() if not r.completed)
        if active_count >= self.max_children:
            return {"status": "error", "error": f"Max children reached ({self.max_children})"}

        # Label uniqueness
        for r in self.active_runs.values():
            if not r.completed and r.label == label and r.mode == "session":
                return {"status": "error", "error": f"Session with label '{label}' already exists"}

        agent = get_agent(agent_name)
        if not agent:
            return {"status": "error", "error": f"Unknown agent: {agent_name}"}
        if not agent.is_available():
            return {"status": "error", "error": f"Agent '{agent_name}' is not installed (binary '{agent.check_binary}' not found)"}

        run_id = str(uuid4())[:8]

        try:
            acp_sess = ACPSession(
                command=agent.command,
                cwd=cwd or os.getcwd(),
                auto_approve=auto_approve,
            )
            session_id = await acp_sess.start()
        except Exception as e:
            return {"status": "error", "error": f"Failed to start ACP agent '{agent_name}': {e}"}

        run = SubagentRun(
            id=run_id,
            task=f"[acp-session:{agent_name}:{label}]",
            label=label,
            process=acp_sess._process,
            result_file=Path(cwd or os.getcwd()) / ".koi" / "subagent-runs" / f"{run_id}.json",
            started_at=datetime.now(),
            mode="session",
            timeout_seconds=0,
            last_activity=datetime.now(),
            harness="acp",
            agent_name=agent_name,
            acp_session=acp_sess,
        )
        self.active_runs[run_id] = run

        asyncio.create_task(self._idle_watcher(run, idle_timeout))

        return {
            "status": "accepted",
            "run_id": run_id,
            "label": label,
            "agent": agent_name,
            "acp_session_id": session_id,
            "note": f"ACP session '{label}' started with {agent.display_name}. Use send_to_subagent to communicate.",
        }

    async def send_acp(self, target: str, message: str, timeout: float = 300.0) -> dict:
        """Send message to an ACP session."""
        run = self._find_session(target)
        if not run:
            return {"error": f"No active session found for '{target}'"}
        if run.completed:
            return {"error": f"Session '{target}' has ended"}
        if run.harness != "acp" or not run.acp_session:
            return {"error": f"'{target}' is not an ACP session"}

        run.last_activity = datetime.now()

        try:
            result = await run.acp_session.send(message, timeout=timeout)
            if result.stop_reason == "timeout":
                await self._kill_run(run, f"ACP prompt timed out after {timeout}s")
                return {"error": f"ACP prompt timed out after {timeout}s. Session killed."}
        except Exception as e:
            await self._kill_run(run, str(e))
            return {"error": f"ACP send failed: {e}"}

        run.last_activity = datetime.now()
        return {
            "type": "response",
            "content": result.content,
            "stop_reason": result.stop_reason,
            "tool_calls": result.tool_calls,
            "thoughts": result.thoughts,
            "usage": result.usage,
        }

    async def spawn_session(
        self,
        label: str,
        model: Optional[str] = None,
        thinking: Optional[str] = None,
        cwd: Optional[str] = None,
        idle_timeout: int = 1800,
    ) -> dict:
        """Spawn a persistent subagent session.

        The session stays alive and accepts follow-up messages via send().
        Communication is JSON-over-stdin/stdout (pipe mode).
        """
        # Depth guard
        if self._depth >= self.max_depth:
            return {"status": "error", "error": f"Max spawn depth reached ({self.max_depth})."}

        # Children guard
        active_count = sum(1 for r in self.active_runs.values() if not r.completed)
        if active_count >= self.max_children:
            return {"status": "error", "error": f"Max children reached ({self.max_children})"}

        # Label uniqueness
        for r in self.active_runs.values():
            if not r.completed and r.label == label and r.mode == "session":
                return {"status": "error", "error": f"Session with label '{label}' already exists"}

        run_id = str(uuid4())[:8]
        result_file = Path(cwd or os.getcwd()) / ".koi" / "subagent-runs" / f"{run_id}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        cmd: List[str] = [sys.executable, "-m", "koi", "run", "--pipe"]
        if model:
            cmd.extend(["--model", model])
        if thinking:
            cmd.extend(["--thinking", thinking])

        env = {**os.environ, "KOI_SPAWN_DEPTH": str(self._depth + 1)}

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd or os.getcwd(),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        run = SubagentRun(
            id=run_id,
            task=f"[session:{label}]",
            label=label,
            process=process,
            result_file=result_file,
            started_at=datetime.now(),
            mode="session",
            timeout_seconds=0,
            last_activity=datetime.now(),
        )
        self.active_runs[run_id] = run

        # Start idle timeout watcher
        asyncio.create_task(self._idle_watcher(run, idle_timeout))

        return {
            "status": "accepted",
            "run_id": run_id,
            "label": label,
            "note": f"Persistent session '{label}' started. Use send_to_subagent to communicate.",
        }

    async def send(self, target: str, message: str, timeout: float = 120.0) -> dict:
        """Send a message to a persistent subagent and wait for the response."""
        run = self._find_session(target)
        if not run:
            return {"error": f"No active session found for '{target}'"}
        if run.completed:
            return {"error": f"Session '{target}' has ended"}
        if run.mode != "session":
            return {"error": f"'{target}' is a one-shot run, not a persistent session"}

        # Health check: detect already-dead processes
        if not run.completed and run.process and run.process.returncode is not None:
            run.completed = True
            run.exit_code = run.process.returncode
            run.error = f"Process exited with code {run.process.returncode}"
            return {"error": f"Session process has died (exit code {run.process.returncode})"}

        # Dispatch to ACP if applicable
        if run.harness == "acp":
            return await self.send_acp(target, message, timeout=timeout)

        run.last_activity = datetime.now()

        # Write JSON message to stdin
        msg_json = json.dumps({"type": "message", "content": message}) + "\n"
        try:
            run.process.stdin.write(msg_json.encode("utf-8"))
            await run.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            run.completed = True
            run.error = "Process stdin closed"
            return {"error": "Session process has died"}

        # Read JSON response from stdout
        try:
            resp_line = await asyncio.wait_for(
                run.process.stdout.readline(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Timeout = kill to prevent zombie processes and stale responses
            await self._kill_run(run, f"Timed out after {timeout}s")
            return {"error": f"Timed out waiting for response after {timeout}s. Session killed."}

        if not resp_line:
            run.completed = True
            run.error = "Process stdout closed"
            return {"error": "Session process has ended"}

        try:
            resp = json.loads(resp_line.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON from session: {resp_line[:200]}"}

        run.last_activity = datetime.now()
        return resp

    def _find_session(self, target: str) -> Optional["SubagentRun"]:
        """Find an active session by label or run_id."""
        # Try exact ID match first
        if target in self.active_runs:
            return self.active_runs[target]
        # Then search by label
        for r in self.active_runs.values():
            if r.label == target and not r.completed and r.mode == "session":
                return r
        return None

    async def _idle_watcher(self, run: SubagentRun, idle_timeout: int):
        """Kill session if idle for longer than idle_timeout seconds."""
        while not run.completed:
            await asyncio.sleep(60)
            if run.completed:
                break
            if run.last_activity and (datetime.now() - run.last_activity).total_seconds() > idle_timeout:
                await self._kill_run(run, f"Idle timeout ({idle_timeout}s)")
                break

    def list_runs(self) -> list:
        """Return a summary of all tracked sub-agent runs."""
        runs = []
        for r in self.active_runs.values():
            info = {
                "id": r.id,
                "task": r.task[:100],
                "label": r.label,
                "mode": r.mode,
                "status": "completed" if r.completed else "running",
                "started": r.started_at.isoformat(),
                "result_summary": (
                    (r.result or {}).get("summary", "")[:200]
                    if r.completed
                    else None
                ),
            }
            if r.mode == "session" and r.last_activity and not r.completed:
                idle_secs = int((datetime.now() - r.last_activity).total_seconds())
                info["idle_seconds"] = idle_secs
            runs.append(info)
        return runs

    async def kill(self, run_id: str, cascade: bool = True) -> dict:
        """Kill a running sub-agent by ID. Cascade kills children if any."""
        run = self.active_runs.get(run_id)
        if not run:
            return {"error": f"No run with id {run_id}"}
        if not run.completed:
            await self._kill_run(run, "Killed by parent")
        return {"status": "killed", "id": run_id}

    async def kill_all(self) -> dict:
        """Kill all active sub-agents. Used for cascade stop."""
        killed = []
        for run_id, run in list(self.active_runs.items()):
            if not run.completed:
                await self._kill_run(run, "Killed by parent (kill_all)")
                killed.append(run_id)
        return {"status": "killed", "count": len(killed), "ids": killed}

    async def _kill_run(self, run: SubagentRun, reason: str):
        """Internal: kill a run and clean up resources."""
        if run.completed:
            return

        if run.harness == "acp" and run.acp_session:
            try:
                await run.acp_session.close()
            except Exception:
                try:
                    await run.acp_session.kill()
                except Exception:
                    pass
        elif run.mode == "session" and run.process and run.process.stdin:
            # Try graceful shutdown for pipe mode
            try:
                shutdown = json.dumps({"type": "shutdown"}) + "\n"
                run.process.stdin.write(shutdown.encode("utf-8"))
                await run.process.stdin.drain()
                # Brief wait for clean exit
                try:
                    await asyncio.wait_for(run.process.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass
            except Exception:
                pass

        # Force kill if still alive
        if run.process and run.process.returncode is None:
            try:
                run.process.kill()
            except ProcessLookupError:
                pass

        run.completed = True
        run.error = reason
        if self._on_complete:
            await self._on_complete(run)

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove completed runs older than max_age_seconds. Returns count removed."""
        to_remove = []
        now = datetime.now()
        for run_id, run in self.active_runs.items():
            if run.completed:
                age = (now - run.started_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(run_id)
        for run_id in to_remove:
            del self.active_runs[run_id]
        return len(to_remove)

    def get_result(self, run_id: str) -> Optional[dict]:
        """Return the result dict for a completed run (or None)."""
        run = self.active_runs.get(run_id)
        if run and run.completed:
            return {
                "id": run.id,
                "task": run.task,
                "label": run.label,
                "exit_code": run.exit_code,
                "result": run.result,
                "stdout": run.stdout[:2000],
                "stderr": run.stderr[:2000],
                "error": run.error,
            }
        return None

    # ── internals ───────────────────────────────────────────────

    async def _wait_for_completion(self, run: SubagentRun) -> None:
        """Wait for the sub-agent process to finish and collect output."""
        try:
            if run.timeout_seconds > 0:
                stdout, stderr = await asyncio.wait_for(
                    run.process.communicate(),
                    timeout=run.timeout_seconds,
                )
            else:
                stdout, stderr = await run.process.communicate()

            run.exit_code = run.process.returncode
            run.stdout = stdout.decode("utf-8", errors="replace") if stdout else ""
            run.stderr = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Read result from the JSON file written by the child
            if run.result_file.exists():
                try:
                    run.result = json.loads(run.result_file.read_text())
                except (json.JSONDecodeError, OSError):
                    run.result = None

            run.completed = True

            if self._on_complete:
                await self._on_complete(run)

        except asyncio.TimeoutError:
            try:
                run.process.kill()
            except ProcessLookupError:
                pass
            run.completed = True
            run.error = "Timed out"

            if self._on_complete:
                await self._on_complete(run)
