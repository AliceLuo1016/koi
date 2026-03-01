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

    def list_runs(self) -> list:
        """Return a summary of all tracked sub-agent runs."""
        return [
            {
                "id": r.id,
                "task": r.task[:100],
                "label": r.label,
                "status": "completed" if r.completed else "running",
                "started": r.started_at.isoformat(),
                "result_summary": (
                    (r.result or {}).get("summary", "")[:200]
                    if r.completed
                    else None
                ),
            }
            for r in self.active_runs.values()
        ]

    async def kill(self, run_id: str) -> dict:
        """Kill a running sub-agent by ID."""
        run = self.active_runs.get(run_id)
        if not run:
            return {"error": f"No run with id {run_id}"}
        if not run.completed:
            try:
                run.process.kill()
            except ProcessLookupError:
                pass
            run.completed = True
            run.error = "Killed by parent"
        return {"status": "killed", "id": run_id}

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
