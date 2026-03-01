"""Tool definitions and execution for koi agent."""

import asyncio
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from bs4 import BeautifulSoup

from .sandbox import Sandbox
from .skills import SkillsManager

# Per-tool output limits
MAX_READ_LINES = 2000
MAX_READ_BYTES = 50_000  # 50KB
MAX_EXEC_OUTPUT_BYTES = 50_000  # 50KB
MAX_WEB_FETCH_CHARS = 20_000


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get OpenAI-format tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Defaults to first 2000 lines / 50KB. Use offset/limit for larger files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                        "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                        "limit": {"type": "integer", "description": "Maximum number of lines to read"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file with content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Make a surgical edit to a file by replacing exact text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Exact text to find and replace"},
                        "new_text": {"type": "string", "description": "New text to replace the old text with"}
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Execute a shell command and return stdout, stderr, and exit code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "cwd": {"type": "string", "description": "Working directory (optional)"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (optional, default 30)"}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "glob_files",
                "description": "Find files matching a glob pattern. Faster and safer than exec_command with find.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'"},
                        "base_dir": {"type": "string", "description": "Directory to search in (default: current directory)"}
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search file contents using a regex pattern. Returns matching lines with file path and line number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "File or directory to search (default: current directory)"},
                        "file_glob": {"type": "string", "description": "Filename filter, e.g. '*.py' or '*.ts'"},
                        "case_insensitive": {"type": "boolean", "description": "Case-insensitive search (default: false)"}
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using Brave Search API (placeholder - returns TODO)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": "Fetch URL content and convert to readable markdown",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_memory",
                "description": "Append important information to persistent memory (.koi/MEMORY.md). Use this to remember preferences, decisions, and context across sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Text to append to memory"}
                    },
                    "required": ["content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_skill",
                "description": "Read the full content of a skill file by name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string", "description": "Name of the skill to read"}
                    },
                    "required": ["skill_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_cron_job",
                "description": "Add a koi cron job that runs a natural language task on a schedule. The task will be interpreted by koi each time it runs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "schedule": {"type": "string", "description": "Cron schedule expression, e.g. '0 * * * *' for every hour"},
                        "task": {"type": "string", "description": "Natural language task for koi to interpret and execute each time"}
                    },
                    "required": ["schedule", "task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_cron_jobs",
                "description": "List all registered koi cron jobs.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remove_cron_job",
                "description": "Remove a koi cron job by its ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "The job ID to remove"}
                    },
                    "required": ["job_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remove_file",
                "description": "Remove a file or directory under .koi/. Only paths within .koi/ are allowed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file or directory to remove (must be under .koi/)"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_alert",
                "description": "Create a structured alert with severity and proposed fix. Saves to .koi/alerts/ and sends a desktop notification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Alert title"},
                        "summary": {"type": "string", "description": "Description of the issue"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Alert severity"},
                        "proposed_fix": {"type": "string", "description": "Suggested fix for the issue"},
                        "fix_command": {"type": "string", "description": "Optional shell command to apply the fix"}
                    },
                    "required": ["title", "summary", "severity", "proposed_fix"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_alerts",
                "description": "List alerts filtered by status (pending/approved/dismissed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["pending", "approved", "dismissed"], "description": "Filter by status (default: pending)"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "resolve_alert",
                "description": "Resolve an alert by approving or dismissing it. If approved and a fix_command exists, returns the command without executing it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_file": {"type": "string", "description": "Alert filename or path"},
                        "resolution": {"type": "string", "enum": ["approved", "dismissed"], "description": "Resolution action"}
                    },
                    "required": ["alert_file", "resolution"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_subagent",
                "description": "Spawn an isolated Koi sub-agent to run a task in the background. Returns a run_id to track progress.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Natural language task for the sub-agent to execute"},
                        "label": {"type": "string", "description": "Short label for display purposes"},
                        "model": {"type": "string", "description": "Override model for this sub-agent"},
                        "thinking": {"type": "string", "description": "Thinking level for sub-agent (off/minimal/low/medium/high)"},
                        "timeout_seconds": {"type": "integer", "description": "Kill sub-agent after this many seconds (0 = no timeout)"},
                        "cwd": {"type": "string", "description": "Working directory for the sub-agent"}
                    },
                    "required": ["task"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_subagents",
                "description": "List all active and completed sub-agents with their status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "kill_subagent",
                "description": "Kill a running sub-agent by its run ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string", "description": "The run ID of the sub-agent to kill"}
                    },
                    "required": ["run_id"]
                }
            }
        }
    ]


class ToolExecutor:
    """Execute tool calls and return results."""

    def __init__(self, skills_manager: SkillsManager, sandbox: Sandbox = None, subagent_manager=None):
        """Initialize tool executor with skills manager and sandbox."""
        self.skills_manager = skills_manager
        self.sandbox = sandbox or Sandbox()
        self.subagent_manager = subagent_manager
    
    async def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        function_name = tool_call["function"]["name"].replace("-", "_")

        try:
            raw_args = tool_call["function"].get("arguments", "{}") or "{}"
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse arguments: {e}",
                "success": False
            }
        
        try:
            if function_name == "read_file":
                return await self._read_file(**arguments)
            elif function_name == "write_file":
                return await self._write_file(**arguments)
            elif function_name == "edit_file":
                return await self._edit_file(**arguments)
            elif function_name == "exec_command":
                return await self._exec_command(**arguments)
            elif function_name == "glob_files":
                return await self._glob_files(**arguments)
            elif function_name == "grep_files":
                return await self._grep_files(**arguments)
            elif function_name == "web_search":
                return await self._web_search(**arguments)
            elif function_name == "web_fetch":
                return await self._web_fetch(**arguments)
            elif function_name == "update_memory":
                return await self._update_memory(**arguments)
            elif function_name == "read_skill":
                return await self._read_skill(**arguments)
            elif function_name == "add_cron_job":
                return await self._add_cron_job(**arguments)
            elif function_name == "list_cron_jobs":
                return await self._list_cron_jobs(**arguments)
            elif function_name == "remove_cron_job":
                return await self._remove_cron_job(**arguments)
            elif function_name == "remove_file":
                return await self._remove_file(**arguments)
            elif function_name == "create_alert":
                return await self._create_alert(**arguments)
            elif function_name == "list_alerts":
                return await self._list_alerts(**arguments)
            elif function_name == "resolve_alert":
                return await self._resolve_alert(**arguments)
            elif function_name == "spawn_subagent":
                return await self._spawn_subagent(**arguments)
            elif function_name == "list_subagents":
                return await self._list_subagents(**arguments)
            elif function_name == "kill_subagent":
                return await self._kill_subagent(**arguments)
            else:
                return {
                    "error": f"Unknown function: {function_name}",
                    "success": False
                }
        
        except asyncio.CancelledError:
            raise

        except Exception as e:
            return {
                "error": f"Tool execution failed: {e}",
                "success": False
            }
    
    async def _read_file(self, path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Read file contents with optional offset and limit."""
        try:
            allowed, reason = self.sandbox.check_read(path)
            if not allowed:
                return {"error": reason, "success": False}

            file_path = Path(path)

            if not file_path.exists():
                return {"error": f"File not found: {path}", "success": False}

            if not file_path.is_file():
                return {"error": f"Path is not a file: {path}", "success": False}

            with open(file_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)
            lines = all_lines

            # Apply offset
            if offset:
                lines = lines[offset - 1:]  # Convert to 0-indexed

            # Apply limit: use explicit limit if provided, otherwise default
            explicit_limit = limit is not None
            effective_limit = limit if explicit_limit else MAX_READ_LINES
            truncated_by_lines = len(lines) > effective_limit
            if truncated_by_lines:
                lines = lines[:effective_limit]

            content = "".join(lines)

            # Check byte size limit
            truncated_by_bytes = False
            if len(content.encode("utf-8")) > MAX_READ_BYTES:
                truncated_by_bytes = True
                # Truncate to fit within byte limit
                encoded = content.encode("utf-8")[:MAX_READ_BYTES]
                content = encoded.decode("utf-8", errors="ignore")
                # Recount lines after byte truncation
                lines = content.splitlines(keepends=True)

            # Add truncation notice
            if truncated_by_lines or truncated_by_bytes:
                content += f"\n[output truncated: {len(lines)} of {total_lines} lines shown. Use offset/limit for more.]"

            return {
                "content": content,
                "lines_read": len(lines),
                "success": True
            }

        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file."""
        try:
            allowed, reason = self.sandbox.check_write(path)
            if not allowed:
                return {"error": reason, "success": False}

            file_path = Path(path)
            
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {
                "message": f"Successfully wrote {len(content)} characters to {path}",
                "success": True
            }
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _edit_file(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Make surgical edit to file."""
        try:
            allowed, reason = self.sandbox.check_write(path)
            if not allowed:
                return {"error": reason, "success": False}

            file_path = Path(path)
            
            if not file_path.exists():
                return {"error": f"File not found: {path}", "success": False}
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if old_text not in content:
                return {"error": f"Text not found in file: {old_text[:100]}...", "success": False}
            
            # Find the line number where the change occurs
            lines = content.splitlines(keepends=True)
            line_num = None
            for i, line in enumerate(lines):
                if old_text in line:
                    line_num = i + 1  # 1-indexed
                    break
            
            # Replace first occurrence
            new_content = content.replace(old_text, new_text, 1)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            # Create a visual diff-like output
            diff_lines = []
            if line_num and len(old_text) < 200 and len(new_text) < 200:
                diff_lines.append(f"Line {line_num}:")
                diff_lines.append(f"- {old_text}")
                diff_lines.append(f"+ {new_text}")
                diff_msg = "\n".join(diff_lines)
            else:
                # For longer text, show a summary
                diff_msg = f"Replaced {len(old_text)} characters with {len(new_text)} characters"
            
            return {
                "message": f"Successfully edited {path}\n{diff_msg}",
                "success": True
            }
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _exec_command(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute shell command with sandbox restrictions."""
        # Check command against sandbox rules
        allowed, reason, needs_confirm = self.sandbox.check_command(command)
        if not allowed:
            return {"error": reason, "success": False}
        if needs_confirm:
            return {
                "error": f"Command requires confirmation: {command}\n{reason}",
                "needs_confirmation": True,
                "success": False
            }

        timeout = timeout or 60

        try:
            # Set working directory
            work_dir = str(Path(cwd)) if cwd else None

            # Use sandboxed environment (only allowlisted vars)
            env = self.sandbox.get_safe_env()
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=env,
                executable="/bin/bash",
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    process.kill()
                return {
                    "error": f"Command timed out after {timeout} seconds",
                    "success": False
                }

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # Truncate if combined output exceeds limit
            combined_len = len(stdout) + len(stderr)
            truncated = False
            if combined_len > MAX_EXEC_OUTPUT_BYTES:
                truncated = True
                # Truncate stdout first, then stderr
                if len(stdout) > MAX_EXEC_OUTPUT_BYTES:
                    stdout = stdout[:MAX_EXEC_OUTPUT_BYTES]
                    stderr = ""
                else:
                    remaining = MAX_EXEC_OUTPUT_BYTES - len(stdout)
                    stderr = stderr[:remaining]

            result = {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "success": process.returncode == 0,
            }
            if truncated:
                shown = len(stdout) + len(stderr)
                result["truncation_notice"] = (
                    f"[output truncated: showing first {shown} of {combined_len} bytes]"
                )
            return result

        except asyncio.CancelledError:
            # Kill the subprocess on cancellation
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except asyncio.TimeoutError:
                    process.kill()
            except (NameError, ProcessLookupError):
                pass
            raise

        except Exception as e:
            return {"error": str(e), "success": False}
    
    _SKIP_DIRS = frozenset({
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
    })

    async def _glob_files(self, pattern: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
        """Find files matching a glob pattern."""
        try:
            base = Path(base_dir) if base_dir else Path.cwd()
            allowed, reason = self.sandbox.check_read(str(base))
            if not allowed:
                return {"error": reason, "success": False}
            if not base.is_dir():
                return {"error": f"Not a directory: {base}", "success": False}

            iterator = base.rglob(pattern) if "**" in pattern else base.glob(pattern)
            matches = []
            for p in sorted(iterator):
                if any(part in self._SKIP_DIRS for part in p.parts):
                    continue
                matches.append(str(p.relative_to(base)))
                if len(matches) >= 500:
                    break

            return {
                "matches": matches,
                "count": len(matches),
                "truncated": len(matches) == 500,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _grep_files(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_glob: Optional[str] = None,
        case_insensitive: bool = False,
    ) -> Dict[str, Any]:
        """Search file contents using a regex pattern."""
        try:
            base = Path(path) if path else Path.cwd()
            allowed, reason = self.sandbox.check_read(str(base))
            if not allowed:
                return {"error": reason, "success": False}

            flags = re.IGNORECASE if case_insensitive else 0
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                return {"error": f"Invalid regex: {e}", "success": False}

            MAX_MATCHES = 200

            def _candidate_files():
                if base.is_file():
                    yield base
                else:
                    glob_pat = file_glob or "*"
                    for p in sorted(base.rglob(glob_pat)):
                        if p.is_file() and not any(part in self._SKIP_DIRS for part in p.parts):
                            yield p

            matches = []
            for file_path in _candidate_files():
                if len(matches) >= MAX_MATCHES:
                    break
                try:
                    text = file_path.read_text(encoding="utf-8", errors="strict")
                except (UnicodeDecodeError, OSError):
                    continue
                rel = str(file_path.relative_to(base)) if not base.is_file() else str(file_path)
                for lineno, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        matches.append({"file": rel, "line": lineno, "text": line[:200]})
                        if len(matches) >= MAX_MATCHES:
                            break

            return {
                "matches": matches,
                "count": len(matches),
                "truncated": len(matches) == MAX_MATCHES,
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _web_search(self, query: str) -> Dict[str, Any]:
        """Placeholder for web search - returns TODO."""
        return {
            "message": f"TODO: Implement web search for query: {query}",
            "results": [],
            "success": True
        }
    
    async def _web_fetch(self, url: str) -> Dict[str, Any]:
        """Fetch URL content and convert to markdown."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                
                # Parse HTML and extract text
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Simple markdown conversion
                title = soup.find('title')
                if title:
                    text = f"# {title.get_text().strip()}\n\n{text}"
                
                truncated = len(text) > MAX_WEB_FETCH_CHARS
                content = text[:MAX_WEB_FETCH_CHARS]
                if truncated:
                    content += f"\n[output truncated: showing first {MAX_WEB_FETCH_CHARS} of {len(text)} chars]"

                return {
                    "content": content,
                    "url": url,
                    "success": True
                }
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _update_memory(self, content: str) -> Dict[str, Any]:
        """Append content to persistent memory."""
        try:
            from .memory import Memory
            memory = Memory()
            memory.append(content)
            return {
                "message": f"Added to memory: {content[:100]}{'...' if len(content) > 100 else ''}",
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _read_skill(self, skill_name: str) -> Dict[str, Any]:
        """Read skill content by name."""
        try:
            content = self.skills_manager.read_skill(skill_name)
            
            return {
                "content": content,
                "skill_name": skill_name,
                "success": True
            }
        
        except FileNotFoundError as e:
            return {"error": str(e), "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _add_cron_job(self, schedule: str, task: str) -> Dict[str, Any]:
        """Add a cron job via CronManager directly."""
        try:
            from .cron import CronManager
            cron_manager = CronManager()
            job_id = cron_manager.add_job(schedule, task)
            return {
                "message": f"Cron job added (ID: {job_id}). Schedule: {schedule}. Task: {task}",
                "job_id": job_id,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _list_cron_jobs(self) -> Dict[str, Any]:
        """List all cron jobs."""
        try:
            from .cron import CronManager
            cron_manager = CronManager()
            jobs = cron_manager.list_jobs()
            return {"jobs": jobs, "count": len(jobs), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _remove_cron_job(self, job_id: str) -> Dict[str, Any]:
        """Remove a cron job by ID."""
        try:
            from .cron import CronManager
            cron_manager = CronManager()
            cron_manager.remove_job(job_id)
            return {"message": f"Cron job {job_id} removed.", "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _remove_file(self, path: str) -> Dict[str, Any]:
        """Remove a file or directory, restricted to .koi/."""
        try:
            import shutil
            target = Path(path)
            if not target.is_absolute():
                target = (Path.cwd() / target).resolve()
            else:
                target = target.resolve()

            koi_dir = (Path.cwd() / ".koi").resolve()
            try:
                target.relative_to(koi_dir)
            except ValueError:
                return {"error": f"Access denied: can only remove paths under .koi/", "success": False}

            if not target.exists():
                return {"error": f"Path not found: {path}", "success": False}

            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

            return {"message": f"Removed {path}", "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _get_alerts_dir(self) -> Path:
        """Get the alerts directory, creating it if needed."""
        alerts_dir = Path.cwd() / ".koi" / "alerts"
        alerts_dir.mkdir(parents=True, exist_ok=True)
        return alerts_dir

    def _send_desktop_notification(self, title: str, message: str) -> None:
        """Send a desktop notification (best-effort, cross-platform)."""
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(
                    ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                    capture_output=True, timeout=5
                )
            elif system == "Linux" and os.environ.get("DISPLAY"):
                subprocess.run(
                    ["notify-send", title, message],
                    capture_output=True, timeout=5
                )
        except Exception:
            pass  # Silently skip if notification fails

    async def _create_alert(self, title: str, summary: str, severity: str, proposed_fix: str, fix_command: str = None) -> Dict[str, Any]:
        """Create a structured alert file."""
        try:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            sanitized = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
            filename = f"{now.strftime('%Y-%m-%d_%H-%M')}_{sanitized}.md"
            
            alerts_dir = self._get_alerts_dir()
            file_path = alerts_dir / filename
            
            fix_line = f"- **Fix Command:** `{fix_command}`" if fix_command else "- **Fix Command:** N/A"
            content = f"""# {title}
- **Status:** pending
- **Severity:** {severity}
- **Detected:** {timestamp}
- **Summary:** {summary}
- **Proposed Fix:** {proposed_fix}
{fix_line}
"""
            file_path.write_text(content, encoding="utf-8")
            self._send_desktop_notification(f"Koi Alert [{severity.upper()}]", title)
            
            return {"message": f"Alert created: {file_path}", "file": str(file_path), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _list_alerts(self, status: str = "pending") -> Dict[str, Any]:
        """List alerts filtered by status."""
        try:
            alerts_dir = self._get_alerts_dir()
            alerts = []
            for f in sorted(alerts_dir.glob("*.md")):
                text = f.read_text(encoding="utf-8")
                status_match = re.search(r'\*\*Status:\*\*\s*(\w+)', text)
                file_status = status_match.group(1) if status_match else "unknown"
                if file_status != status:
                    continue
                title_match = re.search(r'^#\s+(.+)', text, re.MULTILINE)
                severity_match = re.search(r'\*\*Severity:\*\*\s*(\w+)', text)
                detected_match = re.search(r'\*\*Detected:\*\*\s*(.+)', text)
                summary_match = re.search(r'\*\*Summary:\*\*\s*(.+)', text)
                alerts.append({
                    "title": title_match.group(1) if title_match else f.stem,
                    "severity": severity_match.group(1) if severity_match else "unknown",
                    "detected": detected_match.group(1).strip() if detected_match else "unknown",
                    "summary": summary_match.group(1).strip() if summary_match else "",
                    "file": f.name
                })
            return {"alerts": alerts, "count": len(alerts), "status_filter": status, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _resolve_alert(self, alert_file: str, resolution: str) -> Dict[str, Any]:
        """Resolve an alert by updating its status."""
        try:
            alerts_dir = self._get_alerts_dir()
            file_path = alerts_dir / Path(alert_file).name
            if not file_path.exists():
                return {"error": f"Alert file not found: {alert_file}", "success": False}

            text = file_path.read_text(encoding="utf-8")
            new_text = re.sub(r'(\*\*Status:\*\*\s*)\w+', f'\\1{resolution}', text)
            file_path.write_text(new_text, encoding="utf-8")

            result = {"message": f"Alert resolved as {resolution}: {file_path.name}", "success": True}

            if resolution == "approved":
                cmd_match = re.search(r'\*\*Fix Command:\*\*\s*`([^`]+)`', text)
                if cmd_match and cmd_match.group(1) != "N/A":
                    result["fix_command"] = cmd_match.group(1)
                    result["message"] += f"\nFix command available: {cmd_match.group(1)}"

            return result
        except Exception as e:
            return {"error": str(e), "success": False}

    # ── Sub-agent tools ─────────────────────────────────────────

    async def _spawn_subagent(
        self,
        task: str,
        label: Optional[str] = None,
        model: Optional[str] = None,
        thinking: Optional[str] = None,
        timeout_seconds: int = 0,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Spawn an isolated Koi sub-agent."""
        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available", "success": False}
        try:
            result = await self.subagent_manager.spawn(
                task=task,
                label=label,
                model=model,
                thinking=thinking,
                timeout_seconds=timeout_seconds,
                cwd=cwd,
            )
            if result["status"] == "error":
                return {"error": result["error"], "success": False}
            return {
                "message": f"Sub-agent {result['run_id']} started: {task[:80]}",
                "run_id": result["run_id"],
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _list_subagents(self) -> Dict[str, Any]:
        """List active and completed sub-agents."""
        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available", "success": False}
        try:
            runs = self.subagent_manager.list_runs()
            return {"runs": runs, "count": len(runs), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _kill_subagent(self, run_id: str) -> Dict[str, Any]:
        """Kill a running sub-agent."""
        if self.subagent_manager is None:
            return {"error": "Sub-agent spawning is not available", "success": False}
        try:
            result = await self.subagent_manager.kill(run_id)
            if "error" in result:
                return {"error": result["error"], "success": False}
            return {
                "message": f"Sub-agent {run_id} killed.",
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "success": False}