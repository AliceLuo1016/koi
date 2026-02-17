"""Tool definitions and execution for koi agent."""

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


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Get OpenAI-format tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file",
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
                "description": "Append important information to persistent memory (.agent/MEMORY.md). Use this to remember preferences, decisions, and context across sessions.",
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
                "name": "create_alert",
                "description": "Create a structured alert with severity and proposed fix. Saves to .agent/alerts/ and sends a desktop notification.",
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
        }
    ]


class ToolExecutor:
    """Execute tool calls and return results."""
    
    def __init__(self, skills_manager: SkillsManager, sandbox: Sandbox = None):
        """Initialize tool executor with skills manager and sandbox."""
        self.skills_manager = skills_manager
        self.sandbox = sandbox or Sandbox()
    
    async def execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        function_name = tool_call["function"]["name"]
        
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
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
            elif function_name == "web_search":
                return await self._web_search(**arguments)
            elif function_name == "web_fetch":
                return await self._web_fetch(**arguments)
            elif function_name == "update_memory":
                return await self._update_memory(**arguments)
            elif function_name == "read_skill":
                return await self._read_skill(**arguments)
            elif function_name == "create_alert":
                return await self._create_alert(**arguments)
            elif function_name == "list_alerts":
                return await self._list_alerts(**arguments)
            elif function_name == "resolve_alert":
                return await self._resolve_alert(**arguments)
            else:
                return {
                    "error": f"Unknown function: {function_name}",
                    "success": False
                }
        
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
                lines = f.readlines()
            
            # Apply offset and limit
            if offset:
                lines = lines[offset - 1:]  # Convert to 0-indexed
            
            if limit:
                lines = lines[:limit]
            
            content = "".join(lines)
            
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
            
            # Replace first occurrence
            new_content = content.replace(old_text, new_text, 1)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return {
                "message": f"Successfully edited {path}",
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
            work_dir = Path(cwd) if cwd else None
            
            # Use sandboxed environment (only allowlisted vars)
            env = self.sandbox.get_safe_env()
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
                env=env,
                executable="/bin/zsh",
                timeout=timeout
            )
            
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "exit_code": process.returncode,
                "success": process.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            return {
                "error": f"Command timed out after {timeout} seconds",
                "success": False
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
                
                return {
                    "content": text[:50000],  # Limit content size
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

    def _get_alerts_dir(self) -> Path:
        """Get the alerts directory, creating it if needed."""
        alerts_dir = Path.cwd() / ".agent" / "alerts"
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