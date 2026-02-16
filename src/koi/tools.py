"""Tool definitions and execution for koi agent."""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
from bs4 import BeautifulSoup

from .skills import SkillsManager


# Dangerous command patterns that require confirmation
DANGEROUS_PATTERNS = [
    r'rm\s+-rf\s+/',
    r'DROP\s+TABLE',
    r'aws\s+s3\s+rm\s+--recursive',
    r'format\s+[a-zA-Z]:\\',
    r'del\s+/[qsf]',
    r'sudo\s+rm\s+-rf',
    r'mkfs\.',
    r'fdisk',
    r'dd\s+if=.*of=/dev/',
]


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
                        "cwd": {"type": "string", "description": "Working directory (optional)"}
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
        }
    ]


class ToolExecutor:
    """Execute tool calls and return results."""
    
    def __init__(self, skills_manager: SkillsManager):
        """Initialize tool executor with skills manager."""
        self.skills_manager = skills_manager
    
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
            elif function_name == "read_skill":
                return await self._read_skill(**arguments)
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
    
    async def _exec_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute shell command."""
        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return {
                    "error": f"Dangerous command detected: {command}. Manual confirmation required.",
                    "success": False
                }
        
        try:
            # Set working directory
            work_dir = Path(cwd) if cwd else None
            
            # Execute command
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir
            )
            
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "exit_code": process.returncode,
                "success": process.returncode == 0
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