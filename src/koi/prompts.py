"""System prompt building for koi agent."""

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .config import Config
from .memory import Memory
from .skills import SkillsManager
from .tools import get_tool_definitions


def build_system_prompt(config: Config) -> str:
    """Build the complete system prompt for the agent."""
    
    # Base agent instructions
    base_prompt = """You are Koi, a terminal-based AI agent that helps users with tasks through conversation and tool usage.

You have access to the following capabilities:
- Read, write, and edit files
- Execute shell commands (with safety checks)
- Search and fetch web content
- Manage memory and skills
- Work with project-specific configurations

You think step by step and use tools to accomplish tasks. When a user asks you to do something:
1. Understand what they want
2. Plan the approach
3. Use appropriate tools to gather information or make changes
4. Verify results and report back

Be helpful, accurate, and safe. Always confirm before running potentially dangerous commands.

When analyzing logs:
- Look for error patterns, spikes, and anomalies
- If you find issues, use create_alert to record them with a proposed fix
- Use list_alerts to check for pending alerts
- Use resolve_alert when the user approves or dismisses a fix

Important: For scheduling tasks, ALWAYS use koi's built-in cron system:
- Add: exec_command with "koi cron add '<schedule>' '<task>'"
- List: exec_command with "koi cron list"
- Remove: exec_command with "koi cron remove <job_id>"
Cron logs are stored in .agent/cron-logs/ automatically. Do NOT modify the system crontab directly."""

    sections = [base_prompt]
    
    # Add tools information
    tools_section = _build_tools_section()
    sections.append(tools_section)
    
    # Add skills information
    skills_section = _build_skills_section(config)
    sections.append(skills_section)
    
    # Add project instructions if available
    project_section = _build_project_section()
    if project_section:
        sections.append(project_section)
    
    # Add memory if available
    memory_section = _build_memory_section()
    if memory_section:
        sections.append(memory_section)
    
    # Add alerts check
    alerts_section = _build_alerts_section()
    if alerts_section:
        sections.append(alerts_section)
    
    # Add context information
    context_section = _build_context_section()
    sections.append(context_section)
    
    return "\n\n".join(sections)


def _build_tools_section() -> str:
    """Build tools section of system prompt."""
    tools = get_tool_definitions()
    
    tool_list = []
    for tool in tools:
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        tool_list.append(f"- {name}: {description}")
    
    return f"""Available Tools:
{chr(10).join(tool_list)}

Use tools by making function calls. Always check tool results before proceeding."""


def _build_skills_section(config: Config) -> str:
    """Build skills section of system prompt."""
    try:
        skills_manager = SkillsManager(config.skills_paths)
        skills_summary = skills_manager.get_skills_summary()
        
        if "No skills available" in skills_summary:
            return "Skills: No skills found in configured paths."
        
        return f"""Skills:
{skills_summary}

Use the read_skill tool to load full skill content when needed."""
    
    except Exception:
        return "Skills: Error loading skills."


def _build_project_section() -> str:
    """Build project instructions section if AGENTS.md exists."""
    agents_file = Path.cwd() / ".agent" / "AGENTS.md"
    
    if not agents_file.exists():
        return ""
    
    try:
        with open(agents_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        return f"""Project Instructions:
{content}"""
    
    except Exception:
        return ""


def _build_memory_section() -> str:
    """Build memory section if MEMORY.md exists."""
    try:
        memory = Memory()
        content = memory.load()
        
        if not content.strip():
            return ""
        
        return f"""Memory:
{content}"""
    
    except Exception:
        return ""


def _build_alerts_section() -> str:
    """Check for pending alerts and add to prompt if any exist."""
    import re as _re
    alerts_dir = Path.cwd() / ".agent" / "alerts"
    if not alerts_dir.exists():
        return ""
    try:
        pending = []
        for f in sorted(alerts_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            status_match = _re.search(r'\*\*Status:\*\*\s*(\w+)', text)
            if status_match and status_match.group(1) == "pending":
                title_match = _re.search(r'^#\s+(.+)', text, _re.MULTILINE)
                title = title_match.group(1) if title_match else f.stem
                pending.append(title)
        if not pending:
            return ""
        titles = "\n".join(f"- {t}" for t in pending)
        return f"⚠️ You have {len(pending)} pending alert(s). Offer to review them.\n{titles}"
    except Exception:
        return ""


def _build_context_section() -> str:
    """Build context section with environment information."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_dir = Path.cwd()
    system_info = platform.system()
    python_version = platform.python_version()
    
    return f"""Current Context:
- Time: {current_time}
- Working Directory: {current_dir}
- Operating System: {system_info}
- Python Version: {python_version}
- User: {os.getenv('USER', 'unknown')}

Use this context to provide relevant assistance."""


def build_tool_call_message(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Build a message for a tool call in OpenAI format."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call]
    }


def build_tool_result_message(tool_call: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a message for tool result in OpenAI format."""
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": _format_tool_result(result)
    }


def _format_tool_result(result: Dict[str, Any]) -> str:
    """Format tool result for inclusion in conversation."""
    if not result.get("success", False):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    # Format based on result content
    if "content" in result:
        return result["content"]
    elif "message" in result:
        return result["message"]
    elif "stdout" in result:
        output = result["stdout"]
        if result.get("stderr"):
            output += f"\n[stderr]: {result['stderr']}"
        return output
    else:
        # Return JSON representation for complex results
        import json
        return json.dumps(result, indent=2)