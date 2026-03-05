"""System prompt building for koi agent."""

import json
import os
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config
from .memory import Memory
from .skills import SkillsManager
from .tools import get_tool_definitions


def build_system_prompt(
    config: Config,
    non_interactive: bool = False,
    use_reasoning_tags: bool = False,
) -> str:
    """Build the complete system prompt for the agent."""

    sections = []

    # 1. Core identity
    sections.append(
        "You are Koi, a terminal-based AI agent. "
        "Be helpful, accurate, and concise. Always respond in English."
    )

    # 2. Reasoning format (early for emphasis, before tools)
    if use_reasoning_tags:
        sections.append(_build_reasoning_format_section())

    # 3. Tool call style
    sections.append("""## Tool Call Style
- Do not narrate routine, low-risk tool calls — just call the tool.
- Narrate only for multi-step work, complex problems,
  or sensitive actions (e.g., deletions).
- Keep narration brief and value-dense.""")

    # 4. Safety
    sections.append("""## Safety
- Prioritize safety and human oversight over task completion.
- Do not run destructive commands without asking first.
- Confirm before: deleting files, sending emails, anything irreversible.
- When in doubt, ask.""")

    # 5. Tools (with per-tool usage tips)
    tools_section = _build_tools_section()
    sections.append(tools_section)

    # 5. Skills
    skills_section = _build_skills_section(config)
    sections.append(skills_section)

    # 6. Memory guidance
    sections.append("""## Memory
Before answering questions about prior work, decisions,
or preferences: check memory first using update_memory.
When analyzing logs, use create_alert / list_alerts /
resolve_alert for structured issue tracking.""")

    # 7. Cron
    sections.append("""## Cron
Use built-in cron tools for scheduling — do NOT use exec_command for cron management.
- add_cron_job(schedule, task) — task is a natural
  language instruction koi interprets each run
- list_cron_jobs() / remove_cron_job(job_id)
Cron logs are stored in .koi/cron-logs/ automatically.""")

    # 8. Project instructions
    project_section = _build_project_section()
    if project_section:
        sections.append(project_section)

    # 9. Memory content
    memory_section = _build_memory_section()
    if memory_section:
        sections.append(memory_section)

    # 10. Alerts
    alerts_section = _build_alerts_section()
    if alerts_section:
        sections.append(alerts_section)

    # 11. Non-interactive mode
    if non_interactive:
        sections.append("""## Non-Interactive Mode
IMPORTANT: You are running in non-interactive (cron) mode.
There is no user to respond.
- Do NOT ask for confirmation or clarification.
  Execute all tool calls and commands directly.
- Do NOT wait for user input. Complete the task
  autonomously and report the result.
- Do NOT create or schedule cron jobs. You ARE a cron
  job. Just execute the task immediately.
- Ignore phrases like "every hour" or "every minute"
  in the task — those describe the cron schedule, not
  what you should do. Focus on the actual action.""")

    # 12. Context
    context_section = _build_context_section()
    sections.append(context_section)

    return "\n\n".join(sections)


def _build_reasoning_format_section() -> str:
    """Build the reasoning format section for tag-based thinking models."""
    return """## Reasoning Format
ALL internal reasoning MUST be inside <think>...</think>.
Do not output any analysis outside <think>.
Format every reply as <think>...</think> then <final>...</final>, with no other text.
Only text inside <final> is shown to the user; everything else is discarded.
Example: <think>Short internal reasoning.</think><final>Hey there!</final>"""


_TOOL_TIPS = {
    "read_file": (
        "Output truncated to 2000 lines / 50KB. Use offset/limit for large files."
    ),
    "write_file": "Creates parent directories automatically.",
    "edit_file": "old_text must match exactly including whitespace.",
    "exec_command": "Output capped at 50KB. Use timeout for long-running commands.",
    "glob_files": "Faster and safer than exec_command with find. Max 500 results.",
    "grep_files": "Returns up to 200 matches with file path and line number.",
    "web_fetch": "Content capped at 20K chars.",
    "web_search": "Placeholder — not yet implemented.",
}


def _build_tools_section() -> str:
    """Build tools section of system prompt."""
    tools = get_tool_definitions()

    tool_list = []
    for tool in tools:
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        tip = _TOOL_TIPS.get(name, "")
        if tip:
            tool_list.append(f"- {name}: {description} — {tip}")
        else:
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
            return "## Skills\nNo skills found in configured paths."

        return f"""## Skills
{skills_summary}

Before responding: scan available skills above.
- If one clearly matches the user's request, use
  read_skill to load it, then follow its instructions.
- If none clearly match, do not read any skill.
- Never read more than one skill upfront; only read after selecting.
- Use read_skill (not read_file) to load skills."""

    except Exception:
        return "## Skills\nError loading skills."


def _build_project_section() -> str:
    """Build project instructions section if AGENTS.md exists."""
    agents_file = Path.cwd() / ".koi" / "AGENTS.md"

    if not agents_file.exists():
        return ""

    try:
        with open(agents_file, encoding="utf-8") as f:
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
    alerts_dir = Path.cwd() / ".koi" / "alerts"
    if not alerts_dir.exists():
        return ""
    try:
        pending = []
        for f in sorted(alerts_dir.glob("*.md")):
            text = f.read_text(encoding="utf-8")
            status_match = re.search(r"\*\*Status:\*\*\s*(\w+)", text)
            if status_match and status_match.group(1) == "pending":
                title_match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
                title = title_match.group(1) if title_match else f.stem
                pending.append(title)
        if not pending:
            return ""
        titles = "\n".join(f"- {t}" for t in pending)
        return (
            f"⚠️ You have {len(pending)} pending"
            f" alert(s). Offer to review them.\n{titles}"
        )
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
- User: {os.getenv("USER", "unknown")}

Use this context to provide relevant assistance."""


_TRUNCATION_MIN_CHARS = 2000
_TRUNCATION_MAX_CHARS = 400_000
_TRUNCATION_SUFFIX = (
    "\n\n\u26a0\ufe0f [Content truncated \u2014 original was too large for the model "
    "context window. Use offset/limit parameters to read smaller chunks.]"
)


def truncate_tool_result(text: str, context_window: int) -> str:
    """Truncate a tool result string to fit within context-window budget.

    Budget = min(context_window * 0.3 * 4, 400_000) chars, but always at
    least 2000 chars.  When truncating, try to break at a newline boundary.
    """
    max_chars = max(
        _TRUNCATION_MIN_CHARS,
        min(int(context_window * 0.3 * 4), _TRUNCATION_MAX_CHARS),
    )
    if len(text) <= max_chars:
        return text

    budget = max_chars - len(_TRUNCATION_SUFFIX)
    if budget < _TRUNCATION_MIN_CHARS:
        budget = _TRUNCATION_MIN_CHARS

    # Try to break at a newline boundary
    cut = text[:budget]
    last_nl = cut.rfind("\n")
    if last_nl > _TRUNCATION_MIN_CHARS:
        cut = cut[: last_nl + 1]

    return cut + _TRUNCATION_SUFFIX


def build_tool_call_message(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Build a message for a tool call in OpenAI format."""
    return {"role": "assistant", "content": None, "tool_calls": [tool_call]}


def build_tool_result_message(
    tool_call: dict[str, Any],
    result: dict[str, Any],
    context_window: int = 128_000,
) -> dict[str, Any]:
    """Build a message for tool result in OpenAI format."""
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": _format_tool_result(result, context_window),
    }


def _format_tool_result(result: dict[str, Any], context_window: int = 128_000) -> str:
    """Format tool result for inclusion in conversation."""
    if not result.get("success", False):
        error_msg = result.get("error", "")
        if not error_msg:
            error_msg = (
                result.get("stderr") or result.get("stdout") or "Unknown error"
            ).strip()
        return f"Error: {error_msg}"

    # Format based on result content
    if "content" in result:
        text = result["content"]
    elif "message" in result:
        text = result["message"]
    elif "stdout" in result:
        text = result["stdout"]
        if result.get("stderr"):
            text += f"\n[stderr]: {result['stderr']}"
        if result.get("truncation_notice"):
            text += f"\n{result['truncation_notice']}"
    else:
        # Return JSON representation for complex results
        text = json.dumps(result, indent=2)

    # Layer 2: generic safety-net truncation based on context window
    return truncate_tool_result(text, context_window)
