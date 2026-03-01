"""CLI commands for koi agent."""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from .agent import Agent
from .config import Config, create_default_config, load_claude_code_api_key, normalize_think_level
from .cron import CronManager
from .llm import LLMClient
from .memory import Memory
from .skills import SkillsManager

console = Console()


@click.group()
@click.version_option()
def main():
    """Koi - Terminal-based AI agent with memory, tool calling, skills, and system cron."""
    pass


MODEL_PRESETS = {
    "1": {
        "name": "GPT-5.2 Codex",
        "model": "openai/openai/gpt-5.2-codex",
        "api_base": "https://inference-api.nvidia.com/v1/responses",
        "api_format": "responses",
        "context_window": 128000,
    },
    "2": {
        "name": "Claude Opus 4.6",
        "model": "aws/anthropic/bedrock-claude-opus-4-6",
        "api_base": "https://inference-api.nvidia.com/v1/chat/completions",
        "api_format": "chat_completions",
        "context_window": 200000,
    },
    "3": {
        "name": "Claude Opus 4.6 (via Claude Code)",
        "model": "claude-opus-4-20250514",
        "api_base": "https://api.anthropic.com/v1/messages",
        "api_format": "anthropic",
        "context_window": 200000,
        "claude_code_key": True,
    },
}


def _gather_project_files(project_root: Path) -> str:
    """Collect key project files for LLM workspace scan. Returns ~12K chars max."""
    BUDGET = 12000
    sections: list[str] = []
    used = 0

    # Top-level directory listing
    try:
        entries = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in project_root.iterdir()
            if not p.name.startswith(".")
        )
        listing = "  ".join(entries)
        sections.append(f"=== Directory structure ===\n{listing}")
        used += len(listing)
    except OSError:
        pass

    # Ordered list of candidate files to include
    candidates = [
        "README.md", "README.rst", "README",
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
        "package.json",
        "Cargo.toml", "go.mod",
        "Makefile", "makefile",
        "docker-compose.yml", "Dockerfile",
    ]

    for name in candidates:
        if used >= BUDGET:
            break
        path = project_root / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(errors="replace")
            # For package.json skip node_modules reference bloat; truncate large files
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            chunk = f"=== {name} ===\n{content}"
            sections.append(chunk)
            used += len(chunk)
        except OSError:
            pass

    # GitHub Actions workflows (first 2)
    workflows_dir = project_root / ".github" / "workflows"
    if workflows_dir.is_dir() and used < BUDGET:
        wf_files = sorted(workflows_dir.glob("*.yml"))[:2]
        for wf in wf_files:
            if used >= BUDGET:
                break
            try:
                content = wf.read_text(errors="replace")
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                chunk = f"=== .github/workflows/{wf.name} ===\n{content}"
                sections.append(chunk)
                used += len(chunk)
            except OSError:
                pass

    return "\n\n".join(sections)


async def _scan_workspace(project_root: Path, config: "Config", username: str) -> str:
    """Call the LLM to generate a project-specific MEMORY.md."""
    context = _gather_project_files(project_root)

    system = (
        "You write concise MEMORY.md files for a terminal AI agent called koi. "
        "The agent reads this file at the start of every session. "
        "Be brief: bullet points, no prose. Max 50 lines total."
    )

    user_name_line = username if username else "unknown"
    user_msg = f"""Analyze this project and write a MEMORY.md.

Include sections:
## User
- Name: {user_name_line}

## Project
- Name, purpose, tech stack (1-2 bullets)

## Commands
- Build, test, lint, run commands (only if found in the files)

## Structure
- Key directories and what they contain (only non-obvious ones)

## Notes
- Important conventions, gotchas, or env setup (only if found)

Omit any section if there's nothing meaningful to say.
Do NOT include a header "# Memory" — start directly with ## User.
Do NOT add advice about how to use memory.

Project files:
{context}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    llm = LLMClient(config)
    try:
        response = await llm.chat(messages)
        return response["choices"][0]["message"].get("content", "").strip()
    finally:
        await llm.close()


@main.command()
@click.option(
    "--force",
    is_flag=True,
    help="Recreate config/sandbox/AGENTS even if they already exist"
)
def init(force: bool):
    """Initialize or re-run setup for .koi directory."""
    koi_dir = Path.cwd() / ".koi"
    existing = koi_dir.exists()

    # Interactive setup (fall back to defaults if not a TTY)
    import sys
    interactive = sys.stdin.isatty()

    if interactive:
        console.print("\n🐠 Koi Agent Setup\n", style="bold blue")
        if existing:
            console.print("Re-running setup (existing config/skills preserved unless --force).\n")

        console.print("Select a model:")
        for key, preset in MODEL_PRESETS.items():
            console.print(f"  [{key}] {preset['name']} ({preset['model']})")

        choice = click.prompt(
            ">",
            type=click.Choice(list(MODEL_PRESETS.keys())),
            default="1",
            show_choices=False,
            show_default=False,
        )
        preset = MODEL_PRESETS[choice]

        api_base = preset["api_base"]

        if preset.get("claude_code_key"):
            cc_key = load_claude_code_api_key()
            if cc_key:
                masked = cc_key[:12] + "..." + cc_key[-4:]
                console.print(f"\n✅ Found Claude Code API key: {masked}")
                api_key = cc_key
            else:
                console.print(
                    "\n⚠️  No Claude Code API key found in ~/.claude.json.",
                    style="yellow",
                )
                console.print("Run 'claude auth login' first, or enter a key manually.")
                api_key = click.prompt("API Key (sk-ant-...)")
        else:
            api_key = click.prompt("\nAPI Key")
        if api_key and not preset.get("claude_code_key"):
            masked = api_key[:4] + "*" * (len(api_key) - 4)
            console.print(f"  Key: {masked}")

        username = click.prompt("\nYour name (for memory context)", default="")

        model = preset["model"]
        api_format = preset["api_format"]
        context_window = preset["context_window"]
    else:
        # Non-interactive defaults (CI)
        preset = MODEL_PRESETS["1"]
        model = preset["model"]
        api_base = preset["api_base"]
        api_key = ""
        api_format = preset["api_format"]
        context_window = preset["context_window"]
        username = ""

    # Create .koi directory structure (only if new or --force)
    if not existing or force:
        koi_dir.mkdir(exist_ok=True)
        (koi_dir / "cron-logs").mkdir(exist_ok=True)
        (koi_dir / "credentials").mkdir(exist_ok=True)

        # Copy bundled skills into .koi/skills/
        skills_dir = koi_dir / "skills"
        bundled_skills_dir = Path(__file__).parent / "bundled_skills"
        if bundled_skills_dir.exists():
            if skills_dir.exists() and force:
                shutil.rmtree(skills_dir)
            if not skills_dir.exists():
                shutil.copytree(bundled_skills_dir, skills_dir)
                # Remove __init__.py if copied (it's only needed for packaging)
                init_file = skills_dir / "__init__.py"
                if init_file.exists():
                    init_file.unlink()
            else:
                # Copy any new skills that don't already exist
                for skill in bundled_skills_dir.iterdir():
                    if skill.is_dir() and not (skills_dir / skill.name).exists():
                        shutil.copytree(skill, skills_dir / skill.name)

        # Create config from wizard selections
        config_path = koi_dir / "config.json"
        if not config_path.exists() or force:
            default_config = create_default_config(
                model=model,
                api_base=api_base,
                api_key=api_key,
                api_format=api_format,
                context_window=context_window,
            )
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)

        # Create agents file
        agents_path = koi_dir / "AGENTS.md"
        if not agents_path.exists() or force:
            with open(agents_path, "w") as f:
                f.write("""# Project Instructions

You are **Koi** — a terminal-based AI agent that lives in the user's project directory.

## Session Startup

Every session begins the same way — before responding to the user:

1. Read `.koi/MEMORY.md` to recall what you already know about this project.
2. Check `.koi/alerts/` for any pending alerts.
3. Then proceed with the user's request.

Do not ask permission. Do not skip this. Your memory resets between sessions — MEMORY.md is the only thing that survives.

## Core Behavior

- Learn from execution history to minimize tool/command invocations and prefer the shortest safe path to achieve the goal.
- When repeated steps or errors are observed, update the relevant skill(s) or memory to encode the more efficient path (e.g., avoid failing paths, use known working commands, consolidate commands where safe).
- If a faster workflow becomes the default, apply it directly in future runs without re-trying known failing actions.

## Memory Discipline

You have no memory between sessions. Anything not written to MEMORY.md is gone forever.

**Where to store learnings:**

- **Skill-specific** → Update the skill's `SKILL.md` directly (loaded on-demand, keeps MEMORY.md lean).
- **General** → Write to `MEMORY.md`: user preferences, environment quirks, project-wide patterns, cross-cutting mistakes.

**Never write skill-specific learnings to MEMORY.md.**

## Mistake Documentation

When something goes wrong:
- Skill-related → update that skill's `SKILL.md`
- General → document in `MEMORY.md`

## Output & Alerts

- In interactive sessions, always output results directly in the terminal. Do not use `create_alert` — just print the answer.
- Only use `create_alert` when running as a cron job (non-interactive).

## Skills

All skills live in `.koi/skills/`. Each skill is a directory containing a `SKILL.md` file. Use `read_skill` with the directory name to load a skill.
""")

        # Create default sandbox config
        sandbox_path = koi_dir / "sandbox.yaml"
        if not sandbox_path.exists() or force:
            with open(sandbox_path, "w") as f:
                f.write("""# Sandbox Security Configuration
# Controls what koi can access to prevent catastrophic mistakes

filesystem:
  # Directories koi can read/write freely (relative to project root)
  allowed_paths:
    - "."

  # Extra read-only paths
  readonly_paths:
    - "/usr/local"
    - "/opt/homebrew"

  # Blocked paths — NEVER accessible, even via shell commands
  blocked_paths:
    - "~/.aws"
    - "~/.ssh"
    - "~/.config"

# Environment variable control for shell commands
environment:
  allowlist:
    - PATH
    - HOME
    - USER
    - SHELL
    - LANG
    - LC_ALL
    - TERM
    - EDITOR
    - TMPDIR
    - PYTHONPATH
    - VIRTUAL_ENV
    - NODE_PATH
    - SSH_AUTH_SOCK
    - SSH_AGENT_PID

# Shell command restrictions
commands:
  blocked_patterns:
    - 'rm\\s+-rf\\s+/'
    - 'sudo\\s+rm\\s+-rf'
    - 'mkfs\\.'
    - 'dd\\s+if=.*of=/dev/'
    - 'DROP\\s+TABLE'
    - 'DROP\\s+DATABASE'
  confirm_patterns:
    - 'rm\\s+'
    - 'git\\s+push\\s+.*--force'
""")
    else:
        # Existing .koi — ensure subdirs exist (safe to re-create)
        koi_dir.mkdir(exist_ok=True)
        (koi_dir / "cron-logs").mkdir(exist_ok=True)
        (koi_dir / "credentials").mkdir(exist_ok=True)

        # Update config with new wizard selections
        config_path = koi_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                existing_cfg = json.load(f)
            existing_cfg["api_base"] = api_base
            existing_cfg["api_key"] = api_key
            existing_cfg["model"] = model
            existing_cfg["api_format"] = api_format
            existing_cfg["context_window"] = context_window
            with open(config_path, "w") as f:
                json.dump(existing_cfg, f, indent=2)
        else:
            default_config = create_default_config(
                model=model,
                api_base=api_base,
                api_key=api_key,
                api_format=api_format,
                context_window=context_window,
            )
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)

    # Scan workspace and write MEMORY.md (always)
    console.print("\n🔍 Scanning workspace...", style="blue")
    cfg = Config(
        api_base=api_base,
        api_key=api_key,
        model=model,
        api_format=api_format,
        context_window=context_window,
    )
    try:
        memory_content = asyncio.run(_scan_workspace(Path.cwd(), cfg, username))
        memory_path = koi_dir / "MEMORY.md"
        memory_path.write_text(memory_content)
        console.print("✅ MEMORY.md generated from workspace scan.", style="green")
    except Exception:
        console.print(
            "\n⚠️  Could not connect to API for workspace scan. "
            "You can fix settings in .koi/config.json later.",
            style="yellow",
        )
        memory_path = koi_dir / "MEMORY.md"
        if not memory_path.exists():
            memory_path.write_text(
                f"## User\n\n- Name: {username or 'unknown'}\n\n"
                "## Project\n\n- (Run `koi init` again after fixing API settings to auto-generate)\n"
            )

    console.print(
        f"\n✅ Koi {'updated' if existing else 'initialized'}!",
        style="green",
    )


@main.command()
def switch():
    """Switch model/backend without resetting skills, memory, or project instructions."""
    koi_dir = Path.cwd() / ".koi"
    config_path = koi_dir / "config.json"

    if not config_path.exists():
        console.print("❌ No .koi/config.json found. Run 'koi init' first.", style="red")
        return

    console.print("\n🐠 Switch Backend\n", style="bold blue")

    console.print("Select a model:")
    for key, preset in MODEL_PRESETS.items():
        console.print(f"  [{key}] {preset['name']} ({preset['model']})")

    choice = click.prompt(
        ">",
        type=click.Choice(list(MODEL_PRESETS.keys())),
        default="1",
        show_choices=False,
        show_default=False,
    )
    preset = MODEL_PRESETS[choice]

    api_base = preset["api_base"]

    if preset.get("claude_code_key"):
        cc_key = load_claude_code_api_key()
        if cc_key:
            masked = cc_key[:12] + "..." + cc_key[-4:]
            console.print(f"\n✅ Found Claude Code API key: {masked}")
            api_key = cc_key
        else:
            console.print(
                "\n⚠️  No Claude Code API key found in ~/.claude.json.",
                style="yellow",
            )
            console.print("Run 'claude auth login' first, or enter a key manually.")
            api_key = click.prompt("API Key (sk-ant-...)")
    else:
        api_key = click.prompt("\nAPI Key")

    # Load existing config to preserve non-backend settings
    with open(config_path, "r") as f:
        existing = json.load(f)

    existing["api_base"] = api_base
    existing["api_key"] = api_key
    existing["model"] = preset["model"]
    existing["api_format"] = preset["api_format"]
    existing["context_window"] = preset["context_window"]

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    console.print(f"\n✅ Switched to {preset['name']}", style="green")

    # Quick connection test
    try:
        cfg = Config.load(config_path)
        async def _test():
            llm = LLMClient(cfg)
            try:
                await llm.chat([
                    {"role": "user", "content": "Say ok"},
                ])
                return True
            finally:
                await llm.close()

        import asyncio
        asyncio.run(_test())
        console.print("✅ Connection verified", style="green")
    except Exception:
        console.print(
            "⚠️  Could not connect to API. Check .koi/config.json.",
            style="yellow",
        )


@main.command()
@click.option(
    "--task",
    help="Run a specific task and exit (for cron jobs)"
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Run without interactive prompt (for cron jobs)"
)
@click.option(
    "--thinking",
    type=click.Choice(["off", "minimal", "low", "medium", "high"], case_sensitive=False),
    default=None,
    help="Set thinking/reasoning level (overrides config)"
)
@click.option(
    "--result-file",
    default=None,
    help="Write final response to this JSON file (for sub-agent mode)"
)
@click.option(
    "--model",
    "model_override",
    default=None,
    help="Override the model from config"
)
def run(
    task: Optional[str],
    non_interactive: bool,
    thinking: Optional[str],
    result_file: Optional[str],
    model_override: Optional[str],
):
    """Start an interactive agent session or run a specific task."""
    try:
        config = Config.load()
        if thinking is not None:
            config.thinking_level = normalize_think_level(thinking) or thinking
        if model_override is not None:
            config.model = model_override

        agent = Agent(config, non_interactive=non_interactive or bool(task))

        if task:
            # Run specific task and exit
            asyncio.run(
                _run_task(agent, task, non_interactive, result_file)
            )
        else:
            # Interactive session
            asyncio.run(agent.run_interactive())

    except FileNotFoundError as e:
        if ".koi/config.json" in str(e):
            console.print("❌ No .koi/config.json found. Run 'koi init' first.", style="red")
        else:
            console.print(f"❌ File not found: {e}", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


async def _run_task(
    agent: "Agent",
    task: str,
    non_interactive: bool,
    result_file: Optional[str],
):
    """Run a task and optionally write the result to a JSON file."""
    await agent.run_task(task, non_interactive=non_interactive)

    if result_file:
        # Extract final assistant response from conversation
        response_text = ""
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                response_text = msg["content"]
                break

        import json as _json
        result = {
            "summary": response_text[:2000],
            "response": response_text,
            "message_count": len(agent.messages),
        }
        result_path = Path(result_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(_json.dumps(result, indent=2))


@main.command()
@click.argument("schedule")
@click.argument("task")
def add_cron(schedule: str, task: str):
    """Add a cron job. SCHEDULE format: '0 9 * * 1' (9 AM every Monday)."""
    try:
        cron_manager = CronManager()
        job_id = cron_manager.add_job(schedule, task)
        console.print(f"✅ Added cron job {job_id}", style="green")
        console.print(f"Schedule: {schedule}", style="blue")
        console.print(f"Task: {task}", style="blue")
    except Exception as e:
        console.print(f"❌ Error adding cron job: {e}", style="red")


@main.group(name="cron")
def cron_group():
    """Manage cron jobs."""
    pass


cron_group.add_command(add_cron, name="add")


@cron_group.command(name="list")
def list_cron():
    """List all registered cron jobs."""
    try:
        cron_manager = CronManager()
        jobs = cron_manager.list_jobs()
        
        if not jobs:
            console.print("No cron jobs registered.", style="yellow")
            return
        
        table = Table(title="Registered Cron Jobs")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Schedule", style="magenta")
        table.add_column("Task", style="green")
        table.add_column("Status", style="blue")
        
        for job in jobs:
            table.add_row(
                job["id"],
                job["schedule"],
                job["task"],
                "Active" if job.get("active", True) else "Inactive"
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"❌ Error listing cron jobs: {e}", style="red")


@cron_group.command(name="remove")
@click.argument("job_id")
def remove_cron(job_id: str):
    """Remove a cron job by ID."""
    try:
        cron_manager = CronManager()
        cron_manager.remove_job(job_id)
        console.print(f"✅ Removed cron job {job_id}", style="green")
    except Exception as e:
        console.print(f"❌ Error removing cron job: {e}", style="red")


@main.command()
def skills():
    """List available skills."""
    try:
        config = Config.load()
        skills_manager = SkillsManager(config.skills_paths)
        available_skills = skills_manager.list_skills()
        
        if not available_skills:
            console.print("No skills found.", style="yellow")
            return
        
        table = Table(title="Available Skills")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Path", style="blue")
        
        for skill in available_skills:
            table.add_row(
                skill["name"],
                skill["description"][:80] + "..." if len(skill["description"]) > 80 else skill["description"],
                str(skill["path"])
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"❌ Error listing skills: {e}", style="red")


@main.command()
def config():
    """Show current configuration."""
    try:
        config = Config.load()
        
        table = Table(title="Current Configuration")
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Hide API key for security
        api_key = config.api_key
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        else:
            masked_key = "Not set"
        
        table.add_row("API Base", config.api_base)
        table.add_row("API Key", masked_key)
        table.add_row("API Format", config.api_format)
        table.add_row("Model", config.model)
        table.add_row("Max Tokens", str(config.max_tokens))
        table.add_row("Context Window", str(config.context_window))
        table.add_row("Skills Paths", ", ".join(config.skills_paths))
        table.add_row("Temperature", str(config.temperature))
        table.add_row("Thinking Level", config.thinking_level)

        console.print(table)
    
    except Exception as e:
        console.print(f"❌ Error loading config: {e}", style="red")


@main.command()
def memory():
    """Show current memory."""
    try:
        memory = Memory()
        content = memory.load()
        
        if not content.strip():
            console.print("Memory is empty.", style="yellow")
            return
        
        console.print("[bold blue]Current Memory:[/bold blue]")
        console.print(Markdown(content))
    
    except Exception as e:
        console.print(f"❌ Error loading memory: {e}", style="red")


if __name__ == "__main__":
    main()