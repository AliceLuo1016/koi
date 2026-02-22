"""CLI commands for koi agent."""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from .agent import Agent
from .config import Config, create_default_config
from .cron import CronManager
from .memory import Memory
from .skills import SkillsManager

console = Console()


@click.group()
@click.version_option()
def main():
    """Koi - Terminal-based AI agent with memory, tool calling, skills, and system cron."""
    pass


@main.command()
@click.option(
    "--force", 
    is_flag=True, 
    help="Overwrite existing .agent directory"
)
def init(force: bool):
    """Initialize .agent directory with config template and empty memory."""
    agent_dir = Path.cwd() / ".agent"
    
    if agent_dir.exists() and not force:
        console.print("❌ .agent directory already exists. Use --force to overwrite.", style="red")
        return
    
    # Create .agent directory structure
    agent_dir.mkdir(exist_ok=True)
    (agent_dir / "cron-logs").mkdir(exist_ok=True)
    (agent_dir / "credentials").mkdir(exist_ok=True)

    # Copy bundled skills into .agent/skills/
    skills_dir = agent_dir / "skills"
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
    
    # Create default config
    config_path = agent_dir / "config.json"
    if not config_path.exists() or force:
        default_config = create_default_config()
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
    
    # Create empty memory file
    memory_path = agent_dir / "MEMORY.md"
    if not memory_path.exists() or force:
        with open(memory_path, "w") as f:
            f.write("# Memory\n\nThis is your persistent memory. Write down important things to remember.\n")
    
    # Create empty agents file
    agents_path = agent_dir / "AGENTS.md"
    if not agents_path.exists() or force:
        with open(agents_path, "w") as f:
            f.write("# Project Instructions\n\n## Skills\n\nAll skills live in `.agent/skills/`. Each skill is a directory containing a `SKILL.md` file. Use `read_skill` with the directory name to load a skill.\n")
    
    # Create default sandbox config
    sandbox_path = agent_dir / "sandbox.yaml"
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
    
    console.print("✅ Initialized .agent directory", style="green")
    console.print(f"📝 Edit {config_path} to configure your API settings", style="yellow")


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
def run(task: Optional[str], non_interactive: bool):
    """Start an interactive agent session or run a specific task."""
    try:
        config = Config.load()
        agent = Agent(config, non_interactive=bool(task or non_interactive))

        if task:
            # Run specific task and exit
            asyncio.run(agent.run_task(task, non_interactive=non_interactive))
        else:
            # Interactive session
            asyncio.run(agent.run_interactive())
    
    except FileNotFoundError as e:
        if ".agent/config.json" in str(e):
            console.print("❌ No .agent/config.json found. Run 'koi init' first.", style="red")
        else:
            console.print(f"❌ File not found: {e}", style="red")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


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
        table.add_row("Model", config.model)
        table.add_row("Max Tokens", str(config.max_tokens))
        table.add_row("Context Window", str(config.context_window))
        table.add_row("Skills Paths", ", ".join(config.skills_paths))
        table.add_row("Temperature", str(config.temperature))
        
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
        console.print(content)
    
    except Exception as e:
        console.print(f"❌ Error loading memory: {e}", style="red")


if __name__ == "__main__":
    main()