"""Migration system for upgrading .koi/ project files between versions."""

import json
from pathlib import Path

CURRENT_PROJECT_VERSION = "0.3.0"

MIGRATIONS: list[tuple[str, "callable"]] = []


def _is_default_agents(content: str) -> bool:
    """Check if AGENTS.md is still an old default template."""
    old_markers = [
        "Memory Discipline",
        "Do not ask permission",
        "Never write skill-specific learnings",
        "Mistake Documentation",
    ]
    return any(marker in content for marker in old_markers)


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple."""
    return tuple(int(x) for x in v.split("."))


def migrate_020(koi_dir: Path) -> list[str]:
    """Migration for v0.2.0 — lint + pre-commit era, no project changes."""
    return ["Version bump (no project changes)"]


def migrate_030(koi_dir: Path) -> list[str]:
    """Migration for v0.3.0 — memory system."""
    changes: list[str] = []

    # Create memory/ directory
    memory_dir = koi_dir.parent / "memory"
    if not memory_dir.exists():
        memory_dir.mkdir(parents=True)
        changes.append("Created memory/ directory")

    # Add memory_search defaults to config.json
    config_path = koi_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "memory_search" not in cfg:
            cfg["memory_search"] = {
                "provider": "openai",
                "model": "text-embedding-3-small",
            }
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            changes.append("Added memory_search config defaults")

    # Update AGENTS.md if it's still the old default
    agents_path = koi_dir / "AGENTS.md"
    if agents_path.exists():
        content = agents_path.read_text()
        from .cli import _DEFAULT_AGENTS_MD

        if content.strip() == _DEFAULT_AGENTS_MD.strip():
            # Already the new template
            pass
        elif _is_default_agents(content):
            agents_path.write_text(_DEFAULT_AGENTS_MD)
            changes.append("Updated AGENTS.md to new template")
        else:
            changes.append(
                "AGENTS.md has been customized \u2014 review the new template:\n"
                "    - Session startup now reads daily logs (memory/YYYY-MM-DD.md)\n"
                "    - Memory section documents the two-layer system\n"
                "    Run `koi init --force` to reset to the new default"
            )

    return changes


# Register migrations in order
MIGRATIONS = [
    ("0.2.0", migrate_020),
    ("0.3.0", migrate_030),
]


def get_pending_migrations(
    current_version: str,
) -> list[tuple[str, "callable"]]:
    """Return migrations that need to run to go from current_version to latest."""
    current = _version_tuple(current_version)
    return [(v, fn) for v, fn in MIGRATIONS if _version_tuple(v) > current]


def run_upgrade(koi_dir: Path) -> tuple[str, str, list[tuple[str, list[str]]]]:
    """Run all pending migrations on a .koi/ directory.

    Returns (old_version, new_version, [(version, changes), ...]).
    """
    version_file = koi_dir / "version"
    if version_file.exists():
        current = version_file.read_text().strip()
    else:
        current = "0.1.0"

    pending = get_pending_migrations(current)
    results: list[tuple[str, list[str]]] = []

    for version, migrate_fn in pending:
        changes = migrate_fn(koi_dir)
        results.append((version, changes))

    # Write new version
    if pending:
        new_version = pending[-1][0]
        version_file.write_text(new_version)
    else:
        new_version = current

    return current, new_version, results
