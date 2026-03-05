"""Tests for the migration system and koi upgrade command."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

from koi.cli import _DEFAULT_AGENTS_MD, main
from koi.migrations import (
    CURRENT_PROJECT_VERSION,
    _is_default_agents,
    get_pending_migrations,
    migrate_030,
    run_upgrade,
)

# ── _is_default_agents ──


def test_is_default_agents_detects_old_markers():
    """Old template markers are detected."""
    assert _is_default_agents("Some text\nMemory Discipline\nMore text")
    assert _is_default_agents("Do not ask permission to do things")
    assert _is_default_agents("Never write skill-specific learnings here")
    assert _is_default_agents("## Mistake Documentation\n")


def test_is_default_agents_returns_false_for_custom():
    """Custom content without old markers returns False."""
    assert not _is_default_agents("# My Custom Instructions\nDo cool stuff.")
    assert not _is_default_agents("")


# ── get_pending_migrations ──


def test_pending_from_010_includes_all():
    """Starting from 0.1.0, all migrations are pending."""
    pending = get_pending_migrations("0.1.0")
    versions = [v for v, _ in pending]
    assert "0.2.0" in versions
    assert "0.3.0" in versions


def test_pending_from_020_skips_020():
    """Starting from 0.2.0, only 0.3.0 is pending."""
    pending = get_pending_migrations("0.2.0")
    versions = [v for v, _ in pending]
    assert "0.2.0" not in versions
    assert "0.3.0" in versions


def test_pending_from_latest_is_empty():
    """At latest version, no migrations are pending."""
    pending = get_pending_migrations(CURRENT_PROJECT_VERSION)
    assert pending == []


# ── migrate_030 ──


def test_migrate_030_creates_memory_dir():
    """Migration 0.3.0 creates memory/ directory."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text("{}")

        changes = migrate_030(koi_dir)

        assert (Path(tmp) / "memory").is_dir()
        assert any("memory" in c.lower() for c in changes)


def test_migrate_030_adds_memory_search_config():
    """Migration 0.3.0 adds memory_search to config.json."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text(json.dumps({"model": "test"}))

        migrate_030(koi_dir)

        cfg = json.loads((koi_dir / "config.json").read_text())
        assert "memory_search" in cfg
        assert cfg["memory_search"]["provider"] == "openai"


def test_migrate_030_skips_existing_memory_search():
    """Migration 0.3.0 doesn't overwrite existing memory_search config."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        cfg = {"memory_search": {"provider": "custom", "model": "custom-model"}}
        (koi_dir / "config.json").write_text(json.dumps(cfg))

        migrate_030(koi_dir)

        result = json.loads((koi_dir / "config.json").read_text())
        assert result["memory_search"]["provider"] == "custom"


def test_migrate_030_updates_old_agents_md():
    """Migration 0.3.0 replaces old-template AGENTS.md with new default."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text("{}")
        old_content = "# Old Template\n\n## Memory Discipline\nDon't forget stuff.\n"
        (koi_dir / "AGENTS.md").write_text(old_content)

        changes = migrate_030(koi_dir)

        new_content = (koi_dir / "AGENTS.md").read_text()
        assert new_content == _DEFAULT_AGENTS_MD
        assert any("Updated AGENTS.md" in c for c in changes)


def test_migrate_030_preserves_customized_agents_md():
    """Migration 0.3.0 leaves customized AGENTS.md alone with a warning."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text("{}")
        custom_content = "# My Custom Agent\n\nDo my bidding.\n"
        (koi_dir / "AGENTS.md").write_text(custom_content)

        changes = migrate_030(koi_dir)

        # Content should be unchanged
        assert (koi_dir / "AGENTS.md").read_text() == custom_content
        # Should have a warning
        assert any("customized" in c for c in changes)


# ── run_upgrade (full pipeline) ──


def test_full_upgrade_010_to_latest():
    """Full upgrade from 0.1.0 creates memory dir, updates config, writes version."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text(json.dumps({"model": "test"}))
        old_agents = "# Old\n\nMemory Discipline rules.\n"
        (koi_dir / "AGENTS.md").write_text(old_agents)

        old_ver, new_ver, results = run_upgrade(koi_dir)

        assert old_ver == "0.1.0"
        assert new_ver == CURRENT_PROJECT_VERSION
        assert (koi_dir / "version").read_text() == CURRENT_PROJECT_VERSION
        assert (Path(tmp) / "memory").is_dir()

        cfg = json.loads((koi_dir / "config.json").read_text())
        assert "memory_search" in cfg


def test_upgrade_already_at_latest():
    """run_upgrade is a no-op when already at latest version."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "version").write_text(CURRENT_PROJECT_VERSION)

        old_ver, new_ver, results = run_upgrade(koi_dir)

        assert old_ver == CURRENT_PROJECT_VERSION
        assert new_ver == CURRENT_PROJECT_VERSION
        assert results == []


def test_version_file_written_after_upgrade():
    """Version file is created/updated after successful upgrade."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text("{}")

        run_upgrade(koi_dir)

        assert (koi_dir / "version").exists()
        assert (koi_dir / "version").read_text() == CURRENT_PROJECT_VERSION


# ── CLI: koi upgrade ──


def test_upgrade_command_no_koi_dir(monkeypatch):
    """upgrade command errors when .koi/ doesn't exist."""
    with TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        runner = CliRunner()
        result = runner.invoke(main, ["upgrade"])
        assert "koi init" in result.output


def test_upgrade_command_already_up_to_date(monkeypatch):
    """upgrade command reports up-to-date when at latest version."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "version").write_text(CURRENT_PROJECT_VERSION)

        monkeypatch.chdir(tmp)
        runner = CliRunner()
        result = runner.invoke(main, ["upgrade"])
        assert "up to date" in result.output.lower()


def test_upgrade_command_runs_migrations(monkeypatch):
    """upgrade command runs migrations and prints summary."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text(json.dumps({"model": "test"}))

        monkeypatch.chdir(tmp)
        runner = CliRunner()
        result = runner.invoke(main, ["upgrade"])

        assert "Upgrading" in result.output
        assert "v0.2.0" in result.output
        assert "v0.3.0" in result.output
        assert "Upgrade complete" in result.output
        assert (koi_dir / "version").read_text() == CURRENT_PROJECT_VERSION


def test_upgrade_command_shows_warning_for_custom_agents(monkeypatch):
    """upgrade command shows warning when AGENTS.md is customized."""
    with TemporaryDirectory() as tmp:
        koi_dir = Path(tmp) / ".koi"
        koi_dir.mkdir()
        (koi_dir / "config.json").write_text("{}")
        (koi_dir / "AGENTS.md").write_text("# My Custom Setup\nDo things.\n")

        monkeypatch.chdir(tmp)
        runner = CliRunner()
        result = runner.invoke(main, ["upgrade"])

        assert "customized" in result.output


# ── CLI: koi init writes version ──


def test_init_creates_version_file(monkeypatch):
    """init command writes .koi/version with current version."""
    with TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        runner = CliRunner()
        # Non-interactive (CI mode): stdin is not a TTY in CliRunner
        runner.invoke(main, ["init"])

        version_file = Path(tmp) / ".koi" / "version"
        assert version_file.exists()
        assert version_file.read_text() == CURRENT_PROJECT_VERSION
