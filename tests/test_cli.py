"""Tests for cli module — _gather_project_files, _scan_workspace, and CLI commands."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from koi.cli import _gather_project_files, _scan_workspace, main
from koi.config import Config

# ── _gather_project_files ──


def test_gather_project_files_returns_directory_structure():
    """Always includes a directory structure section."""
    with TemporaryDirectory() as tmp:
        result = _gather_project_files(Path(tmp))
        assert "=== Directory structure ===" in result


def test_gather_project_files_includes_readme():
    """README.md content is included in output."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "README.md").write_text("# My Project\nDoes stuff.")
        result = _gather_project_files(td)
        assert "=== README.md ===" in result
        assert "My Project" in result


def test_gather_project_files_includes_pyproject():
    """pyproject.toml is included when present."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "pyproject.toml").write_text('[project]\nname = "myapp"\n')
        result = _gather_project_files(td)
        assert "=== pyproject.toml ===" in result
        assert "myapp" in result


def test_gather_project_files_includes_package_json():
    """package.json is included when present."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "package.json").write_text('{"name": "myapp", "version": "1.0.0"}')
        result = _gather_project_files(td)
        assert "=== package.json ===" in result


def test_gather_project_files_includes_github_workflows():
    """GitHub Actions workflow files are included (first 2 only)."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        wf_dir = td / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text("name: CI\non: push\n")
        (wf_dir / "deploy.yml").write_text("name: Deploy\non: push\n")
        (wf_dir / "third.yml").write_text("name: Third\n")
        result = _gather_project_files(td)
        assert ".github/workflows/ci.yml" in result
        assert ".github/workflows/deploy.yml" in result
        # Third file may or may not appear depending on budget; just check the first two


def test_gather_project_files_large_file_truncated():
    """Files over 3000 chars are truncated with a marker."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "README.md").write_text("A" * 5000)
        result = _gather_project_files(td)
        assert "truncated" in result
        assert "A" * 5000 not in result


def test_gather_project_files_budget_caps_total_size():
    """Total output stays within a reasonable bound (budget ~12K)."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "README.md").write_text("x" * 3000)
        (td / "pyproject.toml").write_text("y" * 3000)
        (td / "package.json").write_text("z" * 3000)
        (td / "Makefile").write_text("m" * 3000)
        (td / "Cargo.toml").write_text("c" * 3000)
        result = _gather_project_files(td)
        # Budget is 12000; we should stop adding sections once budget is hit
        assert len(result) < 25000


def test_gather_project_files_excludes_dotfiles_from_listing():
    """Hidden entries (starting with .) are excluded from directory listing."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "src").mkdir()
        (td / ".hidden").mkdir()
        result = _gather_project_files(td)
        # src/ should appear in listing; .hidden should not
        assert "src/" in result
        # The listing line is the first content after the header
        dir_line = result.split("=== Directory structure ===\n")[1].split("\n")[0]
        assert ".hidden" not in dir_line


def test_gather_project_files_missing_candidate_skipped():
    """Candidate files that don't exist are silently skipped."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        # No files created — only dir listing
        result = _gather_project_files(td)
        assert "=== Directory structure ===" in result
        assert "=== README.md ===" not in result


def test_gather_project_files_sections_separated_by_blank_lines():
    """Multiple sections are separated by double newlines."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "README.md").write_text("# Hi")
        (td / "pyproject.toml").write_text("[project]")
        result = _gather_project_files(td)
        assert "\n\n" in result


# ── _scan_workspace ──


async def test_scan_workspace_returns_llm_content():
    """_scan_workspace calls LLMClient.chat and returns stripped content."""
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        (td / "README.md").write_text("# Koi\nA fish agent.\n")
        config = Config(
            api_base="https://api.example.com",
            api_key="test-key",
            model="test-model",
        )
        mock_response = {
            "choices": [
                {"message": {"content": "## User\n- Name: Alice\n\n## Project\n- Koi"}}
            ]
        }
        with patch("koi.cli.LLMClient") as mock_llm:
            instance = AsyncMock()
            instance.chat = AsyncMock(return_value=mock_response)
            instance.close = AsyncMock()
            mock_llm.return_value = instance

            result = await _scan_workspace(td, config, "Alice")

        assert "## User" in result
        assert "Alice" in result


async def test_scan_workspace_strips_whitespace():
    """_scan_workspace strips leading/trailing whitespace from LLM response."""
    with TemporaryDirectory() as tmp:
        config = Config(api_base="https://x.com", api_key="k", model="m")
        mock_response = {
            "choices": [{"message": {"content": "  \n## User\n- Name: Bob\n\n  "}}]
        }

        with patch("koi.cli.LLMClient") as mock_llm:
            instance = AsyncMock()
            instance.chat = AsyncMock(return_value=mock_response)
            instance.close = AsyncMock()
            mock_llm.return_value = instance

            result = await _scan_workspace(Path(tmp), config, "Bob")

        assert result == "## User\n- Name: Bob"


async def test_scan_workspace_empty_content_returns_empty_string():
    """_scan_workspace returns empty string when LLM returns only whitespace."""
    with TemporaryDirectory() as tmp:
        config = Config(api_base="https://x.com", api_key="k", model="m")
        mock_response = {"choices": [{"message": {"content": "  "}}]}

        with patch("koi.cli.LLMClient") as mock_llm:
            instance = AsyncMock()
            instance.chat = AsyncMock(return_value=mock_response)
            instance.close = AsyncMock()
            mock_llm.return_value = instance

            result = await _scan_workspace(Path(tmp), config, "")

        assert result == ""


async def test_scan_workspace_always_closes_client():
    """LLMClient.close() is called even when chat raises."""
    with TemporaryDirectory() as tmp:
        config = Config(api_base="https://x.com", api_key="k", model="m")

        with patch("koi.cli.LLMClient") as mock_llm:
            instance = AsyncMock()
            instance.chat = AsyncMock(side_effect=RuntimeError("boom"))
            instance.close = AsyncMock()
            mock_llm.return_value = instance

            with pytest.raises(RuntimeError, match="boom"):
                await _scan_workspace(Path(tmp), config, "Bob")

            instance.close.assert_awaited_once()


async def test_scan_workspace_passes_username_in_prompt():
    """Username is embedded in the user message sent to the LLM."""
    with TemporaryDirectory() as tmp:
        config = Config(api_base="https://x.com", api_key="k", model="m")
        captured = {}

        async def capture_chat(messages):
            captured["messages"] = messages
            return {"choices": [{"message": {"content": "## User\n- Name: Charlie"}}]}

        with patch("koi.cli.LLMClient") as mock_llm:
            instance = AsyncMock()
            instance.chat = AsyncMock(side_effect=capture_chat)
            instance.close = AsyncMock()
            mock_llm.return_value = instance

            await _scan_workspace(Path(tmp), config, "Charlie")

        user_msg = captured["messages"][1]["content"]
        assert "Charlie" in user_msg


# ── CLI commands via CliRunner ──


def test_config_command_shows_settings():
    """config command displays model and format in table."""
    runner = CliRunner()
    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="sk-test-1234",
        model="test-model",
        api_format="responses",
        context_window=128000,
    )
    with patch("koi.cli.Config.load", return_value=config):
        result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert "test-model" in result.output
    assert "responses" in result.output


def test_config_command_masks_api_key():
    """config command never exposes the full API key."""
    runner = CliRunner()
    config = Config(
        api_base="https://api.example.com",
        api_key="sk-supersecretkey1234",
        model="m",
    )
    with patch("koi.cli.Config.load", return_value=config):
        result = runner.invoke(main, ["config"])
    assert "sk-supersecretkey1234" not in result.output


def test_config_command_no_api_key():
    """config command shows 'Not set' when api_key is empty."""
    runner = CliRunner()
    config = Config(api_base="https://x.com", api_key="", model="m")
    with patch("koi.cli.Config.load", return_value=config):
        result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert "Not set" in result.output


def test_config_command_error():
    """config command handles Config.load error gracefully."""
    runner = CliRunner()
    with patch("koi.cli.Config.load", side_effect=RuntimeError("no config")):
        result = runner.invoke(main, ["config"])
    assert result.exit_code == 0
    assert "Error" in result.output


def test_memory_command_shows_content():
    """memory command renders memory content."""
    runner = CliRunner()
    with patch("koi.cli.Memory") as mock_memory:
        mock_memory.return_value.load.return_value = "## User\n- Name: Alice\n"
        result = runner.invoke(main, ["memory"])
    assert result.exit_code == 0
    assert "User" in result.output


def test_memory_command_empty():
    """memory command reports empty when memory contains only whitespace."""
    runner = CliRunner()
    with patch("koi.cli.Memory") as mock_memory:
        mock_memory.return_value.load.return_value = "   "
        result = runner.invoke(main, ["memory"])
    assert result.exit_code == 0
    assert "empty" in result.output.lower()


def test_memory_command_error():
    """memory command handles Memory instantiation error gracefully."""
    runner = CliRunner()
    with patch("koi.cli.Memory", side_effect=RuntimeError("disk error")):
        result = runner.invoke(main, ["memory"])
    assert result.exit_code == 0
    assert "Error" in result.output


def test_skills_command_shows_table():
    """skills command displays available skills."""
    runner = CliRunner()
    with (
        patch("koi.cli.Config.load") as mock_cfg,
        patch("koi.cli.SkillsManager") as mock_sm,
    ):
        mock_cfg.return_value = Config(api_base="x", api_key="k", model="m")
        mock_sm.return_value.list_skills.return_value = [
            {
                "name": "deploy",
                "description": "Deploy the app",
                "path": "/skills/deploy",
            }
        ]
        result = runner.invoke(main, ["skills"])
    assert result.exit_code == 0
    assert "deploy" in result.output


def test_skills_command_long_description_truncated():
    """skills command truncates descriptions longer than 80 chars."""
    runner = CliRunner()
    long_desc = "x" * 100
    with (
        patch("koi.cli.Config.load") as mock_cfg,
        patch("koi.cli.SkillsManager") as mock_sm,
    ):
        mock_cfg.return_value = Config(api_base="x", api_key="k", model="m")
        mock_sm.return_value.list_skills.return_value = [
            {"name": "big", "description": long_desc, "path": "/skills/big"}
        ]
        result = runner.invoke(main, ["skills"])
    assert result.exit_code == 0
    # Rich renders long strings with a Unicode ellipsis (…) or three dots (...)
    assert "…" in result.output or "..." in result.output


def test_skills_command_empty():
    """skills command reports when no skills are found."""
    runner = CliRunner()
    with (
        patch("koi.cli.Config.load") as mock_cfg,
        patch("koi.cli.SkillsManager") as mock_sm,
    ):
        mock_cfg.return_value = Config(api_base="x", api_key="k", model="m")
        mock_sm.return_value.list_skills.return_value = []
        result = runner.invoke(main, ["skills"])
    assert result.exit_code == 0
    assert "No skills" in result.output


def test_skills_command_error():
    """skills command handles Config.load error gracefully."""
    runner = CliRunner()
    with patch("koi.cli.Config.load", side_effect=RuntimeError("broken")):
        result = runner.invoke(main, ["skills"])
    assert result.exit_code == 0
    assert "Error" in result.output


def test_add_cron_command_success():
    """cron add calls CronManager.add_job and prints job ID."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.add_job.return_value = "job-123"
        result = runner.invoke(main, ["cron", "add", "0 9 * * 1", "check things"])
    assert result.exit_code == 0
    assert "job-123" in result.output


def test_add_cron_command_error():
    """cron add handles CronManager exceptions gracefully."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.add_job.side_effect = RuntimeError("cron broken")
        result = runner.invoke(main, ["cron", "add", "bad", "task"])
    assert result.exit_code == 0
    assert "Error" in result.output


def test_list_cron_command_shows_jobs():
    """cron list displays jobs in a table."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.list_jobs.return_value = [
            {
                "id": "job-1",
                "schedule": "0 9 * * 1",
                "task": "check things",
                "active": True,
            }
        ]
        result = runner.invoke(main, ["cron", "list"])
    assert result.exit_code == 0
    assert "job-1" in result.output


def test_list_cron_command_inactive_job():
    """cron list shows inactive status for jobs with active=False."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.list_jobs.return_value = [
            {"id": "job-2", "schedule": "0 * * * *", "task": "stuff", "active": False}
        ]
        result = runner.invoke(main, ["cron", "list"])
    assert result.exit_code == 0
    assert "Inactive" in result.output


def test_list_cron_command_empty():
    """cron list reports when no cron jobs exist."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.list_jobs.return_value = []
        result = runner.invoke(main, ["cron", "list"])
    assert result.exit_code == 0
    assert "No cron jobs" in result.output


def test_list_cron_command_error():
    """cron list handles CronManager instantiation error gracefully."""
    runner = CliRunner()
    with patch("koi.cli.CronManager", side_effect=RuntimeError("cron fail")):
        result = runner.invoke(main, ["cron", "list"])
    assert result.exit_code == 0
    assert "Error" in result.output


def test_remove_cron_command_success():
    """cron remove calls CronManager.remove_job and prints job ID."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.remove_job.return_value = None
        result = runner.invoke(main, ["cron", "remove", "job-123"])
    assert result.exit_code == 0
    assert "job-123" in result.output


def test_remove_cron_command_error():
    """cron remove handles error gracefully."""
    runner = CliRunner()
    with patch("koi.cli.CronManager") as mock_cron:
        mock_cron.return_value.remove_job.side_effect = RuntimeError("not found")
        result = runner.invoke(main, ["cron", "remove", "bad-id"])
    assert result.exit_code == 0
    assert "Error" in result.output


def test_run_command_missing_config():
    """run command prints helpful error when .koi/config.json is missing."""
    runner = CliRunner()
    with patch(
        "koi.cli.Config.load", side_effect=FileNotFoundError(".koi/config.json")
    ):
        result = runner.invoke(main, ["run"])
    assert result.exit_code == 0
    assert "koi init" in result.output


def test_run_command_general_error():
    """run command prints error on unexpected exception."""
    runner = CliRunner()
    with patch("koi.cli.Config.load", side_effect=RuntimeError("something blew up")):
        result = runner.invoke(main, ["run"])
    assert result.exit_code == 0
    assert "Error" in result.output
