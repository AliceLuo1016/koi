"""Tests for cron module — all subprocess calls mocked."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from koi.cron import CronManager


@pytest.fixture
def cron_env():
    """Set up a temp dir as cwd with .koi structure for CronManager."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        (Path(td) / ".koi").mkdir()
        yield td
        os.chdir(old_cwd)


def test_cron_init(cron_env):
    """Creates dirs, loads empty jobs."""
    cm = CronManager()
    assert cm.logs_dir.exists()
    assert cm.list_jobs() == []


@patch("koi.cron.subprocess.run")
@patch("koi.cron.subprocess.Popen")
@patch("koi.cron.shutil.which", return_value="/usr/local/bin/koi")
def test_add_job(mock_which, mock_popen, mock_run, cron_env):
    """Mock subprocess, verify job metadata and script files."""
    # Mock crontab -l returning empty
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    # Mock crontab - writing
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    cm = CronManager()
    job_id = cm.add_job("0 * * * *", "check logs")

    assert job_id is not None
    assert len(job_id) == 8

    jobs = cm.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["schedule"] == "0 * * * *"
    assert jobs[0]["task"] == "check logs"
    assert jobs[0]["active"] is True

    # Verify script file was created
    scripts_dir = Path(cron_env) / ".koi" / "cron-scripts"
    assert (scripts_dir / f"{job_id}.sh").exists()
    assert (scripts_dir / f"{job_id}.task").exists()


@patch("koi.cron.subprocess.run")
@patch("koi.cron.subprocess.Popen")
@patch("koi.cron.shutil.which", return_value="/usr/local/bin/koi")
def test_remove_job(mock_which, mock_popen, mock_run, cron_env):
    """Mock subprocess, verify cleanup."""
    # Setup: add a job first
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    cm = CronManager()
    job_id = cm.add_job("0 * * * *", "test task")

    # Now mock crontab -l to return the job entry
    mock_run.return_value = MagicMock(
        returncode=0, stdout=f"0 * * * * some-script # koi-{job_id}\n"
    )

    cm.remove_job(job_id)

    assert cm.list_jobs() == []
    # Script files should be cleaned up
    scripts_dir = Path(cron_env) / ".koi" / "cron-scripts"
    assert not (scripts_dir / f"{job_id}.sh").exists()
    assert not (scripts_dir / f"{job_id}.task").exists()


def test_remove_job_not_found(cron_env):
    """Raises ValueError for unknown job ID."""
    cm = CronManager()
    with pytest.raises(ValueError, match="not found"):
        cm.remove_job("nonexistent")


def test_list_jobs_empty(cron_env):
    """Returns empty list when no jobs."""
    cm = CronManager()
    assert cm.list_jobs() == []


@patch("koi.cron.subprocess.run")
@patch("koi.cron.subprocess.Popen")
@patch("koi.cron.shutil.which", return_value="/usr/local/bin/koi")
def test_list_jobs_with_jobs(mock_which, mock_popen, mock_run, cron_env):
    """Returns populated list after adding jobs."""
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    cm = CronManager()
    cm.add_job("0 * * * *", "task one")
    cm.add_job("30 * * * *", "task two")

    jobs = cm.list_jobs()
    assert len(jobs) == 2


def test_load_jobs_no_file(cron_env):
    """Returns empty dict when no crontab.json."""
    cm = CronManager()
    assert cm._jobs_cache == {}


def test_load_jobs_corrupt(cron_env):
    """Returns empty dict for corrupt JSON."""
    crontab_file = Path(cron_env) / ".koi" / "crontab.json"
    crontab_file.write_text("not valid json{{{")
    cm = CronManager()
    assert cm._jobs_cache == {}


def test_save_jobs(cron_env):
    """_save_jobs writes valid JSON."""
    cm = CronManager()
    cm._jobs_cache = {"test-id": {"id": "test-id", "task": "hello"}}
    cm._save_jobs()

    data = json.loads(cm.crontab_file.read_text())
    assert data["test-id"]["task"] == "hello"
