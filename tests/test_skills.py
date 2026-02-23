"""Tests for skills module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from koi.skills import SkillsManager


def test_list_skills_empty():
    """No skills dirs → empty list."""
    with TemporaryDirectory() as td:
        sm = SkillsManager([str(Path(td) / "nonexistent")])
        assert sm.list_skills() == []


def test_list_skills_found():
    """Create a SKILL.md and verify discovery."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill\nDoes cool things.")

        sm = SkillsManager([td])
        skills = sm.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "My Skill"
        assert "cool things" in skills[0]["description"]


def test_read_skill_by_name():
    """Read skill by its parsed title."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "helper"
        skill_dir.mkdir()
        skill_content = "# Helper Tool\nHelps with stuff.\n\n## Usage\n..."
        (skill_dir / "SKILL.md").write_text(skill_content)

        sm = SkillsManager([td])
        content = sm.read_skill("Helper Tool")
        assert "Helper Tool" in content
        assert "Helps with stuff" in content


def test_read_skill_by_dir_name():
    """Read skill by directory name."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "log-monitor"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Log Monitor\nMonitors logs.")

        sm = SkillsManager([td])
        content = sm.read_skill("log-monitor")
        assert "Log Monitor" in content


def test_read_skill_not_found():
    """Raises FileNotFoundError for missing skill."""
    with TemporaryDirectory() as td:
        sm = SkillsManager([td])
        with pytest.raises(FileNotFoundError):
            sm.read_skill("nonexistent")


def test_get_skills_summary_empty():
    """No skills returns 'No skills available.'"""
    with TemporaryDirectory() as td:
        sm = SkillsManager([str(Path(td) / "nope")])
        assert sm.get_skills_summary() == "No skills available."


def test_get_skills_summary_with_skills():
    """Summary lists available skills."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill\nA test skill.")

        sm = SkillsManager([td])
        summary = sm.get_skills_summary()
        assert "Test Skill" in summary
        assert "A test skill" in summary


def test_parse_skill_file():
    """_parse_skill_file extracts name and description."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "demo"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        content = "# Demo Skill\nThis does demo things.\n\n## Details\nMore info."
        skill_file.write_text(content)

        sm = SkillsManager([td])
        result = sm._parse_skill_file(skill_file)
        assert result["name"] == "Demo Skill"
        assert result["description"] == "This does demo things."
