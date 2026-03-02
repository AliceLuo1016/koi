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


def test_parse_skill_no_heading():
    """Skill without heading uses directory name."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "my-tool"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("Just a description, no heading.")

        sm = SkillsManager([td])
        skills = sm.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "my-tool"
        assert skills[0]["description"]  # fallback description


def test_parse_skill_long_description_truncated():
    """Long descriptions are truncated to 200 chars."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "verbose"
        skill_dir.mkdir()
        long_desc = "A" * 300
        (skill_dir / "SKILL.md").write_text(f"# Verbose\n{long_desc}")

        sm = SkillsManager([td])
        skills = sm.list_skills()
        assert len(skills[0]["description"]) <= 203  # 200 + "..."


def test_list_skills_skips_unparseable():
    """Unparseable SKILL.md files are skipped."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "bad"
        skill_dir.mkdir()
        # Create a SKILL.md that's actually a directory (will fail to open)
        bad_path = skill_dir / "SKILL.md"
        bad_path.mkdir()

        sm = SkillsManager([td])
        skills = sm.list_skills()
        assert len(skills) == 0


def test_read_skill_case_insensitive():
    """Skill lookup is case-insensitive."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "MySkill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Cool Skill\nDoes stuff.")

        sm = SkillsManager([td])
        content = sm.read_skill("my cool skill")
        assert "My Cool Skill" in content


def test_multiple_skills_paths():
    """Skills discovered from multiple paths."""
    with TemporaryDirectory() as td:
        dir_a = Path(td) / "a"
        dir_b = Path(td) / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        sa = dir_a / "skill-a"
        sb = dir_b / "skill-b"
        sa.mkdir()
        sb.mkdir()
        (sa / "SKILL.md").write_text("# Skill A\nFirst.")
        (sb / "SKILL.md").write_text("# Skill B\nSecond.")

        sm = SkillsManager([str(dir_a), str(dir_b)])
        skills = sm.list_skills()
        names = {s["name"] for s in skills}
        assert names == {"Skill A", "Skill B"}


def test_parse_skill_no_description():
    """Skill with heading but no description text."""
    with TemporaryDirectory() as td:
        skill_dir = Path(td) / "empty-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Empty\n## Section\nContent here.")

        sm = SkillsManager([td])
        skills = sm.list_skills()
        assert len(skills) == 1
        # Should have fallback description
        assert skills[0]["description"]
