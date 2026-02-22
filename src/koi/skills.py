"""Skills management for koi agent."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import re


class SkillsManager:
    """Manage skills discovery and loading."""
    
    def __init__(self, skills_paths: List[str]):
        """Initialize with list of paths to search for skills."""
        self.skills_paths = [Path(p) for p in skills_paths]
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """Scan skills paths and return list of available skills."""
        skills = []
        
        for skills_path in self.skills_paths:
            if not skills_path.exists():
                continue
            
            # Find all SKILL.md files
            for skill_file in skills_path.rglob("SKILL.md"):
                try:
                    skill_info = self._parse_skill_file(skill_file)
                    if skill_info:
                        skill_info["path"] = skill_file
                        skills.append(skill_info)
                except Exception as e:
                    # Skip files that can't be parsed
                    print(f"Warning: Could not parse skill file {skill_file}: {e}")
                    continue
        
        return skills
    
    def read_skill(self, skill_name: str) -> Optional[str]:
        """Read the full content of a skill by name or directory name."""
        skills = self.list_skills()
        query = skill_name.lower().strip()

        for skill in skills:
            # Match on parsed title or parent directory name
            dir_name = skill["path"].parent.name.lower()
            if skill["name"].lower() == query or dir_name == query:
                try:
                    with open(skill["path"], "r", encoding="utf-8") as f:
                        return f.read()
                except Exception as e:
                    raise RuntimeError(f"Failed to read skill {skill_name}: {e}")

        raise FileNotFoundError(f"Skill '{skill_name}' not found")
    
    def get_skills_summary(self) -> str:
        """Get a summary of all available skills for system prompt."""
        skills = self.list_skills()
        
        if not skills:
            return "No skills available."
        
        summary_lines = ["Available skills:"]
        for skill in skills:
            name = skill["name"]
            description = skill["description"]
            summary_lines.append(f"- {name}: {description}")
        
        return "\n".join(summary_lines)
    
    def _parse_skill_file(self, skill_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a SKILL.md file to extract name and description."""
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract name from first heading or filename
            name_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            if name_match:
                name = name_match.group(1).strip()
            else:
                # Use parent directory name as skill name
                name = skill_file.parent.name
            
            # Extract description from content (first paragraph after title)
            lines = content.split("\n")
            description = ""
            
            # Skip title line and empty lines
            start_collecting = False
            for line in lines:
                line = line.strip()
                
                if line.startswith("# "):
                    start_collecting = True
                    continue
                
                if start_collecting:
                    if line and not line.startswith("#"):
                        if description:
                            description += " " + line
                        else:
                            description = line
                    elif line.startswith("#") and description:
                        # Stop at next heading
                        break
                    elif not line:
                        # Stop at empty line if we have description
                        if description:
                            break
            
            # Fallback description
            if not description:
                description = "No description available."
            
            # Truncate long descriptions
            if len(description) > 200:
                description = description[:200] + "..."
            
            return {
                "name": name,
                "description": description,
            }
        
        except Exception:
            return None