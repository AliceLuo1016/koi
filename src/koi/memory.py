"""Memory management for koi agent."""

from pathlib import Path
from typing import Optional


class Memory:
    """Handle loading and saving persistent memory."""
    
    def __init__(self, memory_path: Optional[Path] = None):
        """Initialize memory with path to MEMORY.md file."""
        if memory_path is None:
            memory_path = Path.cwd() / ".koi" / "MEMORY.md"
        self.memory_path = memory_path
    
    def load(self) -> str:
        """Load memory content from MEMORY.md file."""
        if not self.memory_path.exists():
            return ""
        
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load memory from {self.memory_path}: {e}")
    
    def save(self, content: str):
        """Save content to MEMORY.md file."""
        try:
            # Ensure parent directory exists
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.memory_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise RuntimeError(f"Failed to save memory to {self.memory_path}: {e}")
    
    def append(self, content: str):
        """Append content to MEMORY.md file."""
        existing_content = self.load()
        
        # Add newlines for proper formatting
        if existing_content and not existing_content.endswith("\n"):
            existing_content += "\n"
        
        if content and not content.startswith("\n"):
            content = "\n" + content
        
        new_content = existing_content + content
        self.save(new_content)
    
    def exists(self) -> bool:
        """Check if memory file exists."""
        return self.memory_path.exists()
    
    def get_path(self) -> Path:
        """Get the path to the memory file."""
        return self.memory_path