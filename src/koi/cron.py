"""System crontab management for koi agent."""

import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class CronManager:
    """Manage system crontab entries for scheduled tasks."""
    
    def __init__(self):
        """Initialize cron manager."""
        self.koi_dir = Path.cwd() / ".koi"
        self.crontab_file = self.koi_dir / "crontab.json"
        self.logs_dir = self.koi_dir / "cron-logs"
        
        # Ensure directories exist
        self.koi_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Load existing jobs
        self._jobs_cache = self._load_jobs()
    
    def add_job(self, schedule: str, task: str) -> str:
        """Add a new cron job."""
        # Generate unique job ID
        job_id = str(uuid.uuid4())[:8]

        # Get current working directory
        project_path = Path.cwd().absolute()

        # Log filename format: DATE_taskname.log (lowercase, sanitized)
        safe_task_name = re.sub(r'[^\w\-]', '_', task).lower()[:50].strip('_')
        log_filename = f"{datetime.now().strftime('%Y-%m-%d')}_{safe_task_name}.log"
        log_path = self.logs_dir / log_filename

        # Find full path to koi binary
        koi_path = shutil.which("koi")
        if not koi_path:
            raise RuntimeError("Could not find 'koi' in PATH. Make sure koi is installed and accessible.")

        # Capture current PATH so cron has access to the same tools (uv, python, etc.)
        current_path = os.environ.get("PATH", "/usr/bin:/bin")

        # Write a launcher script to avoid crontab line length limits
        scripts_dir = self.koi_dir / "cron-scripts"
        scripts_dir.mkdir(exist_ok=True)
        script_path = scripts_dir / f"{job_id}.sh"
        script_path.write_text(
            f"#!/usr/bin/env bash\n"
            f"export PATH={current_path}\n"
            f"cd {project_path}\n"
            f"{koi_path} run --task \"$(cat {scripts_dir / (job_id + '.task')})\" "
            f"--non-interactive >> {log_path} 2>&1\n",
            encoding="utf-8",
        )
        script_path.chmod(0o755)

        # Write the task text to a separate file (avoids all quoting issues)
        task_path = scripts_dir / f"{job_id}.task"
        task_path.write_text(task, encoding="utf-8")

        # Cron entry is now short
        cron_command = f"{schedule} {script_path}"
        
        # Add job to system crontab
        self._add_to_system_crontab(cron_command, job_id)
        
        # Store job metadata
        job_data = {
            "id": job_id,
            "schedule": schedule,
            "task": task,
            "command": cron_command,
            "created": datetime.now().isoformat(),
            "active": True,
            "project_path": str(project_path)
        }
        
        self._jobs_cache[job_id] = job_data
        self._save_jobs()
        
        return job_id
    
    def remove_job(self, job_id: str):
        """Remove a cron job."""
        if job_id not in self._jobs_cache:
            raise ValueError(f"Job {job_id} not found")

        job = self._jobs_cache[job_id]

        # Remove from system crontab
        self._remove_from_system_crontab(job["command"])

        # Clean up script and task files
        scripts_dir = self.koi_dir / "cron-scripts"
        for ext in (".sh", ".task"):
            f = scripts_dir / f"{job_id}{ext}"
            if f.exists():
                f.unlink()

        # Remove from local cache
        del self._jobs_cache[job_id]
        self._save_jobs()
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all registered cron jobs."""
        return list(self._jobs_cache.values())
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job by ID."""
        return self._jobs_cache.get(job_id)
    
    def _load_jobs(self) -> Dict[str, Any]:
        """Load jobs from local JSON file."""
        if not self.crontab_file.exists():
            return {}
        
        try:
            with open(self.crontab_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_jobs(self):
        """Save jobs to local JSON file."""
        try:
            with open(self.crontab_file, "w") as f:
                json.dump(self._jobs_cache, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to save crontab data: {e}")
    
    def _add_to_system_crontab(self, command: str, job_id: str):
        """Add command to system crontab with job ID comment."""
        try:
            # Add comment with job ID for identification
            commented_command = f"{command} # koi-{job_id}"
            
            # Get current crontab
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            
            current_crontab = result.stdout if result.returncode == 0 else ""
            
            # Add new job
            new_crontab = current_crontab
            if new_crontab and not new_crontab.endswith('\n'):
                new_crontab += '\n'
            new_crontab += commented_command + '\n'
            
            # Write updated crontab
            process = subprocess.Popen(
                ["crontab", "-"],
                stdin=subprocess.PIPE,
                text=True
            )
            process.communicate(input=new_crontab)
            
            if process.returncode != 0:
                raise RuntimeError("Failed to update crontab")
        
        except Exception as e:
            raise RuntimeError(f"Failed to add cron job: {e}")
    
    def _remove_from_system_crontab(self, command: str):
        """Remove command from system crontab."""
        try:
            # Get current crontab
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return  # No crontab exists
            
            # Remove lines containing our command
            lines = result.stdout.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip lines that contain our command (ignoring the comment part)
                if command.split(' # ')[0] not in line:
                    filtered_lines.append(line)
            
            # Write updated crontab
            new_crontab = '\n'.join(filtered_lines)
            
            process = subprocess.Popen(
                ["crontab", "-"],
                stdin=subprocess.PIPE,
                text=True
            )
            process.communicate(input=new_crontab)
            
            if process.returncode != 0:
                raise RuntimeError("Failed to update crontab")
        
        except Exception as e:
            raise RuntimeError(f"Failed to remove cron job: {e}")
    
    def clean_orphaned_jobs(self):
        """Clean up jobs that exist in local cache but not in system crontab."""
        try:
            # Get current crontab
            result = subprocess.run(
                ["crontab", "-l"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # No crontab, mark all jobs as inactive
                for job_id in self._jobs_cache:
                    self._jobs_cache[job_id]["active"] = False
                self._save_jobs()
                return
            
            current_crontab = result.stdout
            
            # Check each job
            for job_id, job in self._jobs_cache.items():
                command_base = job["command"].split(" # ")[0]
                if command_base not in current_crontab:
                    job["active"] = False
                else:
                    job["active"] = True
            
            self._save_jobs()
        
        except Exception as e:
            print(f"Warning: Failed to clean orphaned jobs: {e}")
    
    def get_logs_dir(self) -> Path:
        """Get the path to the cron logs directory."""
        return self.logs_dir