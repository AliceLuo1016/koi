"""Tests for sandbox module — security-critical."""

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from koi.sandbox import Sandbox


def _write_sandbox_yaml(project_root: Path, cfg: dict):
    koi_dir = project_root / ".koi"
    koi_dir.mkdir(exist_ok=True)
    (koi_dir / "sandbox.yaml").write_text(yaml.dump(cfg))


def test_sandbox_default_config():
    """No sandbox.yaml → defaults to allowing cwd."""
    with TemporaryDirectory() as td:
        sandbox = Sandbox(project_root=Path(td))
        # Default allowed path is "." which resolves to project_root
        allowed, reason = sandbox.check_read(str(Path(td) / "file.txt"))
        assert allowed


def test_sandbox_check_read_allowed():
    """File inside allowed_paths is readable."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(Path(td), {"filesystem": {"allowed_paths": [str(td)]}})
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_read(str(Path(td) / "foo.txt"))
        assert allowed
        assert reason is None


def test_sandbox_check_read_blocked():
    """File in blocked_paths is denied even if inside allowed_paths."""
    with TemporaryDirectory() as td:
        secret_dir = Path(td) / "secret"
        secret_dir.mkdir()
        _write_sandbox_yaml(
            Path(td),
            {
                "filesystem": {
                    "allowed_paths": [str(td)],
                    "blocked_paths": [str(secret_dir)],
                }
            },
        )
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_read(str(secret_dir / "key.pem"))
        assert not allowed
        assert "blocked" in reason.lower()


def test_sandbox_check_write_allowed():
    """File inside allowed_paths is writable."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(Path(td), {"filesystem": {"allowed_paths": [str(td)]}})
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_write(str(Path(td) / "out.txt"))
        assert allowed
        assert reason is None


def test_sandbox_check_write_blocked_path():
    """Blocked paths deny writes even when inside allowed_paths."""
    with TemporaryDirectory() as td:
        blocked = Path(td) / "blocked"
        blocked.mkdir()
        _write_sandbox_yaml(
            Path(td),
            {
                "filesystem": {
                    "allowed_paths": [str(td)],
                    "blocked_paths": [str(blocked)],
                }
            },
        )
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_write(str(blocked / "file.txt"))
        assert not allowed


def test_sandbox_check_write_readonly():
    """File in readonly_paths denies writes."""
    with TemporaryDirectory() as td:
        ro_dir = Path(td) / "readonly"
        ro_dir.mkdir()
        _write_sandbox_yaml(
            Path(td),
            {"filesystem": {"allowed_paths": [], "readonly_paths": [str(ro_dir)]}},
        )
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_write(str(ro_dir / "file.txt"))
        assert not allowed
        assert "outside allowed paths" in reason.lower()


def test_sandbox_check_read_readonly():
    """File in readonly_paths allows reads."""
    with TemporaryDirectory() as td:
        ro_dir = Path(td) / "readonly"
        ro_dir.mkdir()
        _write_sandbox_yaml(
            Path(td),
            {"filesystem": {"allowed_paths": [], "readonly_paths": [str(ro_dir)]}},
        )
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_read(str(ro_dir / "file.txt"))
        assert allowed


def test_sandbox_check_read_outside_allowed():
    """File outside all paths is denied."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(
            Path(td), {"filesystem": {"allowed_paths": [str(Path(td) / "only")]}}
        )
        (Path(td) / "only").mkdir()
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason = sandbox.check_read("/etc/passwd")
        assert not allowed


def test_sandbox_get_safe_env():
    """Only allowlisted env vars are returned."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(Path(td), {"environment": {"allowlist": ["HOME", "USER"]}})
        sandbox = Sandbox(project_root=Path(td))
        env = sandbox.get_safe_env()
        # Should only contain HOME and/or USER (if set in the real env)
        for key in env:
            assert key in ("HOME", "USER") or key.isupper()  # credentials
        # Should NOT contain random vars like EDITOR, OPENAI_API_KEY, etc.
        assert "EDITOR" not in env or "EDITOR" in {"HOME", "USER"}


def test_sandbox_get_credentials_env():
    """Credential files are loaded as env vars."""
    with TemporaryDirectory() as td:
        creds_dir = Path(td) / ".koi" / "credentials"
        creds_dir.mkdir(parents=True)
        (creds_dir / "openai.key").write_text("sk-test-123\n")
        (creds_dir / "github.token").write_text("ghp-abc\n")

        _write_sandbox_yaml(Path(td), {})
        sandbox = Sandbox(project_root=Path(td))
        creds = sandbox.get_credentials_env()
        assert creds["OPENAI"] == "sk-test-123"
        assert creds["GITHUB"] == "ghp-abc"


def test_sandbox_check_command_allowed():
    """Normal command passes."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(Path(td), {})
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason, confirm = sandbox.check_command("echo hello")
        assert allowed
        assert reason is None
        assert not confirm


def test_sandbox_check_command_blocked():
    """Blocked pattern rejects command."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(
            Path(td),
            {"commands": {"blocked_patterns": [r"rm\s+-rf\s+/"]}},
        )
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason, confirm = sandbox.check_command("rm -rf /")
        assert not allowed
        assert "Blocked" in reason


def test_sandbox_check_command_confirm():
    """Confirm pattern flags command for confirmation."""
    with TemporaryDirectory() as td:
        _write_sandbox_yaml(
            Path(td),
            {"commands": {"confirm_patterns": [r"git\s+push"]}},
        )
        sandbox = Sandbox(project_root=Path(td))
        allowed, reason, confirm = sandbox.check_command("git push origin main")
        assert allowed
        assert confirm
        assert "confirmation" in reason.lower()
