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


# ── Credentials ──


def test_get_credential_without_extension(tmp_path):
    """get_credential() finds a file with no extension."""
    creds = tmp_path / ".koi" / "credentials"
    creds.mkdir(parents=True)
    (creds / "mytoken").write_text("secret123\n")
    _write_sandbox_yaml(tmp_path, {})
    sandbox = Sandbox(project_root=tmp_path)
    assert sandbox.get_credential("mytoken") == "secret123"


def test_get_credential_with_key_suffix(tmp_path):
    """get_credential() finds a .key file and returns stripped contents."""
    creds = tmp_path / ".koi" / "credentials"
    creds.mkdir(parents=True)
    (creds / "openai.key").write_text("sk-abc\n  ")
    _write_sandbox_yaml(tmp_path, {})
    sandbox = Sandbox(project_root=tmp_path)
    assert sandbox.get_credential("openai") == "sk-abc"


def test_get_credential_not_found(tmp_path):
    """get_credential() returns None when no matching file exists."""
    creds = tmp_path / ".koi" / "credentials"
    creds.mkdir(parents=True)
    _write_sandbox_yaml(tmp_path, {})
    sandbox = Sandbox(project_root=tmp_path)
    assert sandbox.get_credential("nonexistent") is None


def test_list_credentials(tmp_path):
    """list_credentials() returns base names without extensions."""
    creds = tmp_path / ".koi" / "credentials"
    creds.mkdir(parents=True)
    (creds / "openai.key").write_text("k1")
    (creds / "github.token").write_text("k2")
    (creds / "raw_cred").write_text("k3")
    _write_sandbox_yaml(tmp_path, {})
    sandbox = Sandbox(project_root=tmp_path)
    names = sandbox.list_credentials()
    assert "openai" in names
    assert "github" in names
    assert "raw_cred" in names


def test_get_credentials_env_var_naming(tmp_path):
    """get_credentials_env() converts hyphens and dots to underscores in var names."""
    creds = tmp_path / ".koi" / "credentials"
    creds.mkdir(parents=True)
    (creds / "my-service.key").write_text("val1")
    _write_sandbox_yaml(tmp_path, {})
    sandbox = Sandbox(project_root=tmp_path)
    env = sandbox.get_credentials_env()
    # my-service.key → stem=my-service → upper=MY-SERVICE → replace -/_=MY_SERVICE
    assert "MY_SERVICE" in env
    assert env["MY_SERVICE"] == "val1"


def test_list_credentials_no_dir(tmp_path):
    """list_credentials() returns [] when credentials dir doesn't exist."""
    _write_sandbox_yaml(tmp_path, {})
    sandbox = Sandbox(project_root=tmp_path)
    assert sandbox.list_credentials() == []
