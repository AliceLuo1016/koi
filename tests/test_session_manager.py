"""Tests for SessionManager."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from koi.session_manager import SessionManager


@pytest.fixture
def koi_dir(tmp_path):
    """Provide a temporary .koi directory."""
    d = tmp_path / ".koi"
    d.mkdir()
    return d


class TestStartSession:
    def test_creates_session_file(self, koi_dir):
        sm = SessionManager(koi_dir)
        sid = sm.start_session(model="claude-sonnet-4")

        assert sid is not None
        assert sm.session_path.exists()
        assert sm.session_path.suffix == ".jsonl"
        sm.close()

    def test_writes_header(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="claude-sonnet-4", cwd="/tmp/test")
        sm.close()

        with open(sm.session_path) as f:
            header = json.loads(f.readline())

        assert header["type"] == "session"
        assert header["version"] == 2
        assert header["model"] == "claude-sonnet-4"
        assert header["cwd"] == "/tmp/test"
        assert "id" in header
        assert "timestamp" in header


class TestSaveAndLoad:
    def test_round_trip_messages(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test-model")

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        for msg in msgs:
            sm.save_message(msg)
        sm.close()

        # Load it back
        data = sm.load_session()
        assert len(data["messages"]) == 3
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["role"] == "assistant"
        assert data["header"]["model"] == "test-model"

    def test_round_trip_with_tool_calls(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test-model")

        assistant_msg = {
            "role": "assistant",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'}}],
        }
        tool_result = {
            "role": "tool",
            "tool_call_id": "tc1",
            "content": "file contents here",
        }
        sm.save_message(assistant_msg)
        sm.save_message(tool_result)
        sm.close()

        data = sm.load_session()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["tool_calls"][0]["function"]["name"] == "read_file"
        assert data["messages"][1]["role"] == "tool"

    def test_save_compaction(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test-model")
        sm.save_compaction("Summary of earlier conversation", tokens_before=50000)
        sm.close()

        data = sm.load_session()
        assert len(data["compactions"]) == 1
        assert data["compactions"][0]["summary"] == "Summary of earlier conversation"
        assert data["compactions"][0]["tokens_before"] == 50000

    def test_save_model_change(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="claude-sonnet-4")
        sm.save_model_change("claude-sonnet-4", "claude-opus-4")
        sm.close()

        data = sm.load_session()
        assert len(data["model_changes"]) == 1
        assert data["model_changes"][0]["old_model"] == "claude-sonnet-4"
        assert data["model_changes"][0]["new_model"] == "claude-opus-4"

    def test_malformed_lines_skipped(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test-model")
        sm.save_message({"role": "user", "content": "Hello"})
        sm.close()

        # Append a malformed line
        with open(sm.session_path, "a") as f:
            f.write("not valid json\n")

        data = sm.load_session()
        assert len(data["messages"]) == 1


class TestListSessions:
    def test_list_empty(self, koi_dir):
        sm = SessionManager(koi_dir)
        assert sm.list_sessions() == []

    def test_list_multiple(self, koi_dir):
        # Create 3 sessions
        for i in range(3):
            sm = SessionManager(koi_dir)
            sm.start_session(model=f"model-{i}")
            sm.save_message({"role": "user", "content": f"msg {i}"})
            sm.close()

        sm2 = SessionManager(koi_dir)
        sessions = sm2.list_sessions()
        assert len(sessions) == 3
        # Most recent first
        assert sessions[0]["model"] == "model-2"
        assert sessions[0]["message_count"] == 1

    def test_list_respects_limit(self, koi_dir):
        for i in range(5):
            sm = SessionManager(koi_dir)
            sm.start_session(model=f"model-{i}")
            sm.close()

        sm2 = SessionManager(koi_dir)
        sessions = sm2.list_sessions(limit=2)
        assert len(sessions) == 2


class TestGetLatestSession:
    def test_none_when_empty(self, koi_dir):
        sm = SessionManager(koi_dir)
        assert sm.get_latest_session() is None

    def test_returns_latest(self, koi_dir):
        sm1 = SessionManager(koi_dir)
        sm1.start_session(model="model-1")
        sm1.close()
        path1 = sm1.session_path

        sm2 = SessionManager(koi_dir)
        sm2.start_session(model="model-2")
        sm2.close()
        path2 = sm2.session_path

        sm3 = SessionManager(koi_dir)
        latest = sm3.get_latest_session()
        assert latest == path2


class TestResumeSession:
    def test_resume_and_append(self, koi_dir):
        # Create a session with one message
        sm = SessionManager(koi_dir)
        sm.start_session(model="test-model")
        sm.save_message({"role": "user", "content": "Hello"})
        sm.close()

        # Resume and add more
        sm2 = SessionManager(koi_dir, session_path=sm.session_path)
        sm2.resume_session()
        sm2.save_message({"role": "assistant", "content": "Hi!"})
        sm2.close()

        # Load and verify
        data = sm2.load_session()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["content"] == "Hi!"

    def test_resume_nonexistent_raises(self, koi_dir):
        sm = SessionManager(koi_dir, session_path=koi_dir / "sessions" / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            sm.resume_session()


class TestEntryIds:
    def test_entries_have_id_and_parent_id(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test")
        sm.save_message({"role": "user", "content": "Hello"})
        sm.save_message({"role": "assistant", "content": "Hi"})
        sm.close()

        # Read raw JSONL
        with open(sm.session_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        # Header has no id/parentId added by _write_entry
        assert lines[0]["type"] == "session"

        # Message entries have id and parentId
        msg1 = lines[1]
        msg2 = lines[2]
        assert "id" in msg1
        assert msg1["parentId"] is None  # first entry has no parent
        assert "id" in msg2
        assert msg2["parentId"] == msg1["id"]  # chains to previous


class TestForkSession:
    def test_fork_creates_branch(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test")
        sm.save_message({"role": "user", "content": "msg1"})
        sm.save_message({"role": "assistant", "content": "reply1"})

        # Fork from current point
        fork_point = sm.leaf_id
        sm.save_message({"role": "user", "content": "msg2-branch-a"})

        # Now fork back to the fork point and write different branch
        sm.fork(fork_from_id=fork_point)
        sm.save_message({"role": "user", "content": "msg2-branch-b"})
        sm.close()

        # Read raw to find the branch
        with open(sm.session_path) as f:
            entries = [json.loads(l) for l in f if l.strip()]

        non_header = [e for e in entries if e.get("type") != "session"]
        branch_a_entry = [e for e in non_header if e.get("type") == "message" and e["message"]["content"] == "msg2-branch-a"][0]
        branch_b_entry = [e for e in non_header if e.get("type") == "message" and e["message"]["content"] == "msg2-branch-b"][0]

        assert branch_a_entry["parentId"] == fork_point
        assert branch_b_entry["parentId"] == fork_point

        # Load with branch B leaf (default -- last entry)
        data = sm.load_session()
        assert data["messages"][-1]["content"] == "msg2-branch-b"

        # Load with branch A leaf
        data_a = sm.load_session(leaf_id=branch_a_entry["id"])
        assert data_a["messages"][-1]["content"] == "msg2-branch-a"

    def test_load_v1_session_without_ids(self, koi_dir):
        """V1 sessions without id/parentId load correctly."""
        sm = SessionManager(koi_dir)
        # Manually create a V1 session file (no id/parentId)
        session_path = koi_dir / "sessions" / "test_v1.jsonl"
        with open(session_path, "w") as f:
            f.write(json.dumps({"type": "session", "version": 1, "id": "test", "timestamp": "2026-01-01", "cwd": "/tmp", "model": "test"}) + "\n")
            f.write(json.dumps({"type": "message", "timestamp": "2026-01-01", "message": {"role": "user", "content": "old msg"}}) + "\n")

        sm2 = SessionManager(koi_dir, session_path=session_path)
        data = sm2.load_session()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "old msg"

    def test_get_branches(self, koi_dir):
        sm = SessionManager(koi_dir)
        sm.start_session(model="test")
        sm.save_message({"role": "user", "content": "root"})
        fork_point = sm.leaf_id
        sm.save_message({"role": "user", "content": "branch-a"})
        sm.fork(fork_from_id=fork_point)
        sm.save_message({"role": "user", "content": "branch-b"})
        sm.close()

        branches = sm.get_branches()
        assert len(branches) == 1
        assert branches[0]["id"] == fork_point
