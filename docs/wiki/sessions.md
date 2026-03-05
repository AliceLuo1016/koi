# Session Management

Koi persists conversations as JSONL files with support for in-place branching via parentId chains.

## Storage

Sessions are stored in `.koi/sessions/` as JSONL files:

```
.koi/sessions/
  20260304_154500_abc123def456.jsonl
  20260304_160000_789def012345.jsonl
```

Filename format: `<YYYYMMDD_HHMMSS_microseconds>_<session_id>.jsonl`

## JSONL Entry Types

Each line is a JSON object with a `type` field:

| Type | Description |
|------|-------------|
| `session` | Header (first line) — version, model, cwd, timestamp |
| `message` | A conversation message (user, assistant, or tool) |
| `compaction` | Context compaction summary |
| `model_change` | Model switch event |

### Entry IDs (Version 2)

Every non-header entry has:
- **`id`** — unique 8-char hex identifier
- **`parentId`** — the `id` of the previous entry (or `null` for the first)

This creates a linked chain that supports branching.

## Branching

Instead of copying messages to a new file, Koi forks in-place by resetting the leaf pointer:

```
msg1 → msg2 → msg3 (branch A)
              ↘ msg4 (branch B)
```

Both branches share the same session file. When loading, `load_session()` walks from a leaf entry back to root via `parentId` to reconstruct the active branch.

### Fork

```python
sm.fork(fork_from_id=some_entry_id)  # Reset leaf to branch point
sm.save_message(...)                   # New messages branch from there
```

### Load a Branch

```python
data = sm.load_session(leaf_id=branch_a_leaf)  # Walk branch A
data = sm.load_session()                        # Default: last entry as leaf
```

### Find Branch Points

```python
branches = sm.get_branches()  # Entries with multiple children
```

## Backward Compatibility

Version 1 sessions (without `id`/`parentId`) load correctly — entries are treated as a linear sequence.

## Session Listing

`sessions.py` provides session discovery:
- `list_sessions(limit=20)` — recent sessions sorted by timestamp
- `get_latest_session()` — most recent session file

## Chat Commands

| Command | Description |
|---------|-------------|
| `/new` | Start a fresh session |
| `/fork` | Fork from current point (branch in-place) |

## Related Pages

- [Architecture Overview](architecture.md) — Where SessionManager fits
- [Context Management](context-management.md) — Compaction entries in session files
