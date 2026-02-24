---
name: lepton-job-list
description: List and summarize Lepton jobs for the current user. Use when the user asks to check their jobs, list running/failed/queued jobs, filter jobs by name or node group, or get resource utilization summaries.
---

# Lepton Job List

List Lepton jobs for the current user and summarize by state and resource utilization.

## Workflow

1. Determine the username: `username=$(whoami)`
2. List jobs using one of these variants:
   - Default (human-readable): `uv run lep job list -u $username`
   - Non-truncated (for parsing): `COLUMNS=300 RICH_DISABLE=1 uv run lep job list -u $username`
   - Filter by name substring: `COLUMNS=300 RICH_DISABLE=1 uv run lep job list -u $username -n <substring>`
   - Filter by node group: `uv run lep job list -u $username -ng <node_group_name>`
3. Summarize the results (see format below)

### Node Group Nicknames

| Nickname | Full Name |
|----------|-----------|
| azure | `az-sat-lepton-002` |
| oci | `oci-ord-lepton-001` |
| neb-hel | `neb-hel-lepton-001` |
| neb-cdg | `neb-cdg-lepton-001` |

## Summary Format

Report in this order:
- Total jobs (or total matching filter)
- Counts by state: Running, Queueing, Completed, Failed, Stopped
- Job-name breakdown grouped by job name prefix, with counts per state
- Resource utilization: nodes occupied per resource type (sum the Workers column)

**Example:**

**By job prefix**
- splitting-transcoding-general: Running 39, Queueing 33, Completed 110, Failed 68
- filtering-iter/filtering-audio: Running 19, Queueing 28, Completed 61, Failed 17, Stopped 10
- black-border-removal: Stopped 21
- dedup: Completed 2, Failed 4

**Resource utilization (running/restarting/deleting) — nodes occupied**
- my.4xl40s: 312
- gpu.8xa100-80gb: 152
