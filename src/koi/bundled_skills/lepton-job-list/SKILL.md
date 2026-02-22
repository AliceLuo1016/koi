# Lepton Job List Skill

List Lepton jobs for the current user.

## Workflow

1. Determine username:
   - `username=$(whoami)`
2. Run:
   - `uv run lep job list -u $username`
   - If you need to **parse the output**, force non-truncated rows first:
     - `COLUMNS=300 RICH_DISABLE=1 uv run lep job list -u $username`
   - When the user asks about a **subset** (e.g., filtering, splitting, dedup), use the name filter with the substring:
     - `COLUMNS=300 RICH_DISABLE=1 uv run lep job list -u $username -n <substring>`
3. Provide a concise **summary**:
   - Total jobs (or total matching the substring if filtered)
   - Counts by state in this order: Running, Queueing, Completed, Failed, Stopped
   - **Job-name breakdown** (grouped by job name prefix) with counts by state in the same order
   - Include the resource utilization summary if present (report **nodes occupied**, summing the Workers column)

## Usage

When asked to check jobs under the user’s name:

```
1. exec_command: username=$(whoami)
2. exec_command: uv run lep job list -u $username
3. Summarize totals and counts by state
```

## Example summary format

**By job prefix**
- splitting-transcoding-general: Running 39, Queueing 33, Completed 110, Failed 68
- filtering-iter/filtering-audio: Running 19, Queueing 28, Completed 61, Failed 17, Stopped 10
- black-border-removal: Stopped 21
- dedup: Completed 2, Failed 4
- cluster-assignment: Completed 2

**Resource utilization (running/restarting/deleting) — nodes occupied**
- my.4xl40s: 312
- gpu.8xa100-80gb: 152
