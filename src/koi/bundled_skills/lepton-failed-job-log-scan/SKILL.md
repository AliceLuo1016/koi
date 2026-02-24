---
name: lepton-failed-job-log-scan
description: Download and analyze logs from failed Lepton jobs to identify common failure patterns. Use when the user asks to check failed jobs, scan job logs for errors, or diagnose why jobs are failing.
---

# Lepton Failed Job Log Scan

Download logs from failed Lepton jobs and summarize common failure patterns.

## Workflow

1. Determine the username: `username=$(whoami)`
2. List jobs with non-truncated output: `COLUMNS=300 RICH_DISABLE=1 uv run lep job list -u $username`
3. Parse for jobs with **State = Failed** and extract the **Job ID** (second line of each Name/ID cell)
4. Create a log directory for today's date (or a user-specified date):
   ```bash
   log_dir=.koi/lepton-logs/<YYYY-MM-DD>
   mkdir -p $log_dir
   ```
5. Download each failed job's log: `uv run lep log get -j <job_id> --path $log_dir/<job_id>.log`
6. Scan all downloaded logs with python3:
   - Count ERROR/WARN lines per file
   - Extract the first ERROR block per file (ERROR line + ~20 lines of context)
   - Identify shared root causes (e.g., missing NVENC library, missing files, download errors)
7. Summarize:
   - Total failed jobs
   - List of downloaded log files
   - Common failure causes with 1-2 representative snippets
   - Note if no common pattern is found

## Notes

- Default log date is **today** unless the user specifies a date
- Logs are stored in `.koi/lepton-logs/<date>/`
- For older jobs whose logs have expired, use the `--start` flag with `lep log get` to fetch logs from the correct date range
- When reporting GPU/hardware errors, always extract the node IP from log lines — the IP is embedded as `ip=X.X.X.X` in Ray worker log prefixes
