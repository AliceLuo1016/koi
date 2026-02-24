# Best Practices

## Getting Started

### 1. Get API Key

Go to [NVIDIA Inference Key Management](https://inference.nvidia.com/key-management) and generate a new inference key. Save it in a secure location.

### 2. Installation

```bash
cd ~/koi
pip install -e .
```

### 3. Initialize a Project

Follow the init instructions to provide your API key.

```bash
cd imaginaire4
koi init
```

---

## Example Usage

### Cluster Usage

![Cluster usage demo](docs/images/cluster-usage.png)

### Check Jobs

![Job list demo](docs/images/job-list.png)

### Check Failed Jobs

![Failed job analysis demo](docs/images/failed-job.png)

### Check Curation Stats

![Curation stats demo](docs/images/curation-stats.png)

### Set Up Cron Jobs

Set up cron jobs with complicated logic. For example, run black border when there are more than 2M clips to process and there is no existing black border job running — check every 30 minutes.

![Cron job demo](docs/images/cron-job.png)

Cron job example execution log:
```text
============================================================
[2026-02-24 12:00:01] Cron task started: Check if black border removal jobs need to be launched:
1. Run the curation-stats skill to get pipeline stats. Check the "clips_to_remove_black_border" count.
2. If clips_to_remove_black_border > 2,000,000, proceed. Otherwise, do nothing.
3. Run the lepton-job-list skill to check current jobs. Look for any jobs with name prefix "black-border-removal" that are in Running or Queueing state.
4. If there are NO black-border-removal jobs running or queueing, launch new ones by running: uv run bash +x pipelines/video/utils/launch_black_border_removal.sh
5. If black-border-removal jobs already exist, do nothing.
Log what happened and why.
============================================================
## Summary

**Black border removal jobs launched successfully.** Here's the decision log:

| Step | Check | Result |
|------|-------|--------|
| 1. Curation stats | `clips_to_remove_black_border` | **2,043,717** (> 2M threshold ✅) |
| 2. Threshold check | Proceed? | **YES** |
| 3. Existing jobs | Running/Queueing black-border-removal jobs? | **NONE** (0 jobs) |
| 4. Launch | Run `launch_black_border_removal.sh` | **4 jobs created ✅** |

**Jobs launched:**

| Job Name | Job ID | Cluster | Shape | Workers |
|----------|--------|---------|-------|---------|
| `black-border-removal-0` | `black-border-removal-0-xfx9` | oci-ord-lepton-001 | my.4xl40s | 8 |
| `black-border-removal-1` | `black-border-removal-1-wv99` | oci-ord-lepton-001 | my.4xl40s | 8 |
| `black-border-removal-2` | `black-border-removal-2-6cxz` | oci-ord-lepton-001 | my.4xl40s | 8 |
| `black-border-removal-3` | `black-border-removal-3-wd49` | oci-ord-lepton-001 | my.4xl40s | 8 |

Total: **32 worker nodes** queued across 4 jobs, each processing up to 1M clips with priority `high-8000`.
```

---

## How to Add a New Skill

1. Provide a command that you would like to ask koi to run, and make sure it can run successfully.
2. Tell koi your expected output, and ask koi to use `skill-creator` to create a new skill.

> **Rule of thumb:** Always use the `skill-creator` to create a new skill instead of manually creating one.
