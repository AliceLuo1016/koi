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

---

## How to Add a New Skill

1. Provide a command that you would like to ask koi to run, and make sure it can run successfully.
2. Tell koi your expected output, and ask koi to use `skill-creator` to create a new skill.

> **Rule of thumb:** Always use the `skill-creator` to create a new skill instead of manually creating one.
