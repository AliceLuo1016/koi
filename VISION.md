# Koi Vision 🐠

**Koi is the one-stop work assistant for your entire team.**

Not another chatbot. Not another coding agent. Koi is an AI teammate that has access to everything you have access to — your clusters, your storage, your pipelines, your codebase, your communication tools — and can act on all of it through natural language.

## The Problem

Today's AI tools live in silos. Your coding agent can't check your cluster. Your chatbot can't query your data lake. Your monitoring dashboard can't kick off a fix. Teams constantly context-switch between a dozen tools, memorizing commands, bucket names, and URLs that only two people on the team actually remember.

## The Vision

Koi breaks down those walls. It connects to the tools your team already uses and provides a single natural-language interface across all of them:

- **Compute** — Slurm, Lepton
- **Storage** — S3, GCS, local filesystems
- **Code** — GitLab
- **Communication** — Slack, Teams, Outlook
- **Data & ML** — Weights & Biases, LanceDB
- **Docs** — Google Docs
- **Infrastructure** — Docker, Kubernetes, CI/CD

The agent doesn't live in a silo. It sees what you see, and it uses that full picture to make better decisions.

## Use Cases

### 1. Natural Language Access

No more memorizing commands, bucket names, or cluster hostnames.

> "Download the latest checkpoint from the training run we started Tuesday"

Koi knows which S3 bucket, which prefix, which run — because it has access to the same context you do.

### 2. System Monitoring

Ask questions about the state of your world in plain English.

> "How is the curation pipeline progressing?"
> "What's our cluster utilization right now?"
> "How many clips have been processed? How many datasets do we have?"

Koi queries the relevant systems (Slurm, W&B, S3, lancedb), synthesizes the information, and gives you a clear answer — no dashboard hopping.

### 3. Job Orchestration

Describe the logic. Koi handles the execution.

> "We have 200 videos to process. Use up to 8 nodes, but leave at least 4 free for training. Prioritize the splitting first."

Koi translates intent into Slurm commands, monitors progress, and adjusts as resources free up. Every instruction is just a sentence.

### 4. Autonomous Judgment

When something breaks, Koi doesn't just alert — it triages.

> **Job failed on node 3.**
> Koi analyzes the logs, determines it was just something being flaky, and resubmits. No human needed.
>
> **Job failed: CUDA driver mismatch.**
> Koi recognizes this requires a node-level fix, drafts a message to the infra team, and pings the right person on Slack.

The key: **know when to act, and know when to escalate.**

### 5. Coding Orchestrator

Koi assesses the complexity of a task and picks the right approach:

- **Simple fix** → patches it directly (edit a config, fix a typo, adjust a threshold)
- **Complex task** → spins up a Codex or Claude Code agent, provides context, monitors progress, and reports back

One interface. Right tool for the job.

### 6. Deployable & Scalable

Koi isn't just a local dev tool. It runs anywhere your team needs it:

- **Local** — on a developer's laptop for personal use
- **Docker** — containerized for team servers
- **Kubernetes** — scaled across a cluster, one instance per team or per project
- **CI/CD** — embedded in pipelines for automated analysis and action

## What Needs to Be Built

### Core Infrastructure

- [ ] **Webhook server** — HTTP endpoint for receiving events (GitLab pushes, Slurm job completions, CI/CD triggers, Slack interactions). Enables Koi to react to the world, not just respond to prompts.
- [ ] **Session management** — Persistent, multi-turn sessions with identity. Support concurrent users, team-shared context, and session handoff. Each team member talks to the same Koi but with their own thread.
- [ ] **Event bus** — Internal pub/sub system for routing incoming events (webhooks, cron, alerts) to the right handler or session. Decouples event sources from processing logic.
- [ ] **Authentication & identity** — Multi-user auth (API keys, SSO/OIDC). Per-user permissions, credential vaults, and audit trails. Koi needs to act *as* a user when accessing external systems.
- [ ] **Credential management** — Secure storage for API keys, tokens, and service accounts (S3, Slurm, GitLab, Slack, etc.). Per-user and per-team scopes. Rotate and revoke without downtime.

### Integrations (Skills)

- [ ] **Slurm** — Submit, monitor, cancel jobs. Query queue, node status, utilization. Parse sacct/squeue output.
- [ ] **S3 / object storage** — List, upload, download, sync. Handle buckets, prefixes, presigned URLs. Multi-account support.
- [ ] **GitLab / GitHub** — Browse repos, read files, create MRs/PRs, review diffs, trigger pipelines, manage issues.
- [ ] **Slack** — Read/send messages, respond to mentions, post to channels, interactive commands.
- [ ] **Teams / Outlook** — Read/send messages and emails, calendar integration, meeting summaries.
- [ ] **Weights & Biases** — Query runs, compare metrics, pull artifacts, monitor training progress.
- [ ] **LanceDB / vector stores** — Query datasets, check stats, run similarity searches.
- [ ] **Docker** — Build, run, manage containers. Read logs, exec into containers.
- [ ] **Kubernetes** — List pods, check status, scale deployments, read logs, port-forward.
- [ ] **CI/CD pipelines** — Trigger builds, check status, read logs, retry failed stages.
- [ ] **Google Docs / Notion** — Read and create documents, search across team wikis.

### Agent Capabilities

- [ ] **Multi-agent orchestration** — Spawn, monitor, and coordinate sub-agents (coding agents, data analysis agents, monitoring agents). Route tasks by complexity.
- [ ] **Long-running task management** — Track async operations (training runs, data pipelines, CI builds) across sessions. Resume context after hours or days.
- [ ] **Escalation framework** — Configurable rules for when to act autonomously vs. when to ask a human. Confidence thresholds, risk levels, team routing.
- [ ] **Team knowledge base** — Shared memory across team members. "How did we fix the OOM issue last time?" Koi remembers, even if you don't.
- [ ] **Observability & audit** — Full log of every action taken, every decision made. Searchable, exportable, and reviewable. Trust requires transparency.

### Deployment

- [ ] **Dockerfile** — Containerized Koi with all dependencies, ready to deploy.
- [ ] **Helm chart / K8s manifests** — Production deployment with scaling, health checks, and resource limits.
- [ ] **Multi-instance coordination** — Multiple Koi instances sharing state (team context, job tracking) via a backing store (Redis, Postgres, etc.).
- [ ] **API server mode** — Run Koi as a service with REST/gRPC endpoints, not just as a CLI. Enables integrations with web UIs, bots, and other systems.

## Philosophy: Orchestrator, Not Specialist

Koi is not trying to be the best at everything. It's trying to be the best at **knowing who to call**.

**For things we already know how to do — Koi does them directly.** Checking a database, querying Slurm, listing S3 objects, parsing logs, monitoring a pipeline — these are well-understood operations. Koi handles them with its own tools, the same way a senior engineer would: by knowing the right commands, APIs, and sequences.

**For things that need deep expertise — Koi calls an expert.** Need a complex code fix? Spin up a Claude Code or Codex agent. Need a security review? Route to the right tool. Koi doesn't try to out-code a coding agent or out-analyze a dedicated analysis tool. It knows its limits and delegates.

This is the key insight: **Koi is an orchestrator with broad access, not a specialist competing with dedicated tools.** Think of it like a senior engineer who knows the entire system — they can do most operational tasks themselves, but when something needs deep expertise, they know exactly who to pull in and what context to give them.

The value is in the combination:
- Koi checks the database, sees the curation pipeline is stalled
- Queries the cluster, finds a job failed with a code error
- Pulls the relevant logs and code context
- Spins up a coding agent with exactly the right context to fix it
- Monitors the fix, resubmits the job, and reports back

No single specialist agent can do that end-to-end. Koi can, because it sees the full picture.

## Design Principles

1. **Natural language first** — Every interaction should feel like talking to a knowledgeable teammate, not writing shell commands.
2. **Full context, better decisions** — The more Koi can see, the better it can help. Integrations aren't optional — they're the point.
3. **Orchestrate, don't compete** — Do what you know. Delegate what you don't. Never try to out-specialist a specialist.
4. **Know when to escalate** — Autonomy without judgment is dangerous. Koi should act when confident and ask when not.
5. **Team-native** — Built for teams from day one. Shared context, multi-user, collaborative.
6. **Deployable anywhere** — Laptop to cluster. Docker to Kubernetes. One tool that scales with the team.

## Where We Are Today

Koi currently runs as a terminal agent with:
- Multi-provider LLM support (OpenAI, Anthropic)
- Tool calling (file ops, shell, web, memory)
- Persistent memory and context management
- Skills system, sandbox security, cron, sub-agents
- Streaming, prompt caching, extended thinking

What comes next is connecting it to the real world.
