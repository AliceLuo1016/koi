---
name: lepton-cluster-usage
description: Check Lepton cluster node availability by group. Use when the user asks about cluster usage, node availability, or which node groups have free capacity.
---

# Lepton Cluster Usage

List Lepton node groups and summarize availability.

## Workflow

1. Run: `uv run lep node list`
2. Parse the table for each node group: Name, Available Nodes (all GPU available), Ready Nodes
3. Report availability in **only** this format (no extra summary, analysis, or recommendations):

```
**Availability by group**
- **<group-name>:** <available>/<ready> available
```

- Append "(fully free)" when available == ready
- Append "(fully utilized)" when available == 0
- No annotation otherwise
- Do NOT add any summary, recommendations, or additional commentary after the list
