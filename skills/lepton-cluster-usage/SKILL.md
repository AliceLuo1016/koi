# Lepton Cluster Usage Skill

Get Lepton cluster usage by listing node groups and summarizing **availability only** in a simple list.

## Workflow

1. Run: `uv run lep node list`
2. Parse the table output for each node group:
   - Name
   - Available Nodes (all GPU available)
   - Ready Nodes
3. Report **only** availability by group in this format:

```
**Availability by group**
- **<group-name>:** <available>/<ready> available (fully free|fully utilized)
```

Notes:
- Include “(fully free)” when available == ready
- Include “(fully utilized)” when available == 0
- Otherwise just show “<available>/<ready> available” with no extra notes

## Usage

When asked to check Lepton cluster usage:

```
1. exec_command: uv run lep node list
2. Summarize using only the Availability by group list format above
```
