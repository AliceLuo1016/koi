---
name: curation-stats
description: Query video curation pipeline statistics from Databricks. Use when the user asks about curation stats, pipeline progress, how many videos/clips remain, splitting counts, filtering counts, black border removal counts, clustering counts, or metadata totals.
---

# Curation Stats

Report video curation pipeline statistics by querying Databricks.

## Workflow

1. Run: `uv run python .koi/skills/curation-stats/scripts/curation_stats.py`
2. Parse the output lines (format: `key: count`)
3. Report in this format:

```
**Curation Pipeline Stats**
- **Videos to split:** <count>
- **Clips to filter:** <count>
- **Clips to remove black border:** <count>
- **Clips to run cluster assignment:** <count>
- **Clips in metadata table:** <count>
```

## Databricks Tables Reference

| Metric | Table |
|--------|-------|
| Videos to split | `videos_v2_prod.mv_splitting_and_transcoding_input_v2p4_gcs` |
| Clips to filter | `videos_v2_prod.mv_filtering_input_v2p1` |
| Clips to remove black border | `videos_v2_prod.mv_black_border_removal_input_v2` |
| Clips to run cluster assignment | `videos_v2_prod.mv_cluster_assignment_input_v2p1` |
| Clips in metadata | `videos_v2_prod.mv_video_metadata_v0` |
