#!/usr/bin/env python3
"""Query Databricks for video curation pipeline stats."""

from cosmos_internal_data_utils.databases import types as db_types

QUERIES = {
    "videos_to_split": "SELECT count(*) FROM videos_v2_prod.mv_splitting_and_transcoding_input_v2p4_gcs",
    "clips_to_filter": "SELECT count(*) FROM videos_v2_prod.mv_filtering_input_v2p1",
    "clips_to_remove_black_border": "SELECT count(*) FROM videos_v2_prod.mv_black_border_removal_input_v2",
    "clips_to_cluster": "SELECT count(*) FROM videos_v2_prod.mv_cluster_assignment_input_v2p1",
    "clips_in_metadata": "SELECT count(*) FROM videos_v2_prod.mv_video_metadata_v0",
}

def main():
    db = db_types.DatabricksDB.make_from_dir_config(db_types.MediaType.VIDEO)
    for name, query in QUERIES.items():
        result = db.make_query(query)
        # Extract the count value from the result
        if hasattr(result, 'iloc'):
            count = result.iloc[0, 0]
        elif isinstance(result, list):
            count = result[0][0] if result else "N/A"
        else:
            count = result
        print(f"{name}: {count}")

if __name__ == "__main__":
    main()
