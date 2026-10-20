from kedro.pipeline import Pipeline, node

from .nodes import (
    deduplicate_tracks,
    drop_columns,
    drop_missing_values,
    filter_zero_popularity,
    remove_invalid_values,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(drop_columns,
             inputs=["spotify_raw", "params:columns_to_drop"],
             outputs="spotify_dropped",
             name="drop_columns_node"),

        node(drop_missing_values,
             inputs="spotify_dropped",
             outputs="spotify_no_nulls",
             name="drop_missing_values_node"),

        node(remove_invalid_values,
             inputs=["spotify_no_nulls", "params:cleaning.invalid_zero_cols"],
             outputs="spotify_valid",
             name="remove_invalid_values_node"),

        node(deduplicate_tracks,
             inputs=["spotify_valid", "params:cleaning.dedupe_by_name_artist"],
             outputs="spotify_dedup",
             name="deduplicate_tracks_node"),

        node(filter_zero_popularity,
             inputs=["spotify_dedup", "params:cleaning.drop_zero_popularity"],
             outputs="spotify_clean",
             name="filter_zero_popularity_node"),
    ])
