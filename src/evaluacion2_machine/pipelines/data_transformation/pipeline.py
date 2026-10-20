from kedro.pipeline import Pipeline, node

from .nodes import (
    create_targets,
    drop_identifier_columns,
    encode_categorical,
    extract_features,
    prepare_clustering_input,
    scale_numerical,
    split_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(extract_features,
             inputs="spotify_clean",
             outputs="spotify_features",
             name="extract_features_node"),

        # ── Input no supervisado (features de audio escaladas) ──
        node(prepare_clustering_input,
             inputs=["spotify_features",
                     "params:clustering_features",
                     "params:scaler_type"],
             outputs="spotify_clustering_input",
             name="prepare_clustering_input_node"),

        # ── Input supervisado ──
        node(drop_identifier_columns,
             inputs=["spotify_features", "params:id_columns"],
             outputs="spotify_features_no_ids",
             name="drop_identifier_columns_node"),

        node(encode_categorical,
             inputs=["spotify_features_no_ids", "params:categorical_cols_ohe"],
             outputs="spotify_encoded",
             name="encode_categorical_node"),

        node(create_targets,
             inputs=["spotify_encoded", "params:hit_threshold"],
             outputs="spotify_model_input",
             name="create_targets_node"),

        node(split_data,
             inputs=["spotify_model_input", "params:modeling"],
             outputs=["X_train_unscaled", "X_test_unscaled",
                      "y_train_clf", "y_test_clf",
                      "y_train_reg", "y_test_reg"],
             name="split_data_node"),

        node(scale_numerical,
             inputs=["X_train_unscaled", "X_test_unscaled",
                     "params:numerical_cols_scale",
                     "params:scaler_type"],
             outputs=["X_train", "X_test", "scaler"],
             name="scale_numerical_node"),
    ])
