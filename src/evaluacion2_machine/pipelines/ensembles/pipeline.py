from kedro.pipeline import Pipeline, node

from .nodes import build_classification_ensembles, build_regression_ensembles, stacking_weights


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(build_classification_ensembles,
             inputs=["X_train", "y_train_clf", "params:modeling",
                     "logistic_regression_clf", "random_forest_clf", "gradient_boosting_clf"],
             outputs=["voting_clf", "stacking_clf"],
             name="build_classification_ensembles_node"),

        node(build_regression_ensembles,
             inputs=["X_train", "y_train_reg", "params:modeling",
                     "ridge_reg", "random_forest_reg", "gradient_boosting_reg"],
             outputs=["voting_reg", "stacking_reg"],
             name="build_regression_ensembles_node"),

        # Lee los Stacking ya guardados: se puede regenerar sin reentrenar
        node(stacking_weights,
             inputs=["stacking_clf", "stacking_reg"],
             outputs="stacking_weights",
             name="stacking_weights_node"),
    ])
