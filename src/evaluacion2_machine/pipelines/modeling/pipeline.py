from kedro.pipeline import Pipeline, node

from .nodes import train_classifiers, train_regressors


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(train_classifiers,
             inputs=["X_train", "y_train_clf", "params:modeling"],
             outputs=["baseline_clf", "logistic_regression_clf", "random_forest_clf"],
             name="train_classifiers_node"),

        node(train_regressors,
             inputs=["X_train", "y_train_reg", "params:modeling"],
             outputs=["baseline_reg", "ridge_reg", "random_forest_reg"],
             name="train_regressors_node"),
    ])
