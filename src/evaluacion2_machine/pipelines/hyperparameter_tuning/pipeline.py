from kedro.pipeline import Pipeline, node

from .nodes import tune_gradient_boosting_clf, tune_gradient_boosting_reg


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(tune_gradient_boosting_clf,
             inputs=["X_train", "y_train_clf", "params:modeling"],
             outputs=["gradient_boosting_clf", "hyperparameter_results_clf"],
             name="tune_gradient_boosting_clf_node"),

        node(tune_gradient_boosting_reg,
             inputs=["X_train", "y_train_reg", "params:modeling"],
             outputs=["gradient_boosting_reg", "hyperparameter_results_reg"],
             name="tune_gradient_boosting_reg_node"),
    ])
