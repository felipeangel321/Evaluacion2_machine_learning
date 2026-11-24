from kedro.pipeline import Pipeline, node

from .nodes import tune_classifiers, tune_regressors


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(tune_classifiers,
             inputs=["X_train", "y_train_clf", "params:modeling",
                     "logistic_regression_clf_inicial", "random_forest_clf_inicial"],
             outputs=["logistic_regression_clf", "random_forest_clf", "gradient_boosting_clf",
                      "hyperparameter_results_clf", "tuning_summary_clf"],
             name="tune_classifiers_node"),

        node(tune_regressors,
             inputs=["X_train", "y_train_reg", "params:modeling",
                     "ridge_reg_inicial", "random_forest_reg_inicial"],
             outputs=["ridge_reg", "random_forest_reg", "gradient_boosting_reg",
                      "hyperparameter_results_reg", "tuning_summary_reg"],
             name="tune_regressors_node"),
    ])
