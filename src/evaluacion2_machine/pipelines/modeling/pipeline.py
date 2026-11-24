from kedro.pipeline import Pipeline, node

from .nodes import train_classifiers, train_regressors


def create_pipeline(**kwargs) -> Pipeline:
    # Los modelos *_inicial usan la configuración elegida a mano (parameters.yml)
    # y son el punto de partida de hyperparameter_tuning, que guarda las
    # versiones optimizadas con los nombres finales (logistic_regression_clf, ...).
    return Pipeline([
        node(train_classifiers,
             inputs=["X_train", "y_train_clf", "params:modeling"],
             outputs=["baseline_clf", "logistic_regression_clf_inicial", "random_forest_clf_inicial"],
             name="train_classifiers_node"),

        node(train_regressors,
             inputs=["X_train", "y_train_reg", "params:modeling"],
             outputs=["baseline_reg", "ridge_reg_inicial", "random_forest_reg_inicial"],
             name="train_regressors_node"),
    ])
