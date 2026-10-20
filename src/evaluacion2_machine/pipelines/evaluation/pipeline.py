from kedro.pipeline import Pipeline, node

from .nodes import (
    cross_validate_classifiers,
    cross_validate_regressors,
    evaluate_classifiers,
    evaluate_regressors,
    grouped_permutation_importance,
    logistic_coefficients,
    plot_metrics_comparison,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # ── Clasificación ──
        node(evaluate_classifiers,
             inputs=["X_test", "y_test_clf", "baseline_clf", "logistic_regression_clf",
                     "random_forest_clf", "gradient_boosting_clf"],
             outputs=["clf_test_metrics", "plot_clf_confusion_matrices", "plot_clf_roc_curves"],
             name="evaluate_classifiers_node"),

        node(cross_validate_classifiers,
             inputs=["X_train", "y_train_clf", "params:modeling", "logistic_regression_clf",
                     "random_forest_clf", "gradient_boosting_clf"],
             outputs="cv_clf_results",
             name="cross_validate_classifiers_node"),

        # ── Regresión ──
        node(evaluate_regressors,
             inputs=["X_test", "y_test_reg", "baseline_reg", "ridge_reg",
                     "random_forest_reg", "gradient_boosting_reg"],
             outputs=["reg_test_metrics", "plot_reg_real_vs_pred"],
             name="evaluate_regressors_node"),

        node(cross_validate_regressors,
             inputs=["X_train", "y_train_reg", "params:modeling", "ridge_reg",
                     "random_forest_reg", "gradient_boosting_reg"],
             outputs="cv_reg_results",
             name="cross_validate_regressors_node"),

        node(plot_metrics_comparison,
             inputs=["clf_test_metrics", "reg_test_metrics"],
             outputs="plot_metrics_comparison",
             name="plot_metrics_comparison_node"),

        # ── Interpretación ──
        node(grouped_permutation_importance,
             inputs=["X_test", "y_test_clf", "y_test_reg", "gradient_boosting_clf",
                     "gradient_boosting_reg", "params:modeling"],
             outputs=["feature_importance", "plot_feature_importance"],
             name="grouped_permutation_importance_node"),

        node(logistic_coefficients,
             inputs=["logistic_regression_clf", "X_test"],
             outputs=["logistic_coefficients", "plot_logistic_coefficients"],
             name="logistic_coefficients_node"),
    ])
