from kedro.pipeline import Pipeline, node

from .nodes import (
    cross_validate_classifiers,
    cross_validate_regressors,
    evaluate_classifiers,
    evaluate_regressors,
    generalization_classifiers,
    generalization_regressors,
    grouped_permutation_importance,
    learning_curves,
    logistic_coefficients,
    plot_generalization,
    plot_metrics_comparison,
)

CLF_MODELS = ["logistic_regression_clf", "random_forest_clf", "gradient_boosting_clf", "voting_clf", "stacking_clf"]
REG_MODELS = ["ridge_reg", "random_forest_reg", "gradient_boosting_reg", "voting_reg", "stacking_reg"]


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # ── Clasificación ──
        node(evaluate_classifiers,
             inputs=["X_test", "y_test_clf", "baseline_clf", *CLF_MODELS],
             outputs=["clf_test_metrics", "plot_clf_confusion_matrices", "plot_clf_roc_curves"],
             name="evaluate_classifiers_node"),

        node(cross_validate_classifiers,
             inputs=["X_train", "y_train_clf", "params:modeling", *CLF_MODELS],
             outputs="cv_clf_results",
             name="cross_validate_classifiers_node"),

        # ── Regresión ──
        node(evaluate_regressors,
             inputs=["X_test", "y_test_reg", "baseline_reg", *REG_MODELS],
             outputs=["reg_test_metrics", "plot_reg_real_vs_pred"],
             name="evaluate_regressors_node"),

        node(cross_validate_regressors,
             inputs=["X_train", "y_train_reg", "params:modeling", *REG_MODELS],
             outputs="cv_reg_results",
             name="cross_validate_regressors_node"),

        node(plot_metrics_comparison,
             inputs=["clf_test_metrics", "reg_test_metrics"],
             outputs="plot_metrics_comparison",
             name="plot_metrics_comparison_node"),

        # ── Generalización ──
        node(generalization_classifiers,
             inputs=["X_train", "y_train_clf", "clf_test_metrics", "cv_clf_results", *CLF_MODELS],
             outputs="generalization_clf",
             name="generalization_classifiers_node"),

        node(generalization_regressors,
             inputs=["X_train", "y_train_reg", "reg_test_metrics", "cv_reg_results", *REG_MODELS],
             outputs="generalization_reg",
             name="generalization_regressors_node"),

        node(plot_generalization,
             inputs=["generalization_clf", "generalization_reg"],
             outputs="plot_generalization",
             name="plot_generalization_node"),

        node(learning_curves,
             inputs=["X_train", "y_train_clf", "y_train_reg", "gradient_boosting_clf",
                     "gradient_boosting_reg", "params:modeling"],
             outputs=["learning_curve_results", "plot_learning_curves"],
             name="learning_curves_node"),

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
