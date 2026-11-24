import logging

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score,
)

logger = logging.getLogger(__name__)

# Estrategia de búsqueda por modelo:
#
# - GridSearchCV para LogisticRegression (C) y Ridge (alpha): un solo
#   hiperparámetro con pocos valores, se prueba el espacio completo.
# - RandomizedSearchCV para Random Forest y HistGradientBoosting: varios
#   hiperparámetros que interactúan (81 y 432 combinaciones). Muestrear
#   combinaciones al azar encuentra configuraciones casi tan buenas como el
#   grid completo a una fracción del costo (Bergstra & Bengio, 2012).
#
# Todas las búsquedas usan validación cruzada (StratifiedKFold en
# clasificación para mantener el 26.6% de hits en cada fold) y optimizan la
# métrica principal del negocio: F1 en clasificación y MAE en regresión.
#
# Cada modelo optimizado se compara contra su configuración INICIAL (la de la
# EP2, elegida a mano) con exactamente los mismos folds. Si la búsqueda no la
# supera, se conserva la configuración inicial: la optimización nunca empeora
# el modelo final.


def _search_results(search) -> pd.DataFrame:
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]
    results = pd.DataFrame(search.cv_results_)[cols].sort_values("rank_test_score")
    results["params"] = results["params"].astype(str)
    return results.reset_index(drop=True)


def _tune(name, initial, search, X, y, cv, scoring, lower_is_better) -> tuple:
    """Ajusta `search`, lo compara con `initial` en los mismos folds y devuelve el mejor."""
    initial_score = cross_val_score(clone(initial), X, y, cv=cv, scoring=scoring).mean()
    search.fit(X, y)
    tuned_score = search.best_score_

    improved = tuned_score >= initial_score
    final_model = search.best_estimator_ if improved else clone(initial).fit(X, y)

    # Los scorers de sklearn siempre se maximizan: el MAE llega como negativo
    sign = -1 if lower_is_better else 1
    ini, opt = sign * initial_score, sign * tuned_score
    gain = (ini - opt) if lower_is_better else (opt - ini)

    row = {
        "Modelo": name,
        "Busqueda": f"{type(search).__name__} ({len(search.cv_results_['params'])} combinaciones)",
        "Metrica_CV": scoring.replace("neg_mean_absolute_error", "MAE"),
        "Inicial": round(ini, 4),
        "Optimizado": round(opt, 4),
        "Mejora": round(gain, 4),
        "Mejora_pct": round(100 * gain / abs(ini), 2) if ini else 0.0,
        "Modelo_final": "optimizado" if improved else "inicial (la búsqueda no lo superó)",
        "Mejores_parametros": str(search.best_params_),
    }
    logger.info("Tuning %s: %s inicial=%.4f -> optimizado=%.4f (%s)",
                name, row["Metrica_CV"], ini, opt, row["Modelo_final"])
    return final_model, row


def tune_classifiers(X_train, y_train_clf, params, log_reg_initial, rf_initial) -> tuple:
    tuning, rs = params["tuning"], params["random_state"]
    y = y_train_clf.squeeze()
    cv = StratifiedKFold(n_splits=tuning["cv_folds"], shuffle=True, random_state=rs)
    scoring = tuning["scoring_clf"]

    log_reg, row_lr = _tune(
        "LogisticRegression", log_reg_initial,
        GridSearchCV(clone(log_reg_initial), tuning["logreg_param_grid"], scoring=scoring, cv=cv, n_jobs=-1),
        X_train, y, cv, scoring, lower_is_better=False,
    )

    # n_jobs=1 en la búsqueda: el Random Forest ya usa todos los núcleos
    rf, row_rf = _tune(
        "RandomForest", rf_initial,
        RandomizedSearchCV(clone(rf_initial), tuning["rf_clf_param_distributions"], n_iter=tuning["rf_n_iter"],
                           scoring=scoring, cv=cv, random_state=rs, n_jobs=1),
        X_train, y, cv, scoring, lower_is_better=False,
    )

    # HistGradientBoosting construye árboles secuencialmente, cada uno corrige
    # los errores del anterior. La configuración inicial son los valores por
    # defecto de scikit-learn (con class_weight balanceado).
    hgb_initial = HistGradientBoostingClassifier(class_weight="balanced", random_state=rs)
    hgb_search = RandomizedSearchCV(clone(hgb_initial), tuning["hgb_param_distributions"], n_iter=tuning["n_iter"],
                                    scoring=scoring, cv=cv, random_state=rs, n_jobs=1)
    gb, row_gb = _tune("GradientBoosting", hgb_initial, hgb_search, X_train, y, cv, scoring, lower_is_better=False)

    summary = pd.DataFrame([row_lr, row_rf, row_gb])
    return log_reg, rf, gb, _search_results(hgb_search), summary


def tune_regressors(X_train, y_train_reg, params, ridge_initial, rf_initial) -> tuple:
    # Se optimiza MAE (error medio en puntos de popularidad): es la métrica más
    # interpretable para el negocio y menos sensible a outliers que RMSE.
    tuning, rs = params["tuning"], params["random_state"]
    y = y_train_reg.squeeze()
    cv = KFold(n_splits=tuning["cv_folds"], shuffle=True, random_state=rs)
    scoring = tuning["scoring_reg"]

    ridge, row_ridge = _tune(
        "Ridge", ridge_initial,
        GridSearchCV(clone(ridge_initial), tuning["ridge_param_grid"], scoring=scoring, cv=cv, n_jobs=-1),
        X_train, y, cv, scoring, lower_is_better=True,
    )

    rf, row_rf = _tune(
        "RandomForest", rf_initial,
        RandomizedSearchCV(clone(rf_initial), tuning["rf_reg_param_distributions"], n_iter=tuning["rf_n_iter"],
                           scoring=scoring, cv=cv, random_state=rs, n_jobs=1),
        X_train, y, cv, scoring, lower_is_better=True,
    )

    hgb_initial = HistGradientBoostingRegressor(random_state=rs)
    hgb_search = RandomizedSearchCV(clone(hgb_initial), tuning["hgb_param_distributions"], n_iter=tuning["n_iter"],
                                    scoring=scoring, cv=cv, random_state=rs, n_jobs=1)
    gb, row_gb = _tune("GradientBoosting", hgb_initial, hgb_search, X_train, y, cv, scoring, lower_is_better=True)

    summary = pd.DataFrame([row_ridge, row_rf, row_gb])
    return ridge, rf, gb, _search_results(hgb_search), summary
