import logging

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold

logger = logging.getLogger(__name__)


def _search_results(search: RandomizedSearchCV) -> pd.DataFrame:
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]
    results = pd.DataFrame(search.cv_results_)[cols].sort_values("rank_test_score")
    results["params"] = results["params"].astype(str)
    return results.reset_index(drop=True)


def tune_gradient_boosting_clf(X_train: pd.DataFrame, y_train_clf: pd.DataFrame, params: dict) -> tuple:
    # HistGradientBoosting construye árboles secuencialmente corrigiendo los
    # errores del anterior; suele superar a RandomForest en datos tabulares y,
    # al agrupar las features en bins, entrena en segundos con 60k filas.
    # Tiene muchos hiperparámetros sensibles (learning_rate, profundidad,
    # regularización), por eso se ajusta con RandomizedSearchCV: explora
    # combinaciones aleatorias del espacio, mucho más barato que un GridSearch
    # completo con resultados comparables.
    # Se optimiza F1 porque es la métrica principal ante el desbalance.
    tuning = params["tuning"]
    rs = params["random_state"]

    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(class_weight="balanced", random_state=rs),
        param_distributions=tuning["hgb_param_distributions"],
        n_iter=tuning["n_iter"],
        scoring=tuning["scoring_clf"],
        cv=StratifiedKFold(n_splits=tuning["cv_folds"], shuffle=True, random_state=rs),
        random_state=rs,
        n_jobs=1,  # HGB ya paraleliza internamente con todos los núcleos
    )
    search.fit(X_train, y_train_clf.squeeze())

    logger.info("Mejor HGB clf (%s=%.4f): %s", tuning["scoring_clf"], search.best_score_, search.best_params_)
    return search.best_estimator_, _search_results(search)


def tune_gradient_boosting_reg(X_train: pd.DataFrame, y_train_reg: pd.DataFrame, params: dict) -> tuple:
    # Mismo algoritmo y espacio de búsqueda que en clasificación. Se optimiza
    # MAE (error medio en puntos de popularidad) por ser la métrica más
    # interpretable para el negocio y menos sensible a outliers que RMSE.
    tuning = params["tuning"]
    rs = params["random_state"]

    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=rs),
        param_distributions=tuning["hgb_param_distributions"],
        n_iter=tuning["n_iter"],
        scoring=tuning["scoring_reg"],
        cv=KFold(n_splits=tuning["cv_folds"], shuffle=True, random_state=rs),
        random_state=rs,
        n_jobs=1,
    )
    search.fit(X_train, y_train_reg.squeeze())

    logger.info("Mejor HGB reg (%s=%.4f): %s", tuning["scoring_reg"], search.best_score_, search.best_params_)
    return search.best_estimator_, _search_results(search)
