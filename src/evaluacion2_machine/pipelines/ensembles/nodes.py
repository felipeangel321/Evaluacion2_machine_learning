import logging

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)

# Técnicas de ensamble del proyecto:
#
# 1. Bagging  → Random Forest: 200 árboles entrenados sobre muestras bootstrap
#               distintas; promediar árboles independientes reduce la VARIANZA.
# 2. Boosting → HistGradientBoosting: árboles secuenciales, cada uno corrige
#               los errores del anterior; reduce el SESGO.
# 3. Voting   → (este módulo) promedia las predicciones de los 3 modelos
#               optimizados. En clasificación se usa voto "soft": se promedian
#               las probabilidades, no las etiquetas, lo que conserva la
#               confianza de cada modelo.
# 4. Stacking → (este módulo) un meta-modelo aprende CUÁNTO confiar en cada
#               modelo base. Se entrena con predicciones out-of-fold (validación
#               cruzada interna), para que no aprenda de predicciones que los
#               modelos base hicieron sobre datos que ya vieron.
#
# Se combinan modelos de familias distintas (lineal, bagging, boosting) porque
# un ensamble solo mejora si sus miembros cometen errores DIFERENTES.


def _base_estimators(log_or_ridge, rf, gb) -> list:
    # clone(): copia los hiperparámetros optimizados sin los pesos entrenados;
    # el ensamble reentrena cada modelo base desde cero.
    return [("lineal", clone(log_or_ridge)), ("random_forest", clone(rf)), ("gradient_boosting", clone(gb))]


def build_classification_ensembles(X_train, y_train_clf, params, log_reg, rf, gb) -> tuple:
    y = y_train_clf.squeeze()
    rs, ens = params["random_state"], params["ensembles"]

    voting = VotingClassifier(_base_estimators(log_reg, rf, gb), voting=ens["voting"])
    voting.fit(X_train, y)

    # Meta-modelo: regresión logística sobre las 3 probabilidades de hit.
    # class_weight='balanced' mantiene el mismo criterio que los modelos base
    # (umbral 0.5 orientado a Recall ante el desbalance).
    stacking = StackingClassifier(
        _base_estimators(log_reg, rf, gb),
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        cv=StratifiedKFold(n_splits=ens["stacking_cv"], shuffle=True, random_state=rs),
        stack_method="predict_proba",
    )
    stacking.fit(X_train, y)

    logger.info("Ensambles de clasificación entrenados: Voting (%s) y Stacking", ens["voting"])
    return voting, stacking


def build_regression_ensembles(X_train, y_train_reg, params, ridge, rf, gb) -> tuple:
    y = y_train_reg.squeeze()
    rs, ens = params["random_state"], params["ensembles"]

    voting = VotingRegressor(_base_estimators(ridge, rf, gb))
    voting.fit(X_train, y)

    # Meta-modelo: RidgeCV elige su propio alpha; con solo 3 entradas muy
    # correlacionadas, la regularización evita pesos inestables.
    stacking = StackingRegressor(
        _base_estimators(ridge, rf, gb),
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
        cv=KFold(n_splits=ens["stacking_cv"], shuffle=True, random_state=rs),
    )
    stacking.fit(X_train, y)

    logger.info("Ensambles de regresión entrenados: Voting y Stacking")
    return voting, stacking


def stacking_weights(stacking_clf, stacking_reg) -> pd.DataFrame:
    # Pesos que el meta-modelo asignó a cada modelo base: muestran cuánto
    # aporta cada familia (lineal, bagging, boosting) a la predicción final.
    # En clasificación son coeficientes sobre la probabilidad de hit de cada
    # modelo; en regresión, sobre su popularidad predicha.
    rows = []
    for task, model, coefs in [("clasificacion", stacking_clf, stacking_clf.final_estimator_.coef_[0]),
                               ("regresion", stacking_reg, stacking_reg.final_estimator_.coef_)]:
        for (name, _), coef in zip(model.estimators, coefs):
            rows.append({"Tarea": task, "Modelo_base": name, "Peso_meta_modelo": coef})
    weights = pd.DataFrame(rows).round(4)
    logger.info("Pesos del meta-modelo:\n%s", weights.to_string(index=False))
    return weights
