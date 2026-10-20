import logging

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

logger = logging.getLogger(__name__)


def train_classifiers(X_train: pd.DataFrame, y_train_clf: pd.DataFrame, params: dict) -> tuple:
    # Baseline (DummyClassifier): siempre predice la clase mayoritaria
    # ("no hit"). Sirve de piso: con ~73% de negativos logra 73% de Accuracy
    # con F1 = 0, lo que demuestra por qué Accuracy no sirve en este problema.
    #
    # LogisticRegression: modelo lineal e interpretable (sus coeficientes
    # indican la dirección del efecto de cada feature sobre la probabilidad
    # de hit). Requiere features escaladas, lo que ya hace data_transformation.
    #
    # RandomForest: captura las relaciones no lineales que mostró el EDA
    # (la correlación lineal de cada feature de audio con popularity es < 0.1)
    # e interacciones entre género y sonido sin feature engineering adicional.
    #
    # class_weight='balanced' compensa el desbalance (~27% hits) dando más
    # peso a los errores sobre la clase minoritaria.
    y = y_train_clf.squeeze()
    rs = params["random_state"]

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y)

    log_reg = LogisticRegression(**params["clf_logistic_regression"], random_state=rs)
    log_reg.fit(X_train, y)

    rf = RandomForestClassifier(**params["clf_random_forest"], random_state=rs, n_jobs=params["n_jobs"])
    rf.fit(X_train, y)

    logger.info("Clasificadores entrenados: Baseline, LogisticRegression, RandomForest")
    return baseline, log_reg, rf


def train_regressors(X_train: pd.DataFrame, y_train_reg: pd.DataFrame, params: dict) -> tuple:
    # Baseline (DummyRegressor): siempre predice la media de popularity.
    # Cualquier modelo útil debe tener menor MAE que este.
    #
    # Ridge en lugar de LinearRegression simple: el EDA mostró multicolinealidad
    # (energy-loudness r=0.76, energy-acousticness r=-0.73) y el One-Hot sin
    # drop_first hace que las dummies de cada variable sumen 1. La penalización
    # L2 estabiliza los coeficientes sin perder interpretabilidad.
    #
    # RandomForest: mismo criterio que en clasificación (relaciones no lineales).
    # max_features=0.3 decorrelaciona los árboles con 144 columnas.
    y = y_train_reg.squeeze()
    rs = params["random_state"]

    baseline = DummyRegressor(strategy="mean").fit(X_train, y)

    ridge = Ridge(**params["reg_ridge"])
    ridge.fit(X_train, y)

    rf = RandomForestRegressor(**params["reg_random_forest"], random_state=rs, n_jobs=params["n_jobs"])
    rf.fit(X_train, y)

    logger.info("Regresores entrenados: Baseline, Ridge, RandomForest")
    return baseline, ridge, rf
