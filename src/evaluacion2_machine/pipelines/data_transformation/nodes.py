import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # duration_min es más interpretable que duration_ms (mismo contenido).
    # n_artists e is_collab capturan colaboraciones: el campo 'artists' separa
    # a los artistas con ';' y las colaboraciones suelen tener más alcance.
    # explicit se pasa de bool a int porque sklearn no acepta columnas bool.
    df = df.copy()
    df["duration_min"] = df["duration_ms"] / 60_000
    df["n_artists"] = df["artists"].str.count(";") + 1
    df["is_collab"] = (df["n_artists"] > 1).astype(int)
    df["explicit"] = df["explicit"].astype(int)
    df = df.drop(columns=["duration_ms"])
    logger.info("Features extraídas: duration_min, n_artists, is_collab")
    return df


def drop_identifier_columns(df: pd.DataFrame, id_columns: list) -> pd.DataFrame:
    # track_id, artists, album_name y track_name son identificadores de texto
    # libre con decenas de miles de valores únicos: no generalizan y se
    # eliminan después de usarlos para deduplicar y crear features.
    existing = [c for c in id_columns if c in df.columns]
    logger.info(f"Eliminando identificadores: {existing}")
    return df.drop(columns=existing)


def encode_categorical(df: pd.DataFrame, categorical_cols_ohe: list) -> pd.DataFrame:
    # track_genre (114 géneros), key (12 tonalidades) y time_signature son
    # nominales: un orden numérico (key 0 < 11) no tiene sentido musical,
    # por eso se usa One-Hot Encoding en lugar de Label Encoding.
    # drop_first=False para que los modelos de árboles vean todas las categorías.
    # mode y explicit ya son binarias (0/1) y no necesitan encoding.
    existing = [c for c in categorical_cols_ohe if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=False)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    logger.info(f"One-Hot Encoding aplicado a {existing} — columnas totales: {df.shape[1]}")
    return df


def create_targets(df: pd.DataFrame, hit_threshold: int) -> pd.DataFrame:
    # Se definen dos problemas sobre la misma variable de negocio:
    # - Regresión: predecir popularity (1-100) directamente.
    # - Clasificación: is_hit = 1 si popularity >= hit_threshold, que
    #   corresponde aproximadamente al 25% más popular del dataset limpio.
    #   El dataset queda desbalanceado (~25% positivos), lo que justifica usar
    #   F1 / ROC-AUC en vez de Accuracy al evaluar los clasificadores.
    df = df.copy()
    df["is_hit"] = (df["popularity"] >= hit_threshold).astype(int)
    logger.info(
        "Target is_hit (popularity >= %d) — tasa positiva: %.3f",
        hit_threshold, df["is_hit"].mean(),
    )
    return df


def split_data(df: pd.DataFrame, params: dict) -> tuple:
    # Un único split estratificado por is_hit sirve para ambas tareas, así
    # clasificación y regresión se evalúan sobre exactamente las mismas canciones.
    # popularity e is_hit se excluyen de X: son el target (data leakage).
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    X = df.drop(columns=["popularity", "is_hit"])
    y_clf = df["is_hit"]
    y_reg = df["popularity"]

    X_train, X_test, y_train_clf, y_test_clf, y_train_reg, y_test_reg = train_test_split(
        X, y_clf, y_reg,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )

    logger.info(
        "Split: train=%d, test=%d | is_hit train=%.3f test=%.3f",
        len(X_train), len(X_test), y_train_clf.mean(), y_test_clf.mean(),
    )

    return (
        X_train, X_test,
        y_train_clf.to_frame(), y_test_clf.to_frame(),
        y_train_reg.to_frame(), y_test_reg.to_frame(),
    )


def _get_scaler(scaler_type: str):
    scalers = {
        "standard": StandardScaler(),  # media=0, std=1
        "robust":   RobustScaler(),    # mediana+IQR, menos sensible a outliers
        "minmax":   MinMaxScaler(),    # rango [0,1]
    }
    return scalers.get(scaler_type, StandardScaler())


def scale_numerical(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list,
    scaler_type: str = "standard",
) -> tuple:
    # El escalador se ajusta SOLO con train y luego se aplica a test: si se
    # ajustara con todo el dataset, la media y desviación de test se filtrarían
    # al entrenamiento (data leakage). Las columnas One-Hot y binarias no se
    # escalan para mantener su interpretación 0/1.
    # StandardScaler es necesario para KNN, SVM y modelos lineales; los árboles
    # no lo necesitan pero tampoco se ven afectados.
    scaler = _get_scaler(scaler_type)
    existing = [c for c in numerical_cols if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[existing] = scaler.fit_transform(X_train[existing])
    X_test[existing] = scaler.transform(X_test[existing])

    logger.info("Escalado '%s' (fit en train) aplicado a: %s", scaler_type, existing)
    return X_train, X_test, scaler


def prepare_clustering_input(df: pd.DataFrame, clustering_features: list, scaler_type: str) -> pd.DataFrame:
    # Para el modelo no supervisado se usan solo las características de audio
    # (sin género ni popularidad): el objetivo es descubrir segmentos musicales
    # a partir del sonido y luego contrastarlos con género y popularidad.
    # Aquí no hay train/test, por lo que el escalado se ajusta con todo el
    # dataset; es obligatorio porque K-Means y DBSCAN usan distancias euclídeas.
    scaler = _get_scaler(scaler_type)
    X = df[clustering_features].copy()
    X[clustering_features] = scaler.fit_transform(X)
    logger.info("Input de clustering: %d filas x %d features", *X.shape)
    return X
