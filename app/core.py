"""Lógica de la aplicación demo (sin Streamlit, para poder probarla aparte).

La app NO reentrena nada: carga los modelos que guardó `kedro run` y aplica
exactamente la misma transformación que el pipeline data_transformation
(One-Hot de género/tonalidad/compás + StandardScaler ajustado en train).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((ROOT / "conf/base/parameters.yml").read_text(encoding="utf-8"))

HIT_THRESHOLD = PARAMS["hit_threshold"]
ID_COLUMNS = PARAMS["id_columns"]
OHE_COLUMNS = PARAMS["categorical_cols_ohe"]
CLUSTER_FEATURES = PARAMS["clustering_features"]
MAE_TEST = 8.65  # MAE de Gradient Boosting en test (results/metrics/reg_test_metrics.csv)

# Géneros donde se midió el sesgo (MAE 2.3x mayor, README sección de sesgo)
BIASED_GENRES = {"latin", "latino", "reggaeton", "reggae"}

# Nombres de negocio de los 6 clusters (perfiles en results/metrics/cluster_profiles.csv)
SEGMENTS = {
    0: ("Electrónica instrumental", "Playlists funcionales: concentración, clubes, entrenamiento."),
    1: ("Intensa / rock-metal", "Promoción dirigida a nichos fieles; potencial medio."),
    2: ("Acústica melódica", "Prioridad de promoción: alto potencial (30% de hits)."),
    3: ("Ambiental / relajación", "Playlists funcionales: dormir, estudiar, meditar."),
    4: ("Hablada / en vivo", "Bajo potencial comercial (4% de hits): no priorizar."),
    5: ("Bailable y alegre", "Prioridad de promoción: el segmento con más hits (30%)."),
}

AUDIO_SLIDERS = {  # variable: (mín, máx, paso, descripción)
    "danceability": (0.0, 1.0, 0.01, "Qué tan bailable es"),
    "energy": (0.0, 1.0, 0.01, "Intensidad y actividad"),
    "valence": (0.0, 1.0, 0.01, "Positividad (alegre ↔ triste)"),
    "acousticness": (0.0, 1.0, 0.01, "Probabilidad de ser acústica"),
    "instrumentalness": (0.0, 1.0, 0.01, "Ausencia de voz"),
    "speechiness": (0.0, 1.0, 0.01, "Presencia de palabra hablada"),
    "liveness": (0.0, 1.0, 0.01, "Probabilidad de ser en vivo"),
    "loudness": (-50.0, 5.0, 0.5, "Volumen medio (dB)"),
    "tempo": (40.0, 220.0, 1.0, "Pulsaciones por minuto"),
    "duration_min": (0.5, 10.0, 0.1, "Duración en minutos"),
}


def load_artifacts() -> dict:
    """Carga modelos, scaler y datos ya procesados por Kedro."""
    models = ROOT / "models/trained_models"
    features = pd.read_csv(ROOT / "data/04_feature/spotify_features.csv", encoding="utf-8")
    x_columns = pd.read_csv(ROOT / "data/05_model_input/X_test.csv", nrows=0).columns.tolist()

    # Mismo split que el pipeline (80/20 estratificado, semilla 42): así la app
    # muestra el desempeño solo en canciones que el modelo nunca vio.
    is_hit = (features["popularity"] >= HIT_THRESHOLD).astype(int)
    _, test_idx = train_test_split(
        np.arange(len(features)),
        test_size=PARAMS["modeling"]["test_size"],
        random_state=PARAMS["modeling"]["random_state"],
        stratify=is_hit,
    )

    # El scaler del clustering se ajusta con todo el dataset en el pipeline
    # (prepare_clustering_input); aquí se reajusta igual, es determinista.
    cluster_scaler = StandardScaler().fit(features[CLUSTER_FEATURES])

    return {
        "clf": joblib.load(models / "gradient_boosting_clf.joblib"),
        "reg": joblib.load(models / "gradient_boosting_reg.joblib"),
        "kmeans": joblib.load(models / "kmeans_model.joblib"),
        "scaler": joblib.load(ROOT / "data/06_models/scaler.pkl"),
        "cluster_scaler": cluster_scaler,
        "features": features,
        "test": features.iloc[test_idx].reset_index(drop=True),
        "x_columns": x_columns,
        "genres": sorted(features["track_genre"].unique()),
    }


def to_model_input(rows: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Filas con el formato de spotify_features → matriz X de 144 columnas."""
    df = rows.drop(columns=[c for c in ID_COLUMNS + ["popularity"] if c in rows.columns])
    df = pd.get_dummies(df, columns=OHE_COLUMNS, drop_first=False)
    # reindex: una sola canción solo genera su dummy; el resto va en 0
    df = df.reindex(columns=art["x_columns"], fill_value=0).astype(float)
    num = list(art["scaler"].feature_names_in_)
    df[num] = art["scaler"].transform(df[num])
    return df


def predict(rows: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Probabilidad de hit, popularidad estimada y segmento para cada fila."""
    X = to_model_input(rows, art)
    Xc = pd.DataFrame(art["cluster_scaler"].transform(rows[CLUSTER_FEATURES]), columns=CLUSTER_FEATURES)
    return pd.DataFrame({
        "prob_hit": art["clf"].predict_proba(X)[:, 1],
        "pop_estimada": np.clip(art["reg"].predict(X), 1, 100),
        "segmento": art["kmeans"].predict(Xc),
    }, index=rows.index)


def promotion_curve(test_pred: pd.DataFrame, y_hit: pd.Series, budgets=range(1, 51)) -> pd.DataFrame:
    """% de hits reales al promocionar el top b% del ranking del clasificador."""
    order = test_pred["prob_hit"].sort_values(ascending=False).index
    hits_sorted = y_hit.loc[order].to_numpy()
    rows = []
    for b in budgets:
        k = max(1, round(len(order) * b / 100))
        rows.append({
            "presupuesto_%": b,
            "% hits en lo promocionado": hits_sorted[:k].mean() * 100,
            "% hits al azar": y_hit.mean() * 100,
        })
    return pd.DataFrame(rows)
