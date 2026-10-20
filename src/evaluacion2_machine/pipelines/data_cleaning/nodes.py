import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_columns(data: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    # 'Unnamed: 0' es el índice que quedó guardado en el CSV original:
    # no es una característica de la canción y solo agregaría ruido.
    existing_cols = [c for c in columns_to_drop if c in data.columns]
    logger.info(f"Eliminando columnas: {existing_cols}")
    return data.drop(columns=existing_cols)


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # El EDA (NB01) mostró una única fila con nulos (sin artista, álbum ni
    # nombre). Representa el 0.001% del dataset, por lo que eliminarla no
    # tiene impacto y es más simple que imputar texto.
    n_before = len(df)
    df = df.dropna()
    logger.info(f"Filas con nulos eliminadas: {n_before - len(df)}")
    return df


def remove_invalid_values(df: pd.DataFrame, invalid_zero_cols: list) -> pd.DataFrame:
    # Una canción no puede tener 0 BPM, compás 0 ni duración 0 ms:
    # son errores de medición de la API de Spotify detectados en el EDA.
    mask = (df[invalid_zero_cols] == 0).any(axis=1)
    logger.info(f"Filas con valores imposibles (0 en {invalid_zero_cols}): {mask.sum()}")
    return df[~mask]


def deduplicate_tracks(df: pd.DataFrame, dedupe_by_name_artist: bool) -> pd.DataFrame:
    # Una misma canción (track_id) aparece hasta 9 veces, una por cada género
    # en que fue listada, con idénticas características de audio. Si se dejan
    # todas, la misma canción quedaría en train y test (data leakage).
    # Se ordena por popularidad descendente para conservar el registro más
    # reciente cuando la popularidad difiere entre copias (720 casos).
    df = df.sort_values("popularity", ascending=False)
    n_before = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first")
    logger.info(f"Duplicados por track_id eliminados: {n_before - len(df)}")

    # Además, la misma canción publicada en distintos álbumes (single, álbum,
    # recopilatorio) tiene distinto track_id pero el mismo nombre y artista.
    if dedupe_by_name_artist:
        n_before = len(df)
        df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
        logger.info(f"Duplicados por nombre+artista eliminados: {n_before - len(df)}")

    return df.sort_index().reset_index(drop=True)


def filter_zero_popularity(df: pd.DataFrame, drop_zero_popularity: bool) -> pd.DataFrame:
    # El 14% de las canciones tiene popularity = 0, un pico que rompe la
    # distribución (sin ellos es aproximadamente normal). El EDA (NB01, 8.1)
    # mostró que muchos son éxitos tomados de recopilatorios o playlists:
    # Spotify calcula la popularidad por versión y esas copias no acumulan
    # reproducciones propias. Es un artefacto de recopilación, no
    # impopularidad real, y mantenerlos sesgaría el target de regresión.
    if not drop_zero_popularity:
        return df
    n_before = len(df)
    df = df[df["popularity"] > 0].reset_index(drop=True)
    logger.info(f"Canciones con popularity = 0 eliminadas: {n_before - len(df)}")
    return df
