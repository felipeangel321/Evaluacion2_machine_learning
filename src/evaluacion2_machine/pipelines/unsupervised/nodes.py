import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def kmeans_model_selection(X: pd.DataFrame, params: dict) -> tuple:
    # Se prueban varios k y se comparan:
    # - Inercia (método del codo): suma de distancias al centroide; siempre
    #   baja al aumentar k, se busca el punto donde deja de bajar fuerte.
    # - Silhouette (-1 a 1): qué tan separado está cada punto de otros
    #   clusters respecto del suyo. Se calcula sobre una muestra por costo O(n²).
    # - Davies-Bouldin (menor es mejor) como tercer criterio.
    rs = params["random_state"]
    sample = X.sample(min(params["silhouette_sample"], len(X)), random_state=rs)

    rows = []
    for k in range(params["k_min"], params["k_max"] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=rs).fit(X)
        sample_labels = km.predict(sample)
        rows.append({
            "k": k,
            "inercia": km.inertia_,
            "silhouette": silhouette_score(sample, sample_labels),
            "davies_bouldin": davies_bouldin_score(sample, sample_labels),
        })
    results = pd.DataFrame(rows).round(4)
    logger.info("Selección de k:\n%s", results.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, col, title in zip(axes, ["inercia", "silhouette", "davies_bouldin"],
                              ["Método del codo (inercia)", "Silhouette (mayor es mejor)",
                               "Davies-Bouldin (menor es mejor)"]):
        ax.plot(results["k"], results[col], marker="o", color="#1DB954")
        ax.axvline(params["kmeans_k"], color="red", ls="--", lw=1, label=f"k elegido = {params['kmeans_k']}")
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.legend()
    fig.tight_layout()
    plt.close(fig)

    return results, fig


def fit_kmeans(X: pd.DataFrame, params: dict) -> tuple:
    # K-Means se elige porque las features de audio son continuas y escaladas
    # (distancia euclídea con sentido), escala bien a 76k canciones y produce
    # centroides interpretables como "perfil musical promedio" de cada segmento.
    km = KMeans(n_clusters=params["kmeans_k"], n_init=10, random_state=params["random_state"]).fit(X)
    labels = pd.DataFrame({"cluster": km.labels_})
    logger.info("K-Means k=%d — tamaños: %s", params["kmeans_k"], labels["cluster"].value_counts().sort_index().tolist())
    return km, labels


def pca_projection(X: pd.DataFrame, labels: pd.DataFrame, params: dict) -> tuple:
    # PCA reduce las 10 features de audio a 2 componentes solo para VISUALIZAR
    # los clusters (el clustering se hace sobre las 10 features originales).
    pca = PCA(random_state=params["random_state"]).fit(X)
    variance = pd.DataFrame({
        "componente": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "varianza_explicada": pca.explained_variance_ratio_,
        "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_),
    }).round(4)
    loadings = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"]).round(3)
    logger.info("PCA — varianza PC1+PC2: %.3f\n%s", variance["varianza_acumulada"].iloc[1], loadings.to_string())

    coords = pca.transform(X)[:, :2]
    rng = np.random.default_rng(params["random_state"])
    idx = rng.choice(len(X), size=min(8000, len(X)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    sns.scatterplot(x=coords[idx, 0], y=coords[idx, 1], hue=labels["cluster"].values[idx],
                    palette="tab10", s=8, alpha=0.5, ax=axes[0], legend="full")
    axes[0].set_xlabel(f"PC1 ({variance['varianza_explicada'][0]:.0%})")
    axes[0].set_ylabel(f"PC2 ({variance['varianza_explicada'][1]:.0%})")
    axes[0].set_title("Clusters K-Means proyectados en PCA")
    axes[0].legend(title="cluster", markerscale=3)

    axes[1].bar(variance["componente"], variance["varianza_explicada"], color="#1DB954")
    axes[1].plot(variance["componente"], variance["varianza_acumulada"], marker="o", color="#E67E22")
    axes[1].set_title("Varianza explicada por componente")
    fig.tight_layout()
    plt.close(fig)

    return variance, loadings.reset_index(names="feature"), fig


def profile_clusters(features: pd.DataFrame, labels: pd.DataFrame, clustering_features: list,
                     hit_threshold: int) -> tuple:
    # El perfil de cada cluster (media de cada feature de audio en escala
    # original) permite ponerle un nombre de negocio al segmento. Luego se
    # contrasta con variables NO usadas en el clustering: popularidad, tasa de
    # hits y géneros más frecuentes.
    df = features.reset_index(drop=True).copy()
    df["cluster"] = labels["cluster"].values
    df["is_hit"] = (df["popularity"] >= hit_threshold).astype(int)

    profile = df.groupby("cluster").agg(
        n_canciones=("popularity", "size"),
        **{c: (c, "mean") for c in clustering_features},
        popularity_media=("popularity", "mean"),
        tasa_hit=("is_hit", "mean"),
        explicit=("explicit", "mean"),
    )
    profile["pct_canciones"] = profile["n_canciones"] / len(df)

    top_genres = (
        df.groupby("cluster")["track_genre"]
        .apply(lambda s: ", ".join(s.value_counts().head(5).index))
        .rename("top_5_generos")
    )
    profile = profile.join(top_genres).round(3).reset_index()
    logger.info("Perfil de clusters:\n%s", profile[["cluster", "n_canciones", "popularity_media", "tasa_hit"]].to_string(index=False))

    # z-score de las medias por cluster para comparar features de distinta escala
    means = df.groupby("cluster")[clustering_features].mean()
    z = (means - df[clustering_features].mean()) / df[clustering_features].std()

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    sns.heatmap(z, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Perfil de audio por cluster (z-score vs media global)")
    pop = profile.set_index("cluster")[["popularity_media"]]
    bars = axes[1].bar(pop.index.astype(str), pop["popularity_media"], color="#1DB954")
    axes[1].bar_label(bars, labels=[f"{p:.1f}\n({h:.0%} hits)" for p, h in
                                    zip(profile["popularity_media"], profile["tasa_hit"])])
    axes[1].axhline(df["popularity"].mean(), color="red", ls="--", lw=1, label="media global")
    axes[1].set_ylim(0, pop["popularity_media"].max() * 1.25)
    axes[1].set_title("Popularidad media por cluster")
    axes[1].set_xlabel("cluster")
    axes[1].legend()
    fig.tight_layout()
    plt.close(fig)

    return profile, fig


def dbscan_comparison(X: pd.DataFrame, labels: pd.DataFrame, params: dict) -> pd.DataFrame:
    # DBSCAN agrupa por densidad y no necesita k. Se usa como contraste: si los
    # datos tuvieran grupos densos bien separados, DBSCAN los encontraría.
    # Por costo se aplica sobre una muestra y sobre 5 componentes PCA.
    rs = params["random_state"]
    sample_idx = X.sample(min(params["silhouette_sample"], len(X)), random_state=rs).index
    X_pca = PCA(n_components=params["dbscan_pca_components"], random_state=rs).fit_transform(X)
    Xs = X_pca[sample_idx]

    rows = []
    for eps in params["dbscan_eps"]:
        db_labels = DBSCAN(eps=eps, min_samples=params["dbscan_min_samples"]).fit_predict(Xs)
        mask = db_labels != -1
        n_clusters = len(set(db_labels[mask]))
        sil = silhouette_score(Xs[mask], db_labels[mask]) if n_clusters > 1 else np.nan
        sizes = pd.Series(db_labels[mask]).value_counts()
        rows.append({
            "modelo": "DBSCAN", "eps": eps, "n_clusters": n_clusters,
            "pct_ruido": 1 - mask.mean(),
            "pct_cluster_mayor": sizes.iloc[0] / len(db_labels) if n_clusters else np.nan,
            "silhouette": sil,
        })

    km_labels = labels["cluster"].values[sample_idx]
    rows.append({
        "modelo": "KMeans", "eps": np.nan, "n_clusters": params["kmeans_k"], "pct_ruido": 0.0,
        "pct_cluster_mayor": pd.Series(km_labels).value_counts().iloc[0] / len(km_labels),
        "silhouette": silhouette_score(Xs, km_labels),
    })
    results = pd.DataFrame(rows).round(4)
    logger.info("DBSCAN vs KMeans:\n%s", results.to_string(index=False))
    return results
