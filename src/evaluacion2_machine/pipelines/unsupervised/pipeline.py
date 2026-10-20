from kedro.pipeline import Pipeline, node

from .nodes import dbscan_comparison, fit_kmeans, kmeans_model_selection, pca_projection, profile_clusters


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(kmeans_model_selection,
             inputs=["spotify_clustering_input", "params:unsupervised"],
             outputs=["kmeans_selection_results", "plot_kmeans_selection"],
             name="kmeans_model_selection_node"),

        node(fit_kmeans,
             inputs=["spotify_clustering_input", "params:unsupervised"],
             outputs=["kmeans_model", "cluster_labels"],
             name="fit_kmeans_node"),

        node(pca_projection,
             inputs=["spotify_clustering_input", "cluster_labels", "params:unsupervised"],
             outputs=["pca_variance_results", "pca_loadings", "plot_pca_clusters"],
             name="pca_projection_node"),

        node(profile_clusters,
             inputs=["spotify_features", "cluster_labels",
                     "params:clustering_features", "params:hit_threshold"],
             outputs=["cluster_profiles", "plot_cluster_profiles"],
             name="profile_clusters_node"),

        node(dbscan_comparison,
             inputs=["spotify_clustering_input", "cluster_labels", "params:unsupervised"],
             outputs="dbscan_results",
             name="dbscan_comparison_node"),
    ])
