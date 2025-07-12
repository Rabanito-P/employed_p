# pipeline/k_finder.py
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from config.paths import PATHS
import logging

def find_optimal_k():
    """Determina k óptimo con método del codo y silhouette"""
    logging.info("Buscando k óptimo")
    try:
        df_pca = pd.read_csv(PATHS['pca_data'])
        k_range = range(2, 11)
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(df_pca)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(df_pca, labels))
        
        # Gráficos
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        plt.plot(k_range, inertias, marker='o')
        plt.title("Método del Codo")
        
        plt.subplot(122)
        plt.plot(k_range, silhouettes, marker='o', color='green')
        plt.title("Silhouette Score")
        
        plt.savefig(PATHS['elbow_plot'])
        plt.savefig(PATHS['silhouette_plot'])
        
        # Sugerir k óptimo (máximo silhouette)
        optimal_k = k_range[silhouettes.index(max(silhouettes))]
        logging.info(f"K óptimo sugerido: {optimal_k}")
        return optimal_k
        
    except Exception as e:
        logging.error(f"Error en búsqueda de k: {str(e)}")
        raise