# pipeline/evaluator.py
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config.paths import PATHS
import logging
import numpy as np
from sklearn.utils import resample


def evaluate_model(model, df_pca):
    """Genera reportes completos de evaluación y análisis de clusters"""
    logging.info("Generando reportes de evaluación y análisis")
    try:
        # === 1. Calcular métricas de evaluación ===
        labels = model.labels_
        
        # Calcular Silhouette con muestra (para evitar problemas de memoria)
        sample_size = min(1000, len(df_pca))
        sample_indices = resample(range(len(df_pca)), n_samples=sample_size, random_state=42)
        sample_data = df_pca.iloc[sample_indices]
        sample_labels = labels[sample_indices]
        silhouette = silhouette_score(sample_data, sample_labels)
        
        
        calinski = calinski_harabasz_score(df_pca, labels)
        davies = davies_bouldin_score(df_pca, labels)
        
        # Guardar métricas con encoding UTF-8
        with open(PATHS['metrics_report'], 'w', encoding='utf-8') as f:
            f.write("[EVALUACION] Modelo K-Means\n")
            f.write("--------------------------------\n")
            f.write(f"Silhouette Score:       {silhouette:.4f} (mejor valores cercanos a 1)\n")
            f.write(f"Calinski-Harabasz Index: {calinski:.2f} (mejor valores altos)\n")
            f.write(f"Davies-Bouldin Score:   {davies:.4f} (mejor valores cercanos a 0)\n")
        
        # === 2. Análisis de clusters ===
        # Cargar datos originales
        df_original = pd.read_csv(PATHS['scaled_data'])
        df_original['Cluster'] = labels
        
        # Guardar dataset completo con clusters
        clustered_data_path = os.path.join(
            os.path.dirname(PATHS['scaled_data']), 
            f"employ_clustered_k{model.n_clusters}.csv"
        )
        df_original.to_csv(clustered_data_path, index=False, encoding='utf-8')
        
        # Generar reporte estadístico por cluster
        cluster_report = []
        for cluster in sorted(df_original['Cluster'].unique()):
            cluster_df = df_original[df_original['Cluster'] == cluster]
            
            # Estadísticas resumidas
            stats = cluster_df.describe().loc[['mean', 'std', '50%']].T
            stats['Cluster'] = cluster
            stats['Count'] = len(cluster_df)
            cluster_report.append(stats)
        
        # Consolidar reporte
        full_report = pd.concat(cluster_report)
        full_report.to_csv(PATHS['cluster_report'], encoding='utf-8')
        
        # === 3. Visualización de perfiles ===
        plt.figure(figsize=(14, 8))
        numeric_cols = df_original.select_dtypes(include='number').columns.tolist()
        
        # Limitar a 12 gráficos para evitar sobrecarga
        num_plots = min(12, len(numeric_cols))
        cols_per_row = 4
        rows = int(np.ceil(num_plots / cols_per_row))
        
        for i, col in enumerate(numeric_cols[:num_plots]):
            plt.subplot(rows, cols_per_row, i+1)
            sns.boxplot(x='Cluster', y=col, data=df_original)
            plt.title(f'Distribucion de {col}')
            plt.tight_layout()
        
        # Guardar visualización
        profiles_path = os.path.join(os.path.dirname(PATHS['cluster_report']), 'cluster_profiles.png')
        plt.savefig(profiles_path)
        plt.close()
        
        logging.info(f"Reportes completos generados en {PATHS['cluster_report']}")
        
        # === 4. Impresión resumen en consola ===
        print("\n" + "="*50)
        print(f"=== EVALUACION FINAL (k={model.n_clusters}) ===")
        print("="*50)
        print(f"• Silhouette: {silhouette:.4f} | Calinski-Harabasz: {calinski:.2f} | Davies-Bouldin: {davies:.4f}")
        print(f"• Tamaños de clusters: {pd.Series(labels).value_counts().to_dict()}")
        print(f"• Reportes guardados en: {os.path.dirname(PATHS['cluster_report'])}")
        
    except Exception as e:
        logging.error(f"Error en evaluación: {str(e)}")
        raise