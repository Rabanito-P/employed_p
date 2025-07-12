# config/paths.py
import os
import sys

# Verificar si stdout existe antes de reconfigurar
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Asegúrate que estos paths existen


PATHS = {
    'raw_data': os.path.join(BASE_DIR, 'data/raw/employ25_3-4-5.csv'),
    'processed_data': os.path.join(BASE_DIR, 'data/processed/employ_processed.csv'),
    'scaled_data': os.path.join(BASE_DIR, 'data/processed/employ_preprocessed_KMEANS.csv'),
    'pca_data': os.path.join(BASE_DIR, 'data/processed/employ_pca_9d.csv'),
    'model_joblib': os.path.join(BASE_DIR, 'models/kmeans_model.joblib'),
    'model_pkl': os.path.join(BASE_DIR, 'models/kmeans_model.pkl'),
    'cluster_report': os.path.join(BASE_DIR, 'reports/cluster_analysis.csv'),
    'metrics_report': os.path.join(BASE_DIR, 'reports/metrics.txt'),
    'elbow_plot': os.path.join(BASE_DIR, 'reports/visuals/elbow_plot.png'),
    'silhouette_plot': os.path.join(BASE_DIR, 'reports/visuals/silhouette_plot.png'),
    'execution_log': os.path.join(BASE_DIR, 'logs/execution.log'),
    'pca_variance_plot': os.path.join(BASE_DIR, 'reports/visuals/pca_variance.png')
}


os.makedirs(os.path.dirname(PATHS['processed_data']), exist_ok=True)
os.makedirs(os.path.dirname(PATHS['model_joblib']), exist_ok=True)