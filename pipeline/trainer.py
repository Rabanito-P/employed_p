# pipeline/trainer.py
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from config.paths import PATHS
import logging

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocesador personalizado que replica tu lógica de preprocessor.py"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Transformar variables binarias (2 -> 0)
        binarias = [
            'SEXO', 'TUVO_TRABAJO', 'BUSCA_OTRO_TRABAJO',
            'QUIERE_MAS_HORAS', 'SEGURO_SALUD',
            'LIMIT_MOVIMIENTO', 'LIMIT_VISION', 'LIMIT_COMUNICACION',
            'LIMIT_AUDICION', 'LIMIT_APRENDIZAJE', 'LIMIT_RELACION'
        ]
        
        for col in binarias:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace({2: 0})
        
        # 2. Crear variable DISCAPACIDAD
        disc_cols = [
            'LIMIT_MOVIMIENTO', 'LIMIT_VISION', 'LIMIT_COMUNICACION',
            'LIMIT_AUDICION', 'LIMIT_APRENDIZAJE', 'LIMIT_RELACION'
        ]
        
        available_disc_cols = [col for col in disc_cols if col in X_copy.columns]
        if available_disc_cols:
            X_copy["DISCAPACIDAD"] = X_copy[available_disc_cols].sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)
            X_copy.drop(columns=available_disc_cols, inplace=True)
        
        return X_copy

def train_and_save(k, raw_df):
    """Entrena y guarda TODO el pipeline de transformación + modelo con datos crudos"""
    logging.info(f"Iniciando entrenamiento con k={k}")
    try:
        # Definir columnas
        numericas = ['EDAD', 'INGRESO_PRINCIPAL', 'DIAS_AUSENTE']
        nominales = [
            'ETNIA', 'TIPO_TRABAJADOR', 'NIVEL_OCUPACION',
            'REGISTRO_SUNAT', 'LIBROS_CONTABLES', 'FRECUENCIA_PAGO'
        ]
        
        # Crear transformador de columnas
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numericas),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), nominales)
        ], remainder='passthrough')  # Las binarias pasan sin cambios
        
        # Pipeline completo
        full_pipeline = Pipeline([
            ('custom_prep', CustomPreprocessor()),  # Tu lógica personalizada
            ('column_transform', preprocessor),     # One-hot encoding y escalado
            ('imputer', SimpleImputer(strategy='median')),
            ('pca', PCA(n_components=9)),
            ('kmeans', KMeans(n_clusters=k, random_state=42, n_init='auto'))
        ])
        
        # Entrenar con datos crudos
        full_pipeline.fit(raw_df)
        
        # Serialización dual
        joblib.dump(full_pipeline, PATHS['model_joblib'])
        with open(PATHS['model_pkl'], 'wb') as f:
            pickle.dump(full_pipeline, f)
            
        logging.info(f"Pipeline completo guardado en {PATHS['model_joblib']} y {PATHS['model_pkl']}")
        
        # Crear wrapper para compatibilidad
        class ModelWrapper:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                self.labels_ = pipeline.named_steps['kmeans'].labels_
                self.n_clusters = pipeline.named_steps['kmeans'].n_clusters
                
            def predict(self, X):
                return self.pipeline.predict(X)
                
            def fit_predict(self, X):
                return self.pipeline.fit_predict(X)
        
        model_wrapper = ModelWrapper(full_pipeline)
        
        return model_wrapper
        
    except Exception as e:
        logging.error(f"Error en entrenamiento: {str(e)}")
        raise