# pipeline/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.paths import PATHS
import logging

def preprocess_data(df):
    """Transforma variables como en tu segundo bloque"""
    logging.info("Iniciando preprocesamiento")
    try:
        # Tus variables binarias
        binarias = [
            'SEXO', 
            'TUVO_TRABAJO', 
            'BUSCA_OTRO_TRABAJO',
            'QUIERE_MAS_HORAS', 
            'SEGURO_SALUD',
            'LIMIT_MOVIMIENTO', 
            'LIMIT_VISION', 
            'LIMIT_COMUNICACION',
            'LIMIT_AUDICION', 
            'LIMIT_APRENDIZAJE', 
            'LIMIT_RELACION'
        ]
        
        for col in binarias:
            df[col] = df[col].replace({2: 0})
        
        # Crear discapacidad
        # Tus columnas de discapacidad
        disc_cols = [
            'LIMIT_MOVIMIENTO', 
            'LIMIT_VISION', 
            'LIMIT_COMUNICACION',
            'LIMIT_AUDICION', 
            'LIMIT_APRENDIZAJE', 
            'LIMIT_RELACION'
        ]
        df["DISCAPACIDAD"] = df[disc_cols].sum(axis=1).apply(lambda x: 1 if x >= 1 else 0)
        df.drop(columns=disc_cols, inplace=True)
        
        # One-hot encoding
        # Tus variables nominales
        nominales = [
            'ETNIA', 
            'TIPO_TRABAJADOR', 
            'NIVEL_OCUPACION',
            'REGISTRO_SUNAT', 
            'LIBROS_CONTABLES', 
            'FRECUENCIA_PAGO'
        ]
        
        df = pd.get_dummies(df, columns=nominales, drop_first=True)
        
        # Escalado numéricas
        # Tus variables numéricas
        numericas = [
            'EDAD', 
            'INGRESO_PRINCIPAL', 
            'DIAS_AUSENTE'
        ]
        scaler = StandardScaler()
        df[numericas] = scaler.fit_transform(df[numericas])
        
        df.to_csv(PATHS['scaled_data'], index=False)
        logging.info(f"Datos preprocesados guardados en {PATHS['scaled_data']}")
        return df
        
    except Exception as e:
        logging.error(f"Error en preprocesamiento: {str(e)}")
        raise