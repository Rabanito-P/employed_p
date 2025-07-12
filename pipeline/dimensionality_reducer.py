# pipeline/dimensionality_reducer.py
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from config.paths import PATHS

def reduce_dimensionality():
    """Aplica PCA y reduce a 9 dimensiones con manejo robusto de NaN"""
    logging.info("Iniciando reducción dimensional")
    try:
        # Cargar datos preprocesados
        df = pd.read_csv(PATHS['scaled_data'])
        
        # 1. Verificar y reportar NaNs
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logging.warning(f"Se encontraron {nan_count} valores NaN en los datos. Aplicando imputación...")
            
            # 2. Imputación avanzada (MICE)
            imputer = IterativeImputer(
                max_iter=10,
                random_state=42,
                n_nearest_features=10,
                sample_posterior=True
            )
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns
            )
            
            # Verificar que no queden NaNs
            if df_imputed.isnull().sum().sum() > 0:
                logging.error("Falló la imputación. Todavía hay valores NaN.")
                # Imputación de respaldo
                df_imputed = df.fillna(df.median())
        else:
            df_imputed = df.copy()
        
        # 3. Aplicar PCA
        pca = PCA(n_components=9)
        pca_result = pca.fit_transform(df_imputed)
        df_pca = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(9)])
        df_pca.to_csv(PATHS['pca_data'], index=False)
        
        # 4. Gráfico de varianza explicada
        plt.figure(figsize=(10,6))
        plt.plot(range(1,10), np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.title('Varianza Acumulada PCA')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Varianza Acumulada')
        plt.grid(True)
        plt.savefig(PATHS['pca_variance_plot'])
        plt.close()
        
        logging.info(f"Datos PCA guardados en {PATHS['pca_data']}")
        return df_pca
        
    except Exception as e:
        logging.error(f"Error en reducción dimensional: {str(e)}")
        raise