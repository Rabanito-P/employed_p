# pipeline/data_loader.py
import pandas as pd
import numpy as np
from config.paths import PATHS
import logging

def load_and_filter():
    """Carga y filtra columnas según tu primer bloque de código"""
    logging.info("Cargando datos crudos")
    try:
        df = pd.read_csv(PATHS['raw_data'], na_values=[" "])
        df.dropna(axis=1, how='all', inplace=True)
        
        # Definición de columnas y mapeo
        # Tus nombres de columnas
        final_columns = [
            "C207: SEXO",
            "C208: EDAD",
            "C366: NIVEL_EDUCATIVO",
            "C377: ETNIA",
            "C303: TUVO_TRABAJO",
            "C310: TIPO_TRABAJADOR",
            "C335: BUSCA_OTRO_TRABAJO",
            "OCUP300: NIVEL_OCUPACION",
            "C312: REGISTRO_SUNAT",
            "C317: TAMANO_EMPRESA",
            "C333: QUIERE_MAS_HORAS",
            "C313: LIBROS_CONTABLES",
            "C338: FRECUENCIA_PAGO",
            "ingtrabw: INGRESO_PRINCIPAL",
            "C375_1: LIMIT_MOVIMIENTO",
            "C375_2: LIMIT_VISION",
            "C375_3: LIMIT_COMUNICACION",
            "C375_4: LIMIT_AUDICION",
            "C375_5: LIMIT_APRENDIZAJE",
            "C375_6: LIMIT_RELACION",
            "SEGURO1: SEGURO_SALUD",
            "C205: DIAS_AUSENTE"
        ]
        
        column_mapping = {}
        for item in final_columns:
            col_code, col_name = item.split(": ")
            column_mapping[col_code.strip()] = col_name.strip()
            
        df_filtered = df[list(column_mapping.keys())].rename(columns=column_mapping)
        df_filtered.replace(" ", np.nan, inplace=True)
        df_filtered.to_csv(PATHS['processed_data'], index=False)
        
        logging.info(f"Datos filtrados guardados en {PATHS['processed_data']}")
        return df_filtered
        
    except Exception as e:
        logging.error(f"Error en data_loader: {str(e)}")
        raise