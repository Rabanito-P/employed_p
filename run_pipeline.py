# run_pipeline.py
import logging
from pipeline import data_loader, preprocessor, dimensionality_reducer, k_finder, trainer, evaluator
from config.paths import PATHS
import os

# Configurar logging
logging.basicConfig(
    filename=PATHS['execution_log'],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        logging.info("==== INICIO DE PIPELINE ====")
        
        # Paso 1: Carga y filtrado
        raw_df = data_loader.load_and_filter()  # Cambiar 'df' por 'raw_df'
        
        # Paso 2: Preprocesamiento
        processed_df = preprocessor.preprocess_data(raw_df.copy())  # Usar copia de raw_df
        
        # Paso 3: Reducción dimensional
        pca_df = dimensionality_reducer.reduce_dimensionality()
        
        # Paso 4: Determinar k óptimo
        optimal_k = k_finder.find_optimal_k()
        
        # Interacción usuario
        print(f"\n[SUGERENCIA] K óptimo calculado: {optimal_k}")
        user_k = int(input("Ingrese el valor de k a utilizar: "))
        
        # Paso 5: Entrenamiento - CAMBIO AQUÍ
        model = trainer.train_and_save(user_k, raw_df)  # Pasar datos crudos en lugar de processed_df

        # Paso 6: Evaluación (pasar datos PCA para evaluación)
        evaluator.evaluate_model(model, pca_df)
        
        logging.info("==== PIPELINE COMPLETADO EXITOSAMENTE ====")
        print("\n✅ Proceso completado! Verifique los archivos en /reports y /models")
        
    except Exception as e:
        logging.error(f"Error en pipeline: {str(e)}", exc_info=True)
        print(f"❌ Error en ejecución: {str(e)}")

if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs(os.path.dirname(PATHS['processed_data']), exist_ok=True)
    os.makedirs(os.path.dirname(PATHS['model_joblib']), exist_ok=True)
    os.makedirs(os.path.dirname(PATHS['cluster_report']), exist_ok=True)
    os.makedirs(os.path.dirname(PATHS['execution_log']), exist_ok=True)
    
    main()