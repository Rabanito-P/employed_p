# Proyecto de Segmentación Laboral con K-Means

## 📋 Descripción General

Este proyecto implementa un sistema completo de segmentación de trabajadores mediante técnicas de clustering (K-Means) con una API REST para realizar predicciones en tiempo real. Permite identificar distintos perfiles laborales a partir de características sociodemográficas y condiciones de empleo.

La solución incluye:
- Pipeline completo de preprocesamiento, reducción dimensional y entrenamiento
- Herramientas de evaluación y visualización de clusters
- API REST con FastAPI para realizar predicciones
- Sistema de autenticación mediante API Keys

## 🏗️ Estructura del Proyecto

```
employ-kmeans/
│
├── app.py                     # API REST con FastAPI
├── .env                       # Variables de entorno (API KEY)
├── run_pipeline.py            # Script para ejecutar el pipeline completo
│
├── config/                    # Configuración de rutas
│   └── paths.py               # Definición de rutas a archivos
│
├── pipeline/                  # Módulos del pipeline
│   ├── __init__.py
│   ├── data_loader.py         # Carga y filtrado inicial
│   ├── preprocessor.py        # Transformación de variables
│   ├── dimensionality_reducer.py  # Reducción PCA
│   ├── k_finder.py            # Determinación de k óptimo
│   ├── trainer.py             # Entrenamiento del modelo
│   └── evaluator.py           # Evaluación y reportes
│
├── data/
│   ├── raw/                   # Datos originales
│   │   └── employ25_3-4-5.csv
│   └── processed/             # Datos procesados
│       ├── employ_processed.csv
│       ├── employ_preprocessed_KMEANS.csv
│       ├── employ_pca_9d.csv
│       └── employ_clustered_k4.csv
│
├── models/                    # Modelos entrenados
│   ├── kmeans_model.joblib
│   └── kmeans_model.pkl
│
├── notebooks/                 # Jupyter notebooks
│   └── tryend.ipynb
│
├── logs/                      # Registros de ejecución
│   └── execution.log
│
└── reports/                   # Reportes y visualizaciones
    ├── cluster_analysis.csv
    ├── metrics.txt
    └── visuals/
        ├── elbow_plot.png
        ├── pca_variance.png
        └── silhouette_plot.png
```

## 🚀 Primeros Pasos

### Requisitos Previos

- Python 3.8+
- Dependencias principales:
  - pandas
  - scikit-learn
  - fastapi
  - joblib
  - python-dotenv
  - matplotlib
  - seaborn

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/Angel226m/employ-kmeans.git
cd employ-kmeans
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar la API Key para el servicio:
```bash
# Crear archivo .env con la siguiente variable
API_KEY=tu_clave_secreta_aquí
```

## 💻 Uso del Sistema

### Entrenamiento del Modelo

Para ejecutar el pipeline completo de entrenamiento:

```bash
python run_pipeline.py
```

Este script ejecuta secuencialmente:
1. Carga y filtrado de datos iniciales
2. Preprocesamiento (transformación de variables)
3. Reducción dimensional con PCA
4. Determinación del k óptimo para K-Means
5. Entrenamiento del modelo
6. Evaluación y generación de reportes

Durante la ejecución, el sistema solicitará un valor para k (número de clusters):
```
[SUGERENCIA] K óptimo calculado: 4
Ingrese el valor de k a utilizar: 
```

### Ejecución de la API REST

Para iniciar el servidor de la API:

```bash
uvicorn app:app --reload
```

La API estará disponible en: http://localhost:8000

Endpoint principal:
- **POST /predecir**: Realiza una predicción del cluster al que pertenece un trabajador

Ejemplo de solicitud:
```json
{
  "SEXO": 1,
  "TUVO_TRABAJO": 1,
  "BUSCA_OTRO_TRABAJO": 0,
  "QUIERE_MAS_HORAS": 0,
  "SEGURO_SALUD": 1,
  "LIMIT_MOVIMIENTO": 0,
  "LIMIT_VISION": 0,
  "LIMIT_COMUNICACION": 0,
  "LIMIT_AUDICION": 0,
  "LIMIT_APRENDIZAJE": 0,
  "LIMIT_RELACION": 0,
  "EDAD": 45,
  "INGRESO_PRINCIPAL": 3500.0,
  "DIAS_AUSENTE": 0,
  "NIVEL_EDUCATIVO": 11,
  "TAMANO_EMPRESA": 3,
  "ETNIA": "1",
  "TIPO_TRABAJADOR": "1.0",
  "REGISTRO_SUNAT": "1.0",
  "LIBROS_CONTABLES": "1.0",
  "FRECUENCIA_PAGO": "1.0",
  "NIVEL_OCUPACION": "1"
}
```

Respuesta:
```json
{
  "cluster": 2,
  "perfil": "Profesionales y Empleados Formales de Altos Ingresos y Estables"
}
```

**IMPORTANTE**: Todas las solicitudes deben incluir el header `X-API-Key` con la clave configurada.

### Predicción por Lotes

Para procesar un archivo CSV con múltiples registros:

1. Preparar un archivo CSV con las mismas columnas que el dataset original
2. Ejecutar el script de predicción por lotes:
```bash
python notebooks/pred_batch.py --input input_file.csv --output predictions.csv
```

## 🔍 Detalles Técnicos

### Preprocesamiento de Datos

El pipeline aplica las siguientes transformaciones:

1. **Recodificación de variables binarias**:
   - Conversión de valores `2` a `0` en variables como SEXO, TUVO_TRABAJO, etc.

2. **Creación de variables derivadas**:
   - Variable `DISCAPACIDAD` basada en seis limitaciones físicas/cognitivas

3. **One-hot encoding**:
   - Transformación de variables categóricas como ETNIA, TIPO_TRABAJADOR, etc.

4. **Escalado de variables numéricas**:
   - Estandarización de EDAD, INGRESO_PRINCIPAL y DIAS_AUSENTE

5. **Imputación de valores faltantes**:
   - Método MICE (Multiple Imputation by Chained Equations) para NaN

### Reducción Dimensional

Para optimizar el clustering se aplica PCA (Análisis de Componentes Principales):

- Reducción a 9 dimensiones principales
- Preservación aproximada del 85% de la varianza original
- Mejora de la separabilidad entre clusters

### Modelo de Clustering

Se utiliza K-Means con las siguientes características:

- Número de clusters determinado mediante método del codo y coeficiente de silueta
- Inicialización optimizada (k-means++)
- Métricas de evaluación:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Score

### Perfiles de Clusters Identificados

El sistema identifica cuatro perfiles principales:

1. **Adultos Mayores con Baja Educación y Alta Formalidad en Salud**
   - Características: Mayor edad, educación básica, alta cobertura de salud

2. **Trabajadores Informales de Baja Calificación y Menores Ingresos**
   - Características: Sin registro formal, ingresos bajos, empleos temporales

3. **Profesionales y Empleados Formales de Altos Ingresos y Estables**
   - Características: Alta educación, ingresos elevados, empleo estable

4. **Jóvenes Calificados en Empleos con Riesgo de Subempleo/Informalidad**
   - Características: Educación media-alta, ingresos medios, empleos variados

### API REST

Desarrollada con FastAPI, la API ofrece:

- Serialización/deserialización automática con Pydantic
- Autenticación mediante API Keys (header `X-API-Key`)
- Documentación automática (disponible en `/docs`)
- Validación de datos de entrada
- Manejo de excepciones y errores HTTP

## 📊 Evaluación y Reportes

El sistema genera automáticamente:

1. **Métricas de calidad del clustering**:
   - Archivo `reports/metrics.txt` con indicadores clave

2. **Análisis estadístico por cluster**:
   - Archivo `reports/cluster_analysis.csv` con estadísticas descriptivas

3. **Visualizaciones**:
   - Método del codo para determinar k óptimo
   - Coeficiente de silueta por valor de k
   - Varianza explicada por PCA
   - Distribución de variables por cluster

## 👨‍💻 Contribución y Desarrollo

Para contribuir al proyecto:

1. Crear un fork del repositorio
2. Crear una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Realizar cambios y commits (`git commit -am 'Añadir nueva funcionalidad'`)
4. Enviar cambios a tu fork (`git push origin feature/nueva-funcionalidad`)
5. Crear un Pull Request

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 🔗 Referencias

- Scikit-learn: https://scikit-learn.org/
- FastAPI: https://fastapi.tiangolo.com/
- Pandas: https://pandas.pydata.org/
- Joblib: https://joblib.readthedocs.io/
