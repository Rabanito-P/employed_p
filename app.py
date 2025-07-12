# Agregar al inicio de app.py
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Annotated

load_dotenv()

# Cargar modelo
model = joblib.load('models/kmeans_model.joblib')

# Obtener API Key de variables de entorno (mejor práctica)
API_KEY = os.getenv("API_KEY")  # Usar variable de entorno para seguridad
if not API_KEY:
    raise RuntimeError("API_KEY no configurada en variables de entorno")

app = FastAPI()

# Función para verificar API Key
async def verify_api_key(api_key: Annotated[str, Header(alias="X-API-Key")] = None):
    if api_key is None:
        raise HTTPException(status_code=401, detail="API Key faltante")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    return True

# Modelo de solicitud - exactamente como en tu notebook
class PredictionRequest(BaseModel):
    SEXO: int
    TUVO_TRABAJO: int
    BUSCA_OTRO_TRABAJO: int
    QUIERE_MAS_HORAS: int
    SEGURO_SALUD: int
    LIMIT_MOVIMIENTO: int
    LIMIT_VISION: int
    LIMIT_COMUNICACION: int
    LIMIT_AUDICION: int
    LIMIT_APRENDIZAJE: int
    LIMIT_RELACION: int
    EDAD: int
    INGRESO_PRINCIPAL: float
    DIAS_AUSENTE: int
    NIVEL_EDUCATIVO: int
    TAMANO_EMPRESA: int
    ETNIA: str
    TIPO_TRABAJADOR: str
    REGISTRO_SUNAT: str
    LIBROS_CONTABLES: str
    FRECUENCIA_PAGO: str
    NIVEL_OCUPACION: str  # Mantenido como en el notebook
    
# Modelo de respuesta simplificado
class PredictionResponse(BaseModel):
    cluster: int
    perfil: str

# Mapeo de perfiles
PERFILES = {
    0: "Adultos Mayores con Baja Educación y Alta Formalidad en Salud",
    1: "Trabajadores Informales de Baja Calificación y Menores Ingresos",
    2: "Profesionales y Empleados Formales de Altos Ingresos y Estables",
    3: "Jóvenes Calificados en Empleos con Riesgo de Subempleo/Informalidad"
}

# Modificar el endpoint para requerir autenticación
@app.post("/predecir", 
          response_model=PredictionResponse, 
          dependencies=[Depends(verify_api_key)])
async def predecir_perfil(data: PredictionRequest):
    # Convertir a DataFrame
    input_data = data.dict()
    input_df = pd.DataFrame([input_data])
    
    # Predecir
    cluster = model.predict(input_df)[0]
    perfil = PERFILES.get(int(cluster), "Perfil no definido")
    
    return {"cluster": int(cluster), "perfil": perfil}

#uvicorn app:app --reload --port 8000