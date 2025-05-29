#import joblib
import tensorflow as tf
import numpy as np

# Datos fijos para prueba/deben coincidir con el modelo_diabetes.keras
features = [[2.000000, 99.000000, 72.405184, 29.153420, 155.548223, 22.200000, 0.108000, 23.000000]]
# Cargar el modelo
modelo = joblib.load("modelo_diabetes.keras")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])
