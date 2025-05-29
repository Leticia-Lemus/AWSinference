import joblib
#import tensorflow as tf
#import numpy as np

# Datos fijos para prueba/deben coincidir con el modelo_diabetes.keras
features = [[1, 45, 58, 11, 22, 0]]
# Cargar el modelo
modelo = joblib.load("pinguinos_mlp_model.pkl")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])
