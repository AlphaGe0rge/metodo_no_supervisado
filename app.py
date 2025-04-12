# ========================================
# AGRUPAMIENTO DE ESTACIONES DEL METRO
# Basado en datos simulados de transporte masivo
# Aplicación de K-means (Aprendizaje no supervisado)
# ========================================

# Importamos las librerías necesarias
import pandas as pd               # Para manipulación de datos
from sklearn.cluster import KMeans  # Algoritmo de agrupamiento
import matplotlib.pyplot as plt   # Para graficar

# ========================================
# 1. Simulación de Dataset
# ========================================

# Creamos un diccionario con datos simulados de estaciones de metro
data = {
    "Estacion": [
        "Portal Norte", "Calle 187", "Calle 170", "Calle 142", "Heroes",
        "Calle 100", "Calle 72", "Av. Jimenez", "Portal Sur", "Venecia", "Restrepo"
    ],
    "Latitud": [4.754, 4.739, 4.713, 4.689, 4.661, 4.651, 4.647, 4.601, 4.572, 4.566, 4.579],
    "Longitud": [-74.031, -74.032, -74.034, -74.037, -74.060, -74.067, -74.072, -74.083, -74.100, -74.106, -74.099],
    "Pasajeros_dia": [45000, 38000, 42000, 41000, 60000, 55000, 48000, 70000, 50000, 30000, 35000],
    "Tiempo_espera_min": [4, 5, 4, 6, 3, 3, 5, 2, 4, 6, 5]
}

# Convertimos el diccionario en un DataFrame de pandas
df = pd.DataFrame(data)

# ========================================
# 2. Aplicación del modelo de Agrupamiento (K-means)
# ========================================

# Seleccionamos las columnas numéricas que usará el modelo para agrupar
X = df[["Latitud", "Longitud", "Pasajeros_dia", "Tiempo_espera_min"]]

# Creamos una instancia del modelo K-means con 3 grupos (clusters)
kmeans = KMeans(n_clusters=3, random_state=42)

# Entrenamos el modelo y obtenemos los grupos a los que pertenece cada estación
df["Cluster"] = kmeans.fit_predict(X)

# Mostramos los resultados por estación
print("Resultados del agrupamiento por estación:")
print(df[["Estacion", "Cluster"]])
print()