# Práctica RNA - Ejemplo 2: Fitting (regresión)
# Calcular la resistencia del concreto (en MPa) según su mezcla y los días que tiene.
# Dataset: Concrete Compressive Strength (UCI)
#   https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
# Hecho con apoyo de IA (Claude). El prompt está en la página de la práctica.
#
# Para correrlo:
#   pip install numpy pandas scikit-learn matplotlib
#   python 2_regresion_concreto.py

import os
import warnings
warnings.filterwarnings("ignore")  # para que no salgan avisos de las librerías
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RUTA_DATOS = "data/concreto.csv"
URL_DATOS = ("https://raw.githubusercontent.com/stedy/"
             "Machine-Learning-with-R-datasets/master/concrete.csv")

# Leer los datos (si no está el archivo se baja de internet)
# Entradas: cemento, escoria, ceniza, agua, superplastificante,
# grava, arena y edad en días. Salida: resistencia en MPa
df = pd.read_csv(RUTA_DATOS if os.path.exists(RUTA_DATOS) else URL_DATOS)
print(f"Mezclas: {len(df)}  |  Resistencia: {df['strength'].min():.1f} a "
      f"{df['strength'].max():.1f} MPa")

# El concreto se endurece rápido los primeros días y luego casi no cambia,
# por eso a la edad le saco logaritmo (con esto mejoró el resultado)
df["age"] = np.log1p(df["age"])
y = df.pop("strength").values
X = df.values

# 80% para entrenar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Normalizar los datos para que todos queden en la misma escala
escalador = StandardScaler().fit(X_train)
X_train = escalador.transform(X_train)
X_test = escalador.transform(X_test)

# Crear la red neuronal
# Probé varias combinaciones de capas y alpha, esta fue la que dio mejor resultado
red = MLPRegressor(
    hidden_layer_sizes=(128, 64),   # dos capas ocultas
    activation="relu",
    solver="adam",
    alpha=0.1,
    learning_rate_init=0.003,
    max_iter=3000,
    random_state=42,
)
red.fit(X_train, y_train)
print(f"Épocas entrenadas: {red.n_iter_}")

# Probar la red con los datos que no vio
pred = red.predict(X_test)
print("\n=== RESULTADOS (conjunto de prueba, 20%) ===")
print(f"R²:    {r2_score(y_test, pred):.3f}")
print(f"MAE:   {mean_absolute_error(y_test, pred):.2f} MPa")
print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, pred)):.2f} MPa")
print(f"R² en entrenamiento: {r2_score(y_train, red.predict(X_train)):.3f}")

# Algunos ejemplos para comparar
print("\nEjemplos (real vs. predicho, MPa):")
for real, p in list(zip(y_test, pred))[:8]:
    print(f"  {real:6.1f}  ->  {p:6.1f}")

# Gráficas: cómo bajó el error y valores reales contra predichos
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(red.loss_curve_, color="#2f5d8a")
ax[0].set_yscale("log")
ax[0].set_title("Curva de pérdida (entrenamiento)")
ax[0].set_xlabel("Época"); ax[0].set_ylabel("Error cuadrático / 2"); ax[0].grid(alpha=.3)

# Si los puntos quedan cerca de la línea punteada, la red predice bien
ax[1].scatter(y_test, pred, s=14, alpha=.7, color="#2f5d8a")
lim = [0, max(y_test.max(), pred.max()) + 5]
ax[1].plot(lim, lim, "--", color="#999")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("Resistencia real (MPa)"); ax[1].set_ylabel("Resistencia predicha (MPa)")
ax[1].set_title(f"Real vs. predicho  (R² = {r2_score(y_test, pred):.3f})"); ax[1].grid(alpha=.3)
fig.tight_layout()
os.makedirs("resultados", exist_ok=True)
fig.savefig("resultados/2_regresion.png", dpi=110)
print("\nGráfica guardada en resultados/2_regresion.png")
