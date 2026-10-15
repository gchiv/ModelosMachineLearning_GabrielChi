# Práctica RNA - Ejemplo 3: Series de tiempo
# Pronosticar cuántos pasajeros va a tener una aerolínea cada mes.
# Dataset: International Airline Passengers 1949-1960 (miles de pasajeros por mes)
#   https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv
# Hecho con apoyo de IA (Claude). El prompt está en la página de la práctica.
#
# Para correrlo:
#   pip install numpy pandas scikit-learn matplotlib
#   python 3_series_pasajeros.py

import os
import warnings
warnings.filterwarnings("ignore")  # para que no salgan avisos de las librerías
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

RUTA_DATOS = "data/pasajeros_aereos.csv"
URL_DATOS = ("https://raw.githubusercontent.com/jbrownlee/Datasets/"
             "master/airline-passengers.csv")

VENTANA = 15      # cuántos meses anteriores ve la red para predecir el siguiente
N_PRUEBA = 24     # los últimos 2 años se guardan para probar
N_FUTURO = 12     # cuántos meses se pronostican al final

# Leer los datos (si no está el archivo se baja de internet)
df = pd.read_csv(RUTA_DATOS if os.path.exists(RUTA_DATOS) else URL_DATOS)
fechas = pd.to_datetime(df["Month"])
serie = df["Passengers"].values.astype(float)
print(f"Meses: {len(serie)}  ({df['Month'].iloc[0]} a {df['Month'].iloc[-1]})")

# Uso el logaritmo de los pasajeros y comparo cada mes contra el último,
# así la red aprende la forma de la serie (sube cada año y tiene picos en verano)
L = np.log(serie)

def ventana(t):
    # Los meses anteriores al mes t, comparados contra el último mes
    return (L[t - VENTANA:t] - L[t - 1]) * 10

# Armar las entradas (meses anteriores) y la salida (mes siguiente)
X = np.array([ventana(t) for t in range(VENTANA, len(L))])
Y = np.array([(L[t] - L[t - 1]) * 10 for t in range(VENTANA, len(L))])
X_train, Y_train = X[:-N_PRUEBA], Y[:-N_PRUEBA]
X_test = X[-N_PRUEBA:]

# Crear la red neuronal
# Probé varias ventanas, capas y alpha, esta fue la que dio mejor resultado
red = MLPRegressor(
    hidden_layer_sizes=(32, 16),   # dos capas ocultas
    activation="relu",
    solver="lbfgs",                # funcionó mejor que adam porque hay pocos datos
    alpha=0.01,
    max_iter=5000,
    random_state=0,
)
red.fit(X_train, Y_train)

# Probar la red en 1959 y 1960 (mes por mes)
pred_test = np.exp(L[-N_PRUEBA - 1:-1] + red.predict(X_test) / 10)
real_test = serie[-N_PRUEBA:]
mape = np.mean(np.abs(pred_test - real_test) / real_test) * 100
mae = np.mean(np.abs(pred_test - real_test))
rmse = np.sqrt(np.mean((pred_test - real_test) ** 2))

print("\n=== RESULTADOS (prueba: últimos 24 meses) ===")
print(f"MAPE:  {mape:.2f} %")
print(f"MAE:   {mae:.1f} miles de pasajeros")
print(f"RMSE:  {rmse:.1f} miles de pasajeros")
print("\nMes       Real   Predicho")
for f, r, p in zip(df["Month"].iloc[-N_PRUEBA:], real_test, pred_test):
    print(f"{f}   {r:5.0f}   {p:7.1f}")

# Pronóstico de 1961: se entrena con todos los datos y cada mes
# que predice se usa para predecir el siguiente
red.fit(X, Y)
L_ext = list(L)
for _ in range(N_FUTURO):
    h = np.array(L_ext[-VENTANA:])
    L_ext.append(L_ext[-1] + red.predict(((h - h[-1]) * 10).reshape(1, -1))[0] / 10)
futuro = np.exp(L_ext[-N_FUTURO:])
fechas_fut = pd.date_range(fechas.iloc[-1] + pd.offsets.MonthBegin(1), periods=N_FUTURO, freq="MS")
print("\nPronóstico 1961 (miles de pasajeros):")
print("  " + "  ".join(f"{f:%b}:{v:.0f}" for f, v in zip(fechas_fut, futuro)))

# Gráfica: datos reales, lo que predijo en la prueba y el pronóstico
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(fechas, serie, color="#444", lw=1.4, label="Real")
ax.plot(fechas.iloc[-N_PRUEBA:], pred_test, color="#2f5d8a", lw=2,
        label=f"Predicción en prueba (MAPE {mape:.1f}%)")
ax.plot(fechas_fut, futuro, "--", color="#c0703a", lw=2, label="Pronóstico 1961")
ax.axvline(fechas.iloc[-N_PRUEBA], color="#bbb", ls=":")
ax.set_ylabel("Miles de pasajeros"); ax.set_title("Pasajeros aéreos mensuales")
ax.legend(loc="upper left"); ax.grid(alpha=.3)
fig.tight_layout()
os.makedirs("resultados", exist_ok=True)
fig.savefig("resultados/3_series.png", dpi=110)
print("\nGráfica guardada en resultados/3_series.png")
