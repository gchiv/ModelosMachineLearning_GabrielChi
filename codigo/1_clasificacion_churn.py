# Práctica RNA - Ejemplo 1: Pattern Recognition (clasificación)
# Predecir qué clientes de una compañía telefónica van a cancelar su servicio.
# Dataset: IBM Telco Customer Churn
#   https://github.com/IBM/telco-customer-churn-on-icp4d
# Hecho con apoyo de IA (Claude). El prompt está en la página de la práctica.
#
# Para correrlo:
#   pip install numpy pandas scikit-learn matplotlib
#   python 1_clasificacion_churn.py

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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

RUTA_DATOS = "data/telco_churn.csv"
URL_DATOS = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
             "master/data/Telco-Customer-Churn.csv")

# Leer los datos (si no está el archivo se baja de internet)
df = pd.read_csv(RUTA_DATOS if os.path.exists(RUTA_DATOS) else URL_DATOS)
print(f"Registros: {len(df)}  |  Columnas: {df.shape[1]}")

# Quitar el id del cliente porque no sirve para predecir
df = df.drop(columns="customerID")
# TotalCharges viene como texto y tiene algunos vacíos, se pasan a número
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Salida: 1 si el cliente se va, 0 si se queda
y = (df.pop("Churn") == "Yes").astype(int)
# Las columnas de texto (Yes/No, tipo de contrato, etc.) se pasan a 0 y 1
X = pd.get_dummies(df, drop_first=True).astype(float)
print(f"Entradas de la red: {X.shape[1]}  |  Clientes que se van: {y.mean():.1%}")

# 80% para entrenar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# Normalizar los datos para que todos queden en la misma escala
escalador = StandardScaler().fit(X_train)
X_train = escalador.transform(X_train)
X_test = escalador.transform(X_test)

# Crear la red neuronal
# Probé varias combinaciones de capas y alpha, esta fue la que dio mejor resultado
red = MLPClassifier(
    hidden_layer_sizes=(16,),   # una capa oculta con 16 neuronas
    activation="relu",
    solver="adam",
    alpha=0.1,                  # ayuda a que no se aprenda los datos de memoria
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,        # se detiene cuando ya no mejora
    validation_fraction=0.1,
    random_state=42,
)
red.fit(X_train, y_train)
print(f"Épocas entrenadas: {red.n_iter_}")

# Probar la red con los datos que no vio
prob = red.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("\n=== RESULTADOS (conjunto de prueba, 20%) ===")
print(f"Exactitud (accuracy): {accuracy_score(y_test, pred):.3f}")
print(f"Precisión:            {precision_score(y_test, pred):.3f}")
print(f"Sensibilidad (recall):{recall_score(y_test, pred):.3f}")
print(f"F1:                   {f1_score(y_test, pred):.3f}")
print(f"AUC ROC:              {roc_auc_score(y_test, prob):.3f}")

# Matriz de confusión: cuántos acertó y cuántos falló
cm = confusion_matrix(y_test, pred)
print("\nMatriz de confusión [real x predicho]:")
print(f"              Se queda  Se va")
print(f"  Se queda    {cm[0,0]:8d}  {cm[0,1]:5d}")
print(f"  Se va       {cm[1,0]:8d}  {cm[1,1]:5d}")

# Gráficas: cómo bajó el error y la matriz de confusión
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(red.loss_curve_, color="#2f5d8a")
ax[0].set_title("Curva de pérdida (entrenamiento)")
ax[0].set_xlabel("Época"); ax[0].set_ylabel("Log-loss"); ax[0].grid(alpha=.3)

ax[1].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, cm[i, j], ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
ax[1].set_xticks([0, 1], ["Se queda", "Se va"]); ax[1].set_yticks([0, 1], ["Se queda", "Se va"])
ax[1].set_xlabel("Predicho"); ax[1].set_ylabel("Real"); ax[1].set_title("Matriz de confusión")
fig.tight_layout()
os.makedirs("resultados", exist_ok=True)
fig.savefig("resultados/1_clasificacion.png", dpi=110)
print("\nGráfica guardada en resultados/1_clasificacion.png")
