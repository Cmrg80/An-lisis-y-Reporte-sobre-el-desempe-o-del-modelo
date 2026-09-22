import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    log_loss
)

# ============================================================
# CARGA DEL DATASET
# ============================================================

def seleccionar_dataset():
    root = tk.Tk()
    root.withdraw()

    archivo = filedialog.askopenfilename(
        title="Selecciona el dataset",
        filetypes=[
            ("Archivos CSV", "*.csv"),
            ("Todos los archivos", "*.*")
        ]
    )

    root.destroy()
    return archivo


print("Selecciona el archivo CSV del dataset...")
DATASET_PATH = seleccionar_dataset()

if not DATASET_PATH:
    print("No se seleccionó ningún archivo.")
    raise SystemExit

print("\nDataset seleccionado:")
print(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

print("\nDataset cargado correctamente.")
print(f"Número de registros: {len(df)}")
print(f"Número de variables: {len(df.columns)}")

# ============================================================
# PREPROCESAMIENTO
# ============================================================

# La variable objetivo es satisfaction.
# El dataset la contiene como booleana:
# False = cliente insatisfecho
# True  = cliente satisfecho

y = df["satisfaction"].astype(int)

# Eliminamos la variable objetivo de las características.
X = df.drop(columns=["satisfaction"]).copy()

# -----------------------------
# Valores faltantes
# -----------------------------
# Arrival Delay in Minutes contiene valores faltantes.
# Se imputa con la mediana calculada sobre el conjunto de entrenamiento
# para evitar utilizar información del conjunto de prueba.

# -----------------------------
# Variables categóricas
# -----------------------------
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

# Separamos primero para poder realizar la imputación después del split.
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Copias para evitar modificaciones accidentales.
X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

# Imputación de variables numéricas usando SOLO la mediana de train.
numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()

train_medians = X_train[numeric_columns].median()

X_train[numeric_columns] = X_train[numeric_columns].fillna(train_medians)
X_val[numeric_columns] = X_val[numeric_columns].fillna(train_medians)
X_test[numeric_columns] = X_test[numeric_columns].fillna(train_medians)

# One-hot encoding de variables categóricas.
# Se concatena después del split y se alinean las columnas para garantizar
# exactamente las mismas características en train, validation y test.

X_train = pd.get_dummies(X_train, columns=categorical_columns, dtype=int)
X_val = pd.get_dummies(X_val, columns=categorical_columns, dtype=int)
X_test = pd.get_dummies(X_test, columns=categorical_columns, dtype=int)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nPreprocesamiento terminado.")
print(f"Variables originales: {len(df.columns) - 1}")
print(f"Variables después de One-Hot Encoding: {len(X_train.columns)}")
print(f"Valores faltantes en train: {X_train.isna().sum().sum()}")
print(f"Valores faltantes en validation: {X_val.isna().sum().sum()}")
print(f"Valores faltantes en test: {X_test.isna().sum().sum()}")

print("\nDivisión del dataset:")
print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de validación: {len(X_val)}")
print(f"Datos de prueba: {len(X_test)}")

# ============================================================
# RANDOM FOREST
# ============================================================

# ============================================================
# AJUSTE / REGULARIZACIÓN DEL RANDOM FOREST
# ============================================================
# CONFIGURACIÓN ANTERIOR (se conserva como referencia):
# model = RandomForestClassifier(
#     n_estimators=100,
#     criterion="gini",
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features="sqrt",
#     bootstrap=True,
#     random_state=42,
#     n_jobs=-1
# )

# CONFIGURACIÓN AJUSTADA:
# - n_estimators: 100 -> 150. Más árboles estabilizan el ensamble.
# - max_features: "sqrt" -> 0.5. Cada árbol usa como máximo el 50%
#   de las variables disponibles, reduciendo la correlación entre árboles
#   y ayudando a controlar la varianza.
# - Los demás parámetros se mantienen iguales para aislar el efecto
#   principal del ajuste.
model = RandomForestClassifier(
    n_estimators=150,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.5,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

print("\nEntrenando Random Forest...")
model.fit(X_train, y_train)
print("Entrenamiento terminado.")

# ============================================================
# PREDICCIONES
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# CARPETA DE RESULTADOS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# TRAINING VS VALIDATION ACCURACY
# ============================================================

print("\nGenerando gráfica de entrenamiento y validación...")

n_estimators_values = range(10, 201, 10)

training_accuracy = []
validation_accuracy = []

for n in n_estimators_values:
    # Configuración ajustada aplicada a cada valor de n_estimators.
    # ANTES: max_features="sqrt"
    # AHORA: max_features=0.5
    rf = RandomForestClassifier(
        n_estimators=n,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    train_prediction = rf.predict(X_train)
    validation_prediction = rf.predict(X_val)

    training_accuracy.append(
        accuracy_score(y_train, train_prediction)
    )

    validation_accuracy.append(
        accuracy_score(y_val, validation_prediction)
    )

plt.figure(figsize=(10, 6))
plt.plot(
    n_estimators_values,
    training_accuracy,
    marker="o",
    label="Training Accuracy"
)
plt.plot(
    n_estimators_values,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "training_validation_accuracy.png"),
    dpi=900
)
plt.close()

# ============================================================
# TRAINING VS VALIDATION LOSS
# ============================================================

print("\nGenerando gráfica de loss...")

training_loss = []
validation_loss = []

for n in n_estimators_values:
    # Configuración ajustada aplicada a cada valor de n_estimators.
    # ANTES: max_features="sqrt"
    # AHORA: max_features=0.5
    rf = RandomForestClassifier(
        n_estimators=n,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    train_probabilities = rf.predict_proba(X_train)
    validation_probabilities = rf.predict_proba(X_val)

    training_loss.append(
        log_loss(y_train, train_probabilities)
    )

    validation_loss.append(
        log_loss(y_val, validation_probabilities)
    )

plt.figure(figsize=(10, 6))
plt.plot(
    n_estimators_values,
    training_loss,
    marker="o",
    label="Training Loss"
)
plt.plot(
    n_estimators_values,
    validation_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Log Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardamos la gráfica con un nombre ASCII para evitar problemas
# de compatibilidad de rutas/archivos en Windows + OneDrive.
loss_plot_path = os.path.join(
    RESULTS_DIR,
    "training_validation_log_loss.png"
)

try:
    plt.savefig(loss_plot_path, dpi=300)
    print(f"Gráfica guardada: {loss_plot_path}")
except OSError as e:
    print(f"Advertencia: no se pudo guardar la gráfica de loss: {e}")

plt.close()

# ============================================================
# MÉTRICAS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== RESULTADOS ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de confusión:")
print(cm)

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Insatisfecho", "Satisfecho"]
    )
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Insatisfecho", "Satisfecho"]
)

disp.plot()
plt.title("Matriz de Confusión - Random Forest")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "matriz_confusion.png"),
    dpi=900
)
plt.close()

# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== VARIABLES MÁS IMPORTANTES ==========")
print(feature_importance.head(10).to_string(index=False))

# Guardar importancia de variables.
feature_importance.to_csv(
    os.path.join(RESULTS_DIR, "feature_importance.csv"),
    index=False
)

# ============================================================
# GUARDAR RESULTADOS
# ============================================================

with open(
    os.path.join(RESULTS_DIR, "resultados.txt"),
    "w",
    encoding="utf-8"
) as file:

    file.write("RESULTADOS RANDOM FOREST\n")
    file.write("========================\n\n")

    file.write(f"Registros totales: {len(df)}\n")
    file.write(f"Variables originales: {len(df.columns) - 1}\n")
    file.write(f"Variables después del preprocesamiento: {len(X_train.columns)}\n")
    file.write(f"Datos de entrenamiento: {len(X_train)}\n")
    file.write(f"Datos de validación: {len(X_val)}\n")
    file.write(f"Datos de prueba: {len(X_test)}\n\n")

    file.write("Métricas:\n")
    file.write(f"Accuracy:  {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall:    {recall:.4f}\n")
    file.write(f"F1-Score:  {f1:.4f}\n\n")

    file.write("Matriz de confusión:\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("Variables más importantes:\n")
    file.write(feature_importance.head(10).to_string(index=False))

print("\nResultados guardados en la carpeta results.")
