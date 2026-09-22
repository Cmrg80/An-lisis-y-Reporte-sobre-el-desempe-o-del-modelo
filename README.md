# Análisis de Evaluación de un Modelo Random Forest

## Descripción

Este proyecto presenta la implementación, evaluación y ajuste de un
modelo de **Random Forest para clasificación**. El objetivo es analizar
el desempeño del modelo utilizando una separación de los datos en
conjuntos de **entrenamiento, validación y prueba (Train / Validation /
Test)**, así como identificar posibles problemas de sesgo, varianza y
sobreajuste.

El modelo se aplica a un dataset de satisfacción de clientes de una
aerolínea, utilizando la variable `satisfaction` como variable objetivo.

------------------------------------------------------------------------

## Objetivos

-   Preparar y limpiar los datos antes del entrenamiento.
-   Separar los datos en conjuntos de entrenamiento, validación y
    prueba.
-   Entrenar un modelo Random Forest.
-   Evaluar el modelo mediante métricas de clasificación.
-   Analizar las curvas de entrenamiento y validación.
-   Diagnosticar el grado de **bias**, **variance** y **overfitting**.
-   Ajustar hiperparámetros para intentar mejorar la generalización.
-   Comparar el desempeño del modelo antes y después del ajuste.
-   Documentar los resultados y las variables más importantes.

------------------------------------------------------------------------

## Dataset

El dataset contiene información relacionada con la experiencia de
pasajeros de una aerolínea.

La variable objetivo es:

``` text
satisfaction
```

La variable se transforma a una representación binaria:

``` text
False → 0
True  → 1
```

Entre las variables utilizadas se encuentran características
relacionadas con el servicio, viaje y perfil del pasajero, por ejemplo:

-   `Gender`
-   `Customer Type`
-   `Age`
-   `Type of Travel`
-   `Class`
-   `Flight Distance`
-   `Inflight wifi service`
-   `Ease of Online booking`
-   `Food and drink`
-   `Online boarding`
-   `Inflight entertainment`
-   `Checkin service`
-   `Arrival Delay in Minutes`

------------------------------------------------------------------------

## Preprocesamiento

Antes de entrenar el modelo se realizan las siguientes etapas:

1.  Separación de la variable objetivo `satisfaction`.
2.  División estratificada de los datos:
    -   **60% Training**
    -   **20% Validation**
    -   **20% Test**
3.  Imputación de valores faltantes utilizando la mediana calculada
    sobre el conjunto de entrenamiento.
4.  Conversión de variables categóricas mediante **One-Hot Encoding**.
5.  Alineación de las columnas de los conjuntos de validación y prueba
    con respecto al conjunto de entrenamiento.

La imputación se realiza después de separar los datos para evitar
utilizar información del conjunto de prueba durante el entrenamiento.

------------------------------------------------------------------------

## Modelo Random Forest

### Configuración inicial

La configuración utilizada inicialmente fue:

``` python
RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

### Diagnóstico inicial

El análisis inicial mostró:

  Aspecto           Diagnóstico
  ----------------- -------------
  Bias / Sesgo      Bajo
  Varianza          Media
  Nivel de ajuste   Overfitting

El modelo alcanzó aproximadamente 100% de accuracy sobre entrenamiento,
mientras que la validación se mantuvo alrededor de 96%. Esta diferencia
indicó que el modelo estaba aprendiendo muy bien los datos de
entrenamiento, pero presentaba una pérdida de desempeño sobre datos no
utilizados directamente para entrenar.

------------------------------------------------------------------------

## Ajuste y regularización

Para intentar reducir la varianza y mejorar la generalización se
modificaron dos hiperparámetros:

  Hiperparámetro        Antes   Después
  ---------------- ---------- ---------
  `n_estimators`          100       150
  `max_features`     `"sqrt"`     `0.5`

El cambio de `max_features` busca aumentar la diversidad entre los
árboles al limitar la proporción de variables consideradas durante la
construcción de cada división.

El código mantiene la configuración anterior como comentario para
conservar el historial del cambio realizado.

------------------------------------------------------------------------

## Resultados

### Comparación antes y después del ajuste

  Métrica       Modelo original   Modelo ajustado    Cambio
  ----------- ----------------- ----------------- ---------
  Accuracy               0.9586        **0.9593**   +0.0007
  Precision              0.9683            0.9674   -0.0009
  Recall                 0.9353        **0.9379**   +0.0026
  F1-Score               0.9515        **0.9524**   +0.0009

El modelo ajustado obtuvo una accuracy de **95.93%** sobre el conjunto
de prueba.

El Recall aumentó de **93.53% a 93.79%**, mientras que el F1-Score
aumentó de **95.15% a 95.24%**.

La mejora general fue pequeña, por lo que el ajuste puede considerarse
una **mejora marginal** del desempeño.

------------------------------------------------------------------------

## Matriz de confusión

La matriz de confusión obtenida con el modelo ajustado fue:

``` text
[[14333   357]
 [  701 10585]]
```

Interpretación:

-   **14,333** casos insatisfechos fueron clasificados correctamente.
-   **357** casos insatisfechos fueron clasificados como satisfechos.
-   **701** casos satisfechos fueron clasificados como insatisfechos.
-   **10,585** casos satisfechos fueron clasificados correctamente.

------------------------------------------------------------------------

## Curvas de entrenamiento y validación

El proyecto genera dos gráficas principales:

### Training vs Validation Accuracy

Permite observar cómo cambia la precisión del modelo al incrementar el
número de árboles.

La curva muestra que el entrenamiento alcanza prácticamente **1.00**,
mientras que la validación se estabiliza aproximadamente alrededor de
**0.96**.

### Training vs Validation Loss

Permite analizar el comportamiento del `Log Loss`.

El `training loss` se mantiene aproximadamente en **0.028**, mientras
que el `validation loss` se mantiene alrededor de **0.105**.

La diferencia entre ambas curvas proporciona evidencia de que todavía
existe **overfitting**.

------------------------------------------------------------------------

## Diagnóstico final

Después del ajuste, el diagnóstico se mantiene de la siguiente manera:

  Aspecto           Diagnóstico
  ----------------- --------------------------
  Bias / Sesgo      **Bajo**
  Varianza          **Media**
  Nivel de ajuste   **Overfitting moderado**

Aunque el ajuste produjo una mejora ligera en las métricas de prueba, la
separación entre las curvas de entrenamiento y validación permanece.

Por lo tanto, el modelo conserva un buen desempeño de generalización,
pero todavía presenta señales de sobreajuste.

------------------------------------------------------------------------

## Variables más importantes

Las diez variables con mayor importancia obtenidas por el Random Forest
fueron:

  Variable                             Importancia
  ---------------------------------- -------------
  `Online boarding`                       0.257529
  `Inflight wifi service`                 0.166772
  `Class_Business`                        0.086676
  `Type of Travel_Personal Travel`        0.057557
  `Type of Travel_Business travel`        0.052318
  `Inflight entertainment`                0.052210
  `Age`                                   0.027381
  `Ease of Online booking`                0.026457
  `Checkin service`                       0.026428
  `Flight Distance`                       0.026344

Estas importancias representan la contribución de cada variable a las
decisiones del Random Forest. No deben interpretarse como relaciones
causales.

------------------------------------------------------------------------

## Estructura del proyecto

Una estructura recomendada para el repositorio es:

``` text
Análisis-de-evaluación-de-modelo/
│
├── random_forest_airline.py
├── data/
│   └── data.csv
│
├── results/
│   ├── training_validation_accuracy.png
│   ├── training_validation_log_loss.png
│   ├── matriz_confusion.png
│   ├── feature_importance.csv
│   └── resultados.txt
│
├── reporte/
│   └── reporte_random_forest.docx
│
└── README.md
```

> El dataset puede mantenerse fuera del repositorio si contiene
> restricciones de distribución o si se desea evitar almacenar archivos
> grandes en GitHub.

------------------------------------------------------------------------

## Requisitos

Se requiere Python 3 y las siguientes bibliotecas:

``` bash
pip install pandas matplotlib scikit-learn
```

`tkinter` también es utilizado para seleccionar el archivo CSV mediante
una ventana gráfica. En muchas instalaciones de Python para Windows
viene incluido.

------------------------------------------------------------------------

## Ejecución

Clona el repositorio:

``` bash
git clone https://github.com/Cmrg80/Portafolio-de-implementacion.git
```

Entra a la carpeta correspondiente:

``` bash
cd "Análisis de evaluación de modelo"
```

Ejecuta el programa:

``` bash
python random_forest_airline.py
```

El programa abrirá una ventana para seleccionar el archivo CSV.

Después del procesamiento y entrenamiento, los resultados se guardarán
en la carpeta:

``` text
results/
```

------------------------------------------------------------------------

## Archivos generados

El programa genera:

-   `training_validation_accuracy.png` --- comparación de accuracy entre
    entrenamiento y validación.
-   `training_validation_log_loss.png` --- comparación de Log Loss entre
    entrenamiento y validación.
-   `matriz_confusion.png` --- matriz de confusión.
-   `feature_importance.csv` --- importancia de las variables.
-   `resultados.txt` --- métricas y resultados principales.

------------------------------------------------------------------------

## Conclusión

El modelo Random Forest obtuvo un desempeño de **95.93% de accuracy**
después del ajuste. El cambio de hiperparámetros produjo una mejora
ligera respecto al modelo original, particularmente en Recall y
F1-Score.

El análisis de las curvas de entrenamiento y validación muestra que el
modelo continúa presentando **overfitting moderado**, debido a que el
desempeño sobre entrenamiento es prácticamente perfecto mientras que el
desempeño sobre validación es menor.

Como trabajo futuro se pueden explorar otros mecanismos de
regularización, como limitar `max_depth`, aumentar `min_samples_leaf` o
realizar una búsqueda sistemática de hiperparámetros para analizar si es
posible reducir la brecha entre entrenamiento y validación.

------------------------------------------------------------------------

## Autor

**Carlos Manuel Ramos Gastélum**

Proyecto académico de análisis y evaluación de modelos de Machine
Learning.
