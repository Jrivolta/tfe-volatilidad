# Comparativa de Técnicas de Aprendizaje Supervisado para la Clasificación de Regímenes de Volatilidad

Trabajo Fin de Estudios (TFE) del Máster Universitario en Análisis y Visualización de Datos Masivos (UNIR).

Este repositorio contiene el código que implementa la comparativa de cinco clasificadores supervisados (Regresión Logística, Árbol de Decisión, Random Forest, XGBoost y SVM), frente a una referencia de persistencia, para la clasificación de regímenes de volatilidad (bajo, medio y alto) sobre cuatro activos del NASDAQ (AAPL, AMZN, GOOGL y TSLA) en el periodo 2015-2024. La comparación se realiza en dos horizontes complementarios: el reconocimiento del régimen actual y la predicción a 21 días.

## Estructura del proyecto

```
tfe-volatilidad/
├── codigo/
│     └── tfe_clasificacion_volatilidad.py   (script principal del pipeline)
├── data/
│     └── README_data.md                     (los precios se descargan via yfinance)
├── resultados/
│     ├── resultados_metricas_H0.csv / _H21.csv     (metricas por activo, modelo y horizonte)
│     └── resultados_completos_H0.json / _H21.json  (hiperparametros, matrices, importancias)
├── powerbi/
│     ├── tablero_volatilidad.pbix           (tablero de control interactivo)
│     ├── generar_powerbi.py                 (genera los CSV que alimentan el tablero)
│     └── pbi_*.csv                          (datos del tablero)
├── figuras/
│     └── *.png                              (figuras generadas por el pipeline)
├── requirements.txt
└── README.md
```

## Instalación

```
git clone https://github.com/Jrivolta/tfe-volatilidad.git
cd tfe-volatilidad
pip install -r requirements.txt
```

## Ejecución

El experimento se reproduce con un único comando. El parámetro `HORIZONTE` del script fija la tarea: `0` para el reconocimiento del régimen actual y `21` para la predicción a 21 días.

```
python codigo/tfe_clasificacion_volatilidad.py
```

Al finalizar, el script genera en `resultados/` los ficheros `resultados_metricas_H{0,21}.csv` y `resultados_completos_H{0,21}.json`, y las figuras del análisis. Para obtener ambos horizontes se ejecuta dos veces, cambiando el valor de `HORIZONTE`. Estos resultados alimentan el tablero de Power BI (`powerbi/tablero_volatilidad.pbix`).

## Resultado principal

En el horizonte inmediato, ningún modelo supera a la referencia de persistencia (F1-macro de 0.900 frente al 0.892 del mejor modelo entrenado, Random Forest). En el horizonte de predicción a 21 días, el rendimiento de todos los modelos se aproxima al de una clasificación aleatoria. La aportación es metodológica: sin una referencia de persistencia y sin distinguir entre reconocer el régimen actual y predecirlo, el rendimiento aparente de los modelos se sobreestima.

## Reproducibilidad

Todas las fuentes de aleatoriedad se fijan con una semilla común (42). Los terciles que definen los regímenes se estiman únicamente con el tramo de entrenamiento, y el ajuste de hiperparámetros emplea validación cruzada temporal (TimeSeriesSplit), que respeta el orden cronológico de los datos. Los datos son públicos y gratuitos (descargados mediante `yfinance`), y el código completo se publica en abierto para permitir la auditoría de cada decisión metodológica.

## Autor

Jean Raúl Rivolta Baptista — TFE dirigido por el Prof. Deivis Ramírez (UNIR).
