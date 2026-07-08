# Comparativa de Técnicas de Aprendizaje Supervisado para la Clasificación de Regímenes de Volatilidad

Trabajo Fin de Estudios (TFE) del Máster Universitario en Análisis y Visualización de Datos Masivos (UNIR).

Este repositorio contiene el código que implementa la comparativa de cinco clasificadores supervisados (Regresión Logística, Árbol de Decisión, Random Forest, XGBoost y SVM) para la clasificación de regímenes de volatilidad (bajo, medio y alto) sobre cuatro activos del NASDAQ (AAPL, AMZN, GOOGL y TSLA) en el periodo 2015-2025.

## Estructura del proyecto

```
tfe-volatilidad/
├── codigo/
│   └── tfe_clasificacion_volatilidad.py   (script principal del pipeline)
├── data/
│   └── README_data.md                     (los precios se descargan via yfinance)
├── resultados/
│   ├── resultados_metricas.csv            (metricas de los 20 modelos)
│   └── resultados_completos.json          (hiperparametros, matrices, importancias)
├── powerbi/
│   └── tablero_volatilidad.pbix           (tablero de control interactivo)
├── figuras/
│   └── *.png                              (figuras generadas por el pipeline)
├── requirements.txt
└── README.md
```

## Instalación

```bash
git clone https://github.com/<usuario>/tfe-volatilidad.git
cd tfe-volatilidad
pip install -r requirements.txt
```

## Ejecución

Un único comando reproduce todo el experimento de extremo a extremo (descarga de datos, validación de completitud, ingeniería de atributos, entrenamiento y evaluación de los cinco modelos para cada activo):

```bash
python codigo/tfe_clasificacion_volatilidad.py
```

Al finalizar, el script genera en `resultados/` los ficheros `resultados_metricas.csv` y `resultados_completos.json`, y en `figuras/` las imágenes del análisis. Estos mismos resultados alimentan el tablero de Power BI (`powerbi/tablero_volatilidad.pbix`).

## Reproducibilidad

Todas las fuentes de aleatoriedad se fijan con una semilla común, de modo que cada ejecución produce resultados idénticos. Los datos son públicos y gratuitos (descargados mediante `yfinance`), y el código completo se publica en abierto para permitir la auditoría de cada decisión metodológica.

## Autor

Jean Raúl Rivolta Baptista — TFE dirigido por el Prof. Deivis Ramírez (UNIR).
