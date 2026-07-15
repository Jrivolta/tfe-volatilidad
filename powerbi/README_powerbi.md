# Tablero de control (Power BI)

Esta carpeta contiene el tablero interactivo del trabajo y los ficheros que lo
alimentan.

## Contenido

- `tablero_volatilidad.pbix` — tablero interactivo de Power BI. Incluye los datos
  ya cargados, por lo que se puede abrir y explorar directamente. Tiene un
  segmentador de **Horizonte** que permite alternar entre el reconocimiento del
  regimen actual y la prediccion a 21 dias.
- `generar_powerbi.py` — script que genera los seis ficheros CSV que alimentan el
  tablero a partir de las salidas del pipeline (`resultados_completos_H0.json` y
  `resultados_completos_H21.json`).
- `pbi_metricas.csv` — metricas por activo, modelo y horizonte.
- `pbi_matrices_confusion.csv` — matrices de confusion en formato largo.
- `pbi_importancias.csv` — importancia de atributos por permutacion.
- `pbi_recall_regimen.csv` — exhaustividad (recall) por regimen.
- `pbi_distribucion_clases.csv` — distribucion de regimenes en el conjunto de prueba.
- `pbi_estado_actual.csv` — volatilidad y regimen mas reciente de cada activo.

## Como regenerar los datos

Con los ficheros de resultados del pipeline en la misma carpeta y conexion a
internet (para el estado actual):

```
pip install pandas numpy yfinance
python generar_powerbi.py
```

Se generan los seis CSV. Para actualizar el tablero, basta con colocarlos en la
ruta que lee Power BI, abrir el `.pbix` y pulsar *Actualizar*.
