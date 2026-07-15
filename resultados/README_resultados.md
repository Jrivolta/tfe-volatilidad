# Resultados del pipeline

Esta carpeta contiene las salidas que genera el script principal, con un juego de
ficheros por cada horizonte de clasificacion.

- `H0`  -> horizonte inmediato: regimen del dia actual.
- `H21` -> horizonte de prediccion: regimen dentro de 21 dias bursatiles.

## Ficheros

- `resultados_metricas_H0.csv` / `resultados_metricas_H21.csv` — tabla larga con
  una fila por combinacion de activo y modelo y una columna por metrica
  (exactitud, precision, exhaustividad y F1-macro).
- `resultados_completos_H0.json` / `resultados_completos_H21.json` — informacion
  detallada: hiperparametros optimos, matrices de confusion, importancia de
  atributos por permutacion y promedios por modelo.

Estos ficheros se obtienen ejecutando el script principal con el parametro
`HORIZONTE` fijado en 0 y en 21, respectivamente. Alimentan las tablas y figuras
del documento y el tablero de Power BI.

