"""
Genera los seis ficheros que alimentan el tablero de Power BI.

A partir de las salidas del pipeline (resultados_completos_H0.json y _H21.json)
produce los cinco ficheros de comparacion, cada uno con una columna 'Horizonte'
("Regimen actual" y "Regimen prediccion 21") que usa el segmentador del tablero.
Ademas descarga precios recientes con yfinance y genera el sexto fichero,
pbi_estado_actual.csv, con la volatilidad y el regimen mas reciente de cada
activo (panel de estado, independiente del horizonte).

Ficheros generados:
  - pbi_metricas.csv
  - pbi_matrices_confusion.csv
  - pbi_importancias.csv
  - pbi_recall_regimen.csv
  - pbi_distribucion_clases.csv
  - pbi_estado_actual.csv

Requisitos: pip install pandas numpy yfinance
Ejecutar (con los JSON en la misma carpeta): python generar_powerbi.py
"""

import json
import numpy as np
import pandas as pd

ACT = ["AAPL", "AMZN", "GOOGL", "TSLA"]
EMPRESA = {"AAPL": "Apple", "AMZN": "Amazon", "GOOGL": "Alphabet", "TSLA": "Tesla"}
MOD = ["Persistencia", "Regresion Logistica", "Arbol de Decision",
       "Random Forest", "XGBoost", "SVM"]
REG = ["Bajo", "Medio", "Alto"]
HOR = {0: "Regimen actual", 21: "Regimen prediccion 21"}
ATRIB = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4", "ret_lag5",
         "sma_5", "sma_20", "rsi_14", "vol_lag1", "vol_rel"]

VENTANA_VOL = 21
ANUALIZACION = np.sqrt(252)
INICIO, FIN = "2015-01-01", "2025-01-01"
ENC = "utf-8-sig"


def load(h):
    return json.load(open(f"resultados_completos_H{h}.json", encoding="utf-8"))


def generar_comparacion():
    """Genera los cinco ficheros de comparacion a partir de los JSON."""
    datos = {0: load(0), 21: load(21)}
    metr, conf, imp, rec_tmp, dist = [], [], [], [], []

    for h, d in datos.items():
        etq = HOR[h]
        for a in ACT:
            for m in MOD:
                r = d[a][m]
                metr.append([etq, a, EMPRESA[a], m, r["accuracy"],
                             r["precision_macro"], r["recall_macro"], r["f1_macro"]])
                cm = np.array(r["matriz_confusion"], dtype=int)
                for i, real in enumerate(REG):
                    for j, pred in enumerate(REG):
                        conf.append([etq, a, m, real, pred, int(cm[i, j])])
                fila = cm.sum(1)
                for i, real in enumerate(REG):
                    rc = cm[i, i] / fila[i] if fila[i] > 0 else 0.0
                    rec_tmp.append([etq, m, real, float(rc)])
            pct = d[a]["Random Forest"].get("importancia_permutacion_pct", {})
            tot = sum(v for v in pct.values() if v > 0) or 1
            for atr in ATRIB:
                imp.append([etq, a, atr, round(max(pct.get(atr, 0.0), 0.0) / tot, 4)])
            cm = np.array(d[a]["Persistencia"]["matriz_confusion"], dtype=float)
            tt = cm.sum(); fila = cm.sum(1)
            for i, real in enumerate(REG):
                dist.append([etq, a, real, int(fila[i]), round(float(100 * fila[i] / tt), 1)])

    rec = (pd.DataFrame(rec_tmp, columns=["Horizonte", "Modelo", "Regimen", "rc"])
           .groupby(["Horizonte", "Modelo", "Regimen"], sort=False)["rc"]
           .mean().round(4).reset_index().rename(columns={"rc": "Recall_Promedio"}))

    pd.DataFrame(metr, columns=["Horizonte", "Activo", "Empresa", "Modelo",
                                "Accuracy", "Precision", "Recall", "F1_macro"]
                 ).to_csv("pbi_metricas.csv", index=False, encoding=ENC)
    pd.DataFrame(conf, columns=["Horizonte", "Activo", "Modelo",
                                "Regimen_Real", "Regimen_Predicho", "Conteo"]
                 ).to_csv("pbi_matrices_confusion.csv", index=False, encoding=ENC)
    pd.DataFrame(imp, columns=["Horizonte", "Activo", "Atributo", "Importancia"]
                 ).to_csv("pbi_importancias.csv", index=False, encoding=ENC)
    rec.to_csv("pbi_recall_regimen.csv", index=False, encoding=ENC)
    pd.DataFrame(dist, columns=["Horizonte", "Activo", "Regimen",
                                "Conteo_Test", "Porcentaje"]
                 ).to_csv("pbi_distribucion_clases.csv", index=False, encoding=ENC)
    print("Generados los 5 ficheros de comparacion.")


def generar_estado_actual():
    """Genera pbi_estado_actual.csv con la volatilidad y el regimen mas reciente."""
    import yfinance as yf
    filas = []
    for tk in ACT:
        df = yf.download(tk, start=INICIO, end=FIN, auto_adjust=True, progress=False)
        close = df["Close"].squeeze()
        ret = np.log(close / close.shift(1))
        vol = ret.rolling(VENTANA_VOL).std() * ANUALIZACION
        reg = pd.qcut(vol, q=3, labels=["Bajo", "Medio", "Alto"])
        p33, p66 = vol.quantile(1 / 3), vol.quantile(2 / 3)
        d = pd.DataFrame({"vol": vol, "reg": reg}).dropna()
        ult = d.iloc[-1]
        filas.append([tk, d.index[-1].strftime("%Y-%m-%d"),
                      round(float(ult["vol"]) * 100, 1), str(ult["reg"]),
                      round(float(p33) * 100, 1), round(float(p66) * 100, 1)])
    pd.DataFrame(filas, columns=["Activo", "Fecha", "Volatilidad_actual",
                                 "Regimen_actual", "Umbral_bajo_medio", "Umbral_medio_alto"]
                 ).to_csv("pbi_estado_actual.csv", index=False, encoding=ENC)
    print("Generado pbi_estado_actual.csv.")


def main():
    generar_comparacion()
    try:
        generar_estado_actual()
    except Exception as e:
        print(f"AVISO: no se pudo generar pbi_estado_actual.csv ({e}). "
              "Comprueba la conexion a internet y la instalacion de yfinance.")
    print("Proceso terminado.")


if __name__ == "__main__":
    main()
