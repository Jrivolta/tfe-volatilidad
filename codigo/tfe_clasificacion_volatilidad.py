"""
TFE - Comparativa de Tecnicas de Aprendizaje Supervisado para la
Clasificacion de Regimenes de Volatilidad en Mercados Financieros
Jean Raul Rivolta Baptista | UNIR | Director: Deivis Ramirez

Pipeline reproducible que implementa exactamente la metodologia descrita
en el Capitulo 5 del documento:

  - Activos:        AAPL, AMZN, GOOGL, TSLA  (NASDAQ)
  - Periodo:        2015-01-01 a 2024-12-31
  - Etiqueta:       volatilidad realizada a 21 dias, anualizada (sqrt(252)),
                    discretizada en TERCILES EMPIRICOS por activo
                    (regimen bajo / medio / alto)
  - Atributos (10): retornos log rezagados t-1..t-5, SMA 5, SMA 20,
                    RSI 14, volatilidad rezagada t-1, volumen relativo
  - Split:          temporal 70/30 por activo (sin barajar)
  - Tuning:         GridSearchCV con validacion cruzada estratificada
                    (StratifiedKFold) sobre el conjunto de entrenamiento
  - Modelos:        Regresion Logistica, Arbol de Decision, Random Forest,
                    XGBoost, SVM (RBF, OvR)

SALIDAS (resultados que alimentan el analisis y las visualizaciones):
  - resultados_metricas.csv        -> tabla larga (activo, modelo, metricas)
  - resultados_completos.json      -> todo: hiperparametros, matrices de
                                      confusion, importancias, promedios
  - Resumen impreso por consola (copiable)

Requisitos:
  pip install yfinance pandas numpy scikit-learn xgboost

Ejecutar:
  python tfe_clasificacion_volatilidad.py
"""

import json
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuracion global
# ---------------------------------------------------------------------------
ACTIVOS = ["AAPL", "AMZN", "GOOGL", "TSLA"]
FECHA_INICIO = "2015-01-01"
FECHA_FIN = "2025-01-01"     # yfinance excluye 'end'; cubre hasta 2024-12-31
VENTANA_VOL = 21          # dias de la volatilidad realizada
ANUALIZACION = np.sqrt(252)
VENTANA_RSI = 14
TEST_SIZE = 0.30          # 30% final de la serie para prueba (temporal)
SEED = 42
CLASES = ["bajo", "medio", "alto"]
CV_FOLDS = 5

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Descarga de datos
# ---------------------------------------------------------------------------
def descargar_datos(ticker):
    """Descarga precios ajustados y volumen con yfinance (sin limpiar NaN aun)."""
    import yfinance as yf
    df = yf.download(
        ticker, start=FECHA_INICIO, end=FECHA_FIN,
        auto_adjust=True, progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]]
    df.columns = ["close", "volume"]
    return df


def validar_completitud(datos):
    """
    Verifica la disponibilidad continua de datos para los cuatro activos antes
    de proceder al analisis, tal como exige el apartado 5.1.1 del documento.

    Comprueba, por activo: numero de filas, valores faltantes (NaN), rango de
    fechas cubierto y dias de negociacion utiles. Despues comprueba que las
    cuatro series esten alineadas sobre el mismo calendario de negociacion
    (sin huecos relativos entre activos).
    """
    print("=" * 68)
    print("VALIDACION DE COMPLETITUD DE DATOS")
    print("=" * 68)
    fechas = {t: set(df.dropna().index) for t, df in datos.items()}
    union = sorted(set().union(*fechas.values()))
    n_union = len(union)
    sin_problemas = True

    for t, df in datos.items():
        n_filas = len(df)
        n_nan = int(df[["close", "volume"]].isna().any(axis=1).sum())
        n_utiles = len(fechas[t])
        faltan = n_union - n_utiles
        if faltan > 0 or n_nan > 0:
            sin_problemas = False
        print(f"  {t:6s} filas={n_filas}  con_NaN={n_nan}  utiles={n_utiles}  "
              f"faltan_vs_union={faltan}  "
              f"rango={df.index.min().date()} a {df.index.max().date()}")

    interseccion = set.intersection(*[fechas[t] for t in datos])
    alineados = all(fechas[t] == interseccion for t in datos)
    print("-" * 68)
    print(f"  Dias de negociacion en la union de los 4 activos: {n_union}")
    print(f"  Dias comunes a los 4 activos (interseccion):      {len(interseccion)}")
    print(f"  Series perfectamente alineadas (mismas fechas):   {alineados}")
    veredicto = ("OK - disponibilidad continua sin interrupciones significativas"
                 if sin_problemas and alineados
                 else "REVISAR - se detectaron huecos o desalineacion entre activos")
    print(f"  Veredicto: {veredicto}")
    print("=" * 68 + "\n")
    return sin_problemas and alineados


# ---------------------------------------------------------------------------
# 2. Ingenieria de atributos y etiqueta
# ---------------------------------------------------------------------------
def calcular_rsi(precios, ventana=14):
    delta = precios.diff()
    ganancia = delta.clip(lower=0).rolling(ventana).mean()
    perdida = (-delta.clip(upper=0)).rolling(ventana).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def construir_dataset(df):
    """
    Construye atributos SIN filtracion de informacion futura y la etiqueta
    de regimen de volatilidad por terciles empiricos del propio activo.
    """
    d = df.copy()
    d["ret_log"] = np.log(d["close"] / d["close"].shift(1))

    # --- Etiqueta: volatilidad realizada a 21 dias, anualizada ---
    d["vol_real"] = d["ret_log"].rolling(VENTANA_VOL).std() * ANUALIZACION

    # --- Atributos (todos con informacion hasta el dia t) ---
    for k in range(1, 6):                       # retornos log rezagados t-1..t-5
        d[f"ret_lag{k}"] = d["ret_log"].shift(k)
    d["sma_5"] = d["close"].rolling(5).mean()
    d["sma_20"] = d["close"].rolling(20).mean()
    d["rsi_14"] = calcular_rsi(d["close"], VENTANA_RSI)
    d["vol_lag1"] = d["vol_real"].shift(1)      # volatilidad rezagada t-1
    d["vol_rel"] = d["volume"] / d["volume"].rolling(20).mean()

    # --- Discretizacion en terciles empiricos (por activo) ---
    # qcut con 3 grupos balancea las clases por construccion
    d["regimen"] = pd.qcut(d["vol_real"], q=3, labels=CLASES)

    atributos = [f"ret_lag{k}" for k in range(1, 6)] + \
                ["sma_5", "sma_20", "rsi_14", "vol_lag1", "vol_rel"]

    d = d.dropna(subset=atributos + ["regimen"])
    X = d[atributos].copy()
    y = d["regimen"].astype(str).copy()
    return X, y, atributos


# ---------------------------------------------------------------------------
# 3. Definicion de modelos + rejillas de hiperparametros
# ---------------------------------------------------------------------------
def definir_modelos():
    """
    Devuelve dict modelo -> (pipeline, param_grid, requiere_escalado).
    Los modelos sensibles a escala (LogReg, SVM) usan StandardScaler.
    """
    modelos = {
        "Regresion Logistica": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="saga", max_iter=5000,
                    random_state=SEED)),
            ]),
            {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__penalty": ["l1", "l2"],
            },
        ),
        "Arbol de Decision": (
            Pipeline([
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]),
            {
                "clf__max_depth": [3, 5, 10, None],
                "clf__criterion": ["gini", "entropy"],
                "clf__min_samples_split": [2, 10, 20],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [5, 10, None],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        "XGBoost": (
            Pipeline([
                ("clf", XGBClassifier(
                    objective="multi:softprob", num_class=3,
                    eval_metric="mlogloss", random_state=SEED,
                    n_jobs=-1, verbosity=0)),
            ]),
            {
                "clf__learning_rate": [0.05, 0.1, 0.3],
                "clf__max_depth": [3, 5, 7],
                "clf__n_estimators": [100, 300],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", decision_function_shape="ovr",
                            random_state=SEED)),
            ]),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.1],
            },
        ),
    }
    return modelos


# ---------------------------------------------------------------------------
# 4. Entrenamiento y evaluacion por activo
# ---------------------------------------------------------------------------
def evaluar_activo(ticker, X, y, atributos):
    # Split temporal 70/30 (sin barajar para respetar la cronologia)
    n = len(X)
    corte = int(n * (1 - TEST_SIZE))
    X_tr, X_te = X.iloc[:corte], X.iloc[corte:]
    y_tr, y_te = y.iloc[:corte], y.iloc[corte:]

    # XGBoost necesita etiquetas numericas
    mapa = {c: i for i, c in enumerate(CLASES)}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    resultados = {}
    for nombre, (pipe, grid) in definir_modelos().items():
        y_tr_fit = y_tr.map(mapa) if nombre == "XGBoost" else y_tr
        y_te_eval = y_te.map(mapa) if nombre == "XGBoost" else y_te

        gs = GridSearchCV(pipe, grid, scoring="f1_macro",
                          cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr_fit)
        y_pred = gs.predict(X_te)

        labels_num = [0, 1, 2] if nombre == "XGBoost" else CLASES
        cm = confusion_matrix(y_te_eval, y_pred, labels=labels_num)

        resultados[nombre] = {
            "accuracy": round(float(accuracy_score(y_te_eval, y_pred)), 4),
            "precision_macro": round(float(precision_score(
                y_te_eval, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(
                y_te_eval, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(
                y_te_eval, y_pred, average="macro", zero_division=0)), 4),
            "mejores_hiperparametros": gs.best_params_,
            "matriz_confusion": cm.tolist(),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        }

        # Importancia de atributos (solo modelos basados en arboles)
        clf = gs.best_estimator_.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            resultados[nombre]["importancias"] = dict(zip(
                atributos, [round(float(v), 4) for v in clf.feature_importances_]))

        print(f"   {nombre:22s} acc={resultados[nombre]['accuracy']:.3f}  "
              f"f1={resultados[nombre]['f1_macro']:.3f}")

    return resultados


# ---------------------------------------------------------------------------
# 5. Orquestacion
# ---------------------------------------------------------------------------
def exportar_csv_powerbi(todo, datos):
    """Genera los seis CSV que alimentan el tablero de Power BI."""
    import csv
    empresas = {"AAPL": "Apple", "AMZN": "Amazon", "GOOGL": "Alphabet", "TSLA": "Tesla"}
    cl_cap = [c.capitalize() for c in CLASES]
    modelos = list(todo[ACTIVOS[0]].keys())

    # 1. Metricas por activo y modelo
    with open("pbi_metricas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Activo", "Empresa", "Modelo",
                    "Accuracy", "Precision", "Recall", "F1_macro"])
        for a in ACTIVOS:
            for m in modelos:
                r = todo[a][m]
                w.writerow([a, empresas.get(a, a), m, r["accuracy"],
                            r["precision_macro"], r["recall_macro"], r["f1_macro"]])

    # 2. Matrices de confusion (formato largo)
    with open("pbi_matrices_confusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Activo", "Modelo", "Regimen_Real", "Regimen_Predicho", "Conteo"])
        for a in ACTIVOS:
            for m in modelos:
                cm = todo[a][m]["matriz_confusion"]
                for i in range(3):
                    for j in range(3):
                        w.writerow([a, m, cl_cap[i], cl_cap[j], cm[i][j]])

    # 3. Importancia de atributos (Random Forest)
    with open("pbi_importancias.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Activo", "Atributo", "Importancia"])
        for a in ACTIVOS:
            imp = todo[a].get("Random Forest", {}).get("importancias", {})
            for atr, val in imp.items():
                w.writerow([a, atr, val])

    # 4. Recall por regimen (promedio sobre activos)
    with open("pbi_recall_regimen.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Modelo", "Regimen", "Recall_Promedio"])
        for m in modelos:
            rec = np.zeros(3)
            for a in ACTIVOS:
                cm = np.array(todo[a][m]["matriz_confusion"], dtype=float)
                for c in range(3):
                    s = cm[c].sum()
                    rec[c] += cm[c, c] / s if s > 0 else 0.0
            rec /= len(ACTIVOS)
            for c in range(3):
                w.writerow([m, cl_cap[c], round(float(rec[c]), 4)])

    # 5. Distribucion de clases en el conjunto de prueba
    with open("pbi_distribucion_clases.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Activo", "Regimen", "Conteo_Test", "Porcentaje"])
        for a in ACTIVOS:
            fila = np.array(todo[a][modelos[0]]["matriz_confusion"], dtype=float).sum(axis=1)
            tot = fila.sum()
            for c in range(3):
                w.writerow([a, cl_cap[c], int(fila[c]), round(100 * fila[c] / tot, 1)])

    # 6. Estado actual de volatilidad por activo
    with open("pbi_estado_actual.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Activo", "Fecha", "Volatilidad_actual", "Regimen_actual",
                    "Umbral_bajo_medio", "Umbral_medio_alto"])
        for a in ACTIVOS:
            d = datos[a].copy()
            d["ret_log"] = np.log(d["close"] / d["close"].shift(1))
            d["vol_real"] = d["ret_log"].rolling(VENTANA_VOL).std() * ANUALIZACION
            d["regimen"] = pd.qcut(d["vol_real"], q=3, labels=CLASES)
            p33 = d["vol_real"].quantile(1 / 3)
            p66 = d["vol_real"].quantile(2 / 3)
            ult = d.dropna(subset=["vol_real", "regimen"]).iloc[-1]
            w.writerow([a, ult.name.strftime("%Y-%m-%d"),
                        round(float(ult["vol_real"]) * 100, 1),
                        str(ult["regimen"]).capitalize(),
                        round(float(p33) * 100, 1), round(float(p66) * 100, 1)])

    print("\nEstado actual de volatilidad por activo:")
    for a in ACTIVOS:
        d = datos[a].copy()
        d["ret_log"] = np.log(d["close"] / d["close"].shift(1))
        d["vol_real"] = d["ret_log"].rolling(VENTANA_VOL).std() * ANUALIZACION
        d["regimen"] = pd.qcut(d["vol_real"], q=3, labels=CLASES)
        ult = d.dropna(subset=["vol_real", "regimen"]).iloc[-1]
        print(f"  {a}: {round(float(ult['vol_real']) * 100, 1)}%  ->  "
              f"{str(ult['regimen']).capitalize()}")


def main():
    todo = {}
    filas = []

    # Descarga de los cuatro activos y validacion previa de completitud
    print("Descargando los cuatro activos...")
    datos = {ticker: descargar_datos(ticker) for ticker in ACTIVOS}
    completo = validar_completitud(datos)
    if not completo:
        print("AVISO: la validacion detecto posibles huecos. "
              "Revisa el reporte anterior antes de interpretar los resultados.\n")

    for ticker in ACTIVOS:
        print(f"[{ticker}] procesando...")
        X, y, atributos = construir_dataset(datos[ticker])
        print(f"   observaciones utiles: {len(X)} | "
              f"distribucion clases: {y.value_counts().to_dict()}")
        res = evaluar_activo(ticker, X, y, atributos)
        todo[ticker] = res
        for modelo, m in res.items():
            filas.append({
                "activo": ticker, "modelo": modelo,
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "f1_macro": m["f1_macro"],
            })

    # Tabla larga
    tabla = pd.DataFrame(filas)
    tabla.to_csv("resultados_metricas.csv", index=False)

    # Promedio por modelo (para el tablero tipo semaforo)
    promedio = (tabla.groupby("modelo")[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        .mean().round(4).sort_values("f1_macro", ascending=False))

    todo["_promedio_por_modelo"] = promedio.reset_index().to_dict("records")
    with open("resultados_completos.json", "w", encoding="utf-8") as f:
        json.dump(todo, f, ensure_ascii=False, indent=2)

    # Genera los seis CSV que alimentan el tablero de Power BI
    exportar_csv_powerbi(todo, datos)

    print("\n" + "=" * 64)
    print("RESUMEN: PROMEDIO POR MODELO (todos los activos)")
    print("=" * 64)
    print(promedio.to_string())
    print("\nArchivos generados:")
    print("  - resultados_metricas.csv")
    print("  - resultados_completos.json")
    print("  - pbi_metricas.csv")
    print("  - pbi_matrices_confusion.csv")
    print("  - pbi_importancias.csv")
    print("  - pbi_recall_regimen.csv")
    print("  - pbi_distribucion_clases.csv")
    print("  - pbi_estado_actual.csv")
    print("\nProceso finalizado. Los seis archivos pbi_*.csv alimentan el tablero de Power BI.")


if __name__ == "__main__":
    main()
