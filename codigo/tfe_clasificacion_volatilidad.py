"""
TFE - Comparativa de Tecnicas de Aprendizaje Supervisado para la
Clasificacion de Regimenes de Volatilidad en Mercados Financieros
Jean Raul Rivolta Baptista | UNIR | Director: Deivis Ramirez

Pipeline reproducible que compara cinco tecnicas de aprendizaje supervisado
(regresion logistica, arbol de decision, Random Forest, XGBoost y SVM) para la
clasificacion del regimen de volatilidad de cuatro activos del NASDAQ.

  - Activos:     AAPL, AMZN, GOOGL, TSLA
  - Periodo:     2015-01-01 a 2024-12-31
  - Etiqueta:    volatilidad realizada a 21 dias, anualizada, discretizada en
                 terciles empiricos (bajo / medio / alto)
  - Atributos:   retornos log rezagados t-1..t-5, SMA 5, SMA 20, RSI 14,
                 volatilidad rezagada t-1 y volumen relativo (10 en total)
  - Split:       temporal 70/30 por activo, respetando el orden cronologico
  - Ajuste:      GridSearchCV con validacion cruzada temporal (TimeSeriesSplit)
  - Referencia:  regla de persistencia (asigna el ultimo regimen conocido)

El parametro HORIZONTE (ver Configuracion) permite evaluar dos tareas: el
reconocimiento del regimen del dia actual (HORIZONTE = 0) o la prediccion del
regimen a 21 dias vista (HORIZONTE = 21).

Salidas:
  - resultados_metricas_H{HORIZONTE}.csv
  - resultados_completos_H{HORIZONTE}.json

Requisitos:
  pip install yfinance pandas numpy scikit-learn xgboost
"""

import json
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuracion global
# ---------------------------------------------------------------------------
ACTIVOS = ["AAPL", "AMZN", "GOOGL", "TSLA"]
FECHA_INICIO = "2015-01-01"
FECHA_FIN = "2025-01-01"     # yfinance excluye 'end'; cubre hasta 2024-12-31
VENTANA_VOL = 21             # dias de la volatilidad realizada
ANUALIZACION = np.sqrt(252)
VENTANA_RSI = 14
TEST_SIZE = 0.30            # 30% final de la serie para prueba
SEED = 42
CLASES = ["bajo", "medio", "alto"]
CV_FOLDS = 5

# Horizonte de la etiqueta:
#   0  -> regimen del dia t (reconocimiento del estado actual)
#   21 -> regimen dentro de 21 dias bursatiles (prediccion a un mes)
HORIZONTE = 21

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Descarga de datos
# ---------------------------------------------------------------------------
def descargar_datos(ticker):
    """Descarga precios ajustados y volumen con yfinance."""
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
    """Verifica disponibilidad continua y alineacion de las cuatro series."""
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
    print(f"  Dias en la union de los 4 activos:           {n_union}")
    print(f"  Dias comunes a los 4 activos (interseccion): {len(interseccion)}")
    print(f"  Series perfectamente alineadas:              {alineados}")
    veredicto = ("OK - disponibilidad continua sin interrupciones significativas"
                 if sin_problemas and alineados
                 else "REVISAR - huecos o desalineacion entre activos")
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


def construir_atributos(df):
    """
    Calcula los 10 atributos y la serie de volatilidad realizada. La etiqueta
    se construye mas adelante, una vez hecho el split, para que los umbrales de
    los regimenes se estimen solo con datos de entrenamiento.
    """
    d = df.copy()
    d["ret_log"] = np.log(d["close"] / d["close"].shift(1))
    d["vol_real"] = d["ret_log"].rolling(VENTANA_VOL).std() * ANUALIZACION

    for k in range(1, 6):                       # retornos log rezagados t-1..t-5
        d[f"ret_lag{k}"] = d["ret_log"].shift(k)
    d["sma_5"] = d["close"].rolling(5).mean()
    d["sma_20"] = d["close"].rolling(20).mean()
    d["rsi_14"] = calcular_rsi(d["close"], VENTANA_RSI)
    d["vol_lag1"] = d["vol_real"].shift(1)      # volatilidad rezagada t-1
    d["vol_rel"] = d["volume"] / d["volume"].rolling(20).mean()

    atributos = [f"ret_lag{k}" for k in range(1, 6)] + \
                ["sma_5", "sma_20", "rsi_14", "vol_lag1", "vol_rel"]

    d = d.dropna(subset=atributos + ["vol_real"])
    X = d[atributos].copy()
    vol_real = d["vol_real"].copy()
    return X, vol_real, atributos


def etiquetar(vol_real, corte, horizonte):
    """
    Construye la etiqueta de regimen. Los umbrales de los terciles se estiman
    unicamente con el tramo de entrenamiento (primeras 'corte' observaciones) y
    se aplican por igual a entrenamiento y prueba, reproduciendo las condiciones
    de un uso real en el que solo se conoce el pasado.

    Devuelve la etiqueta objetivo (regimen del dia t+horizonte), el regimen del
    dia t (que emplea la referencia de persistencia) y los umbrales usados.
    """
    q1, q2 = vol_real.iloc[:corte].quantile([1/3, 2/3]).values
    bordes = [-np.inf, q1, q2, np.inf]

    regimen_now = pd.cut(vol_real, bins=bordes, labels=CLASES).astype("object")
    target_vol = vol_real.shift(-horizonte)
    y = pd.cut(target_vol, bins=bordes, labels=CLASES).astype("object")

    return (pd.Series(y, index=vol_real.index),
            pd.Series(regimen_now, index=vol_real.index),
            (float(q1), float(q2)))


# ---------------------------------------------------------------------------
# 3. Definicion de modelos + rejillas de hiperparametros
# ---------------------------------------------------------------------------
def definir_modelos():
    return {
        "Regresion Logistica": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="saga", max_iter=5000, random_state=SEED)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l1", "l2"]},
        ),
        "Arbol de Decision": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [3, 5, 10, None],
             "clf__criterion": ["gini", "entropy"],
             "clf__min_samples_split": [2, 10, 20]},
        ),
        "Random Forest": (
            Pipeline([("clf", RandomForestClassifier(
                random_state=SEED, n_jobs=-1))]),
            {"clf__n_estimators": [100, 300],
             "clf__max_depth": [5, 10, None],
             "clf__max_features": ["sqrt", "log2"]},
        ),
        "XGBoost": (
            Pipeline([("clf", XGBClassifier(
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=SEED,
                n_jobs=-1, verbosity=0))]),
            {"clf__learning_rate": [0.05, 0.1, 0.3],
             "clf__max_depth": [3, 5, 7],
             "clf__n_estimators": [100, 300],
             "clf__subsample": [0.8, 1.0],
             "clf__colsample_bytree": [0.8, 1.0]},
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", decision_function_shape="ovr",
                            random_state=SEED))]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]},
        ),
    }


def _metricas(y_true, y_pred, labels):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "matriz_confusion": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Entrenamiento y evaluacion por activo
# ---------------------------------------------------------------------------
def evaluar_activo(ticker, X, vol_real, atributos):
    n = len(X)
    corte = int(n * (1 - TEST_SIZE))
    y, regimen_now, umbrales = etiquetar(vol_real, corte, HORIZONTE)

    # Con horizonte > 0 se pierden las ultimas 'HORIZONTE' filas (sin futuro).
    validos = y.notna()
    X, y, regimen_now = X[validos], y[validos], regimen_now[validos]
    n = len(X)
    corte = int(n * (1 - TEST_SIZE))

    X_tr, X_te = X.iloc[:corte], X.iloc[corte:]
    y_tr, y_te = y.iloc[:corte], y.iloc[corte:]

    mapa = {c: i for i, c in enumerate(CLASES)}   # XGBoost requiere etiqueta numerica
    cv = TimeSeriesSplit(n_splits=CV_FOLDS)

    resultados = {"_umbrales_terciles_train": umbrales}

    # Referencia de persistencia: el regimen previsto es el ultimo conocido.
    #   horizonte 0  -> el regimen de ayer
    #   horizonte >0 -> el regimen de hoy
    pred_persist = regimen_now.shift(1) if HORIZONTE == 0 else regimen_now.copy()
    yp = pred_persist.iloc[corte:]
    ok = yp.notna()
    resultados["Persistencia"] = _metricas(
        y_te[ok].tolist(), yp[ok].tolist(), CLASES)
    resultados["Persistencia"]["nota"] = "referencia sin entrenamiento"
    print(f"   {'Persistencia':22s} "
          f"f1={resultados['Persistencia']['f1_macro']:.3f}  (baseline)")

    for nombre, (pipe, grid) in definir_modelos().items():
        es_xgb = (nombre == "XGBoost")
        y_tr_fit = y_tr.map(mapa) if es_xgb else y_tr
        y_te_eval = y_te.map(mapa) if es_xgb else y_te
        labels = [0, 1, 2] if es_xgb else CLASES

        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr_fit)
        y_pred = gs.predict(X_te)

        m = _metricas(y_te_eval.tolist(), y_pred.tolist(), labels)
        m["mejores_hiperparametros"] = gs.best_params_
        m["n_train"] = int(len(X_tr))
        m["n_test"] = int(len(X_te))

        # Importancia por permutacion sobre el conjunto de prueba.
        perm = permutation_importance(
            gs.best_estimator_, X_te, y_te_eval,
            n_repeats=20, random_state=SEED, scoring="f1_macro", n_jobs=-1)
        total = perm.importances_mean.sum()
        m["importancia_permutacion"] = {
            a: round(float(v), 4) for a, v in
            zip(atributos, perm.importances_mean)}
        m["importancia_permutacion_pct"] = {
            a: (round(float(100 * v / total), 2) if total > 0 else 0.0)
            for a, v in zip(atributos, perm.importances_mean)}

        resultados[nombre] = m
        print(f"   {nombre:22s} acc={m['accuracy']:.3f}  f1={m['f1_macro']:.3f}")

    return resultados


# ---------------------------------------------------------------------------
# 5. Orquestacion
# ---------------------------------------------------------------------------
def main():
    modo = "NOWCAST (H=0)" if HORIZONTE == 0 else f"PREDICCION (H={HORIZONTE})"
    print(f"\n### MODO: {modo} ###\n")

    todo = {"_config": {"horizonte": HORIZONTE, "test_size": TEST_SIZE,
                        "ventana_vol": VENTANA_VOL, "seed": SEED}}
    filas = []

    print("Descargando los cuatro activos...")
    datos = {ticker: descargar_datos(ticker) for ticker in ACTIVOS}
    if not validar_completitud(datos):
        print("AVISO: la validacion detecto posibles huecos.\n")

    for ticker in ACTIVOS:
        print(f"[{ticker}] procesando...")
        X, vol_real, atributos = construir_atributos(datos[ticker])
        res = evaluar_activo(ticker, X, vol_real, atributos)
        todo[ticker] = res
        for modelo, m in res.items():
            if modelo.startswith("_"):
                continue
            filas.append({
                "activo": ticker, "modelo": modelo,
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "f1_macro": m["f1_macro"],
            })

    tabla = pd.DataFrame(filas)
    sufijo = f"_H{HORIZONTE}"
    tabla.to_csv(f"resultados_metricas{sufijo}.csv", index=False)

    promedio = (tabla.groupby("modelo")[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        .mean().round(4).sort_values("f1_macro", ascending=False))
    todo["_promedio_por_modelo"] = promedio.reset_index().to_dict("records")

    with open(f"resultados_completos{sufijo}.json", "w", encoding="utf-8") as f:
        json.dump(todo, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"RESUMEN: PROMEDIO POR MODELO  ({modo})")
    print("=" * 64)
    print(promedio.to_string())
    mejor_modelo = promedio.drop("Persistencia", errors="ignore")["f1_macro"].idxmax()
    f1_persist = promedio.loc["Persistencia", "f1_macro"] if "Persistencia" in promedio.index else None
    f1_mejor = promedio.loc[mejor_modelo, "f1_macro"]
    print("-" * 64)
    if f1_persist is not None:
        veredicto = ("NINGUN modelo supera la persistencia"
                     if f1_persist >= f1_mejor else
                     f"{mejor_modelo} supera la persistencia")
        print(f"  Persistencia: {f1_persist:.4f} | mejor modelo "
              f"({mejor_modelo}): {f1_mejor:.4f}  ->  {veredicto}")
    print("\nArchivos generados:")
    print(f"  - resultados_metricas{sufijo}.csv")
    print(f"  - resultados_completos{sufijo}.json")


if __name__ == "__main__":
    main()
