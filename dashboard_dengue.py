import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import unicodedata
import json
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------
st.set_page_config(
    page_title="Predicción de Riesgo: Dengue",
    page_icon="🦟",
    layout="wide"
)

# --------------------------------------------------
# ESTILOS
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #9aa0a6;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #111827;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #1f2937;
}
.small-text {
    font-size: 0.9rem;
    color: #9aa0a6;
}
.block-card {
    background-color: #0f172a;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #1e293b;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SEGURIDAD / SANITIZACIÓN BÁSICA
# --------------------------------------------------
ALLOWED_BASE_DIR = Path("dataset").resolve()

def ruta_segura(ruta: str | Path) -> Path:
    ruta = Path(ruta).resolve()
    if not str(ruta).startswith(str(ALLOWED_BASE_DIR)):
        raise ValueError("Ruta fuera del directorio permitido.")
    return ruta

def sanitizar_texto(texto: str) -> str:
    texto = str(texto).strip()
    texto = texto.replace("<", "").replace(">", "")
    texto = texto.replace(";", "").replace("--", "")
    return texto

# --------------------------------------------------
# FUNCIONES AUXILIARES
# --------------------------------------------------
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columnas_normalizadas = []

    for col in df.columns:
        col = str(col).strip().lower()
        col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
        col = col.replace(" ", "_")
        col = "".join(ch for ch in col if ch.isalnum() or ch == "_")
        columnas_normalizadas.append(col)

    df.columns = columnas_normalizadas
    return df


def limpiar_texto_ubicacion(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalizar_texto_simple(texto: str) -> str:
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = " ".join(texto.split())
    return texto


def _primera_columna_disponible(df: pd.DataFrame, candidatas: list[str]) -> str | None:
    for columna in candidatas:
        if columna in df.columns:
            return columna
    return None


def estandarizar_nombre_provincia(texto: str) -> str:
    texto = str(texto).strip().upper()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = " ".join(texto.split())

    equivalencias = {
        "CAÑAR": "CANAR",
        "CANAR": "CANAR",
        "GALÁPAGOS": "GALAPAGOS",
        "GALAPAGOS": "GALAPAGOS",
        "LOS RÍOS": "LOS RIOS",
        "LOS RIOS": "LOS RIOS",
        "SANTO DOMINGO DE LOS TSÁCHILAS": "SANTO DOMINGO DE LOS TSACHILAS",
        "SANTO DOMINGO DE LOS TSACHILAS": "SANTO DOMINGO DE LOS TSACHILAS",
        "MORONA-SANTIAGO": "MORONA SANTIAGO",
        "ZAMORA-CHINCHIPE": "ZAMORA CHINCHIPE",
        "SUCUMBÍOS": "SUCUMBIOS",
        "EL ORO": "EL ORO",
        "SANTA ELENA": "SANTA ELENA",
    }

    return equivalencias.get(texto, texto)


def preparar_geojson_provincias(geojson: dict) -> dict:
    for feature in geojson["features"]:
        props = feature.get("properties", {})
        nombre = (
            props.get("name")
            or props.get("NAME")
            or props.get("provincia")
            or props.get("PROVINCIA")
            or props.get("DPA_DESPRO")
            or props.get("NOM_PROV")
            or ""
        )
        feature["properties"]["provincia_std"] = estandarizar_nombre_provincia(nombre)
    return geojson


CODIGOS_PROVINCIA_ECUADOR = {
    "01": "AZUAY",
    "02": "BOLIVAR",
    "03": "CANAR",
    "04": "CARCHI",
    "05": "COTOPAXI",
    "06": "CHIMBORAZO",
    "07": "EL ORO",
    "08": "ESMERALDAS",
    "09": "GUAYAS",
    "10": "IMBABURA",
    "11": "LOJA",
    "12": "LOS RIOS",
    "13": "MANABI",
    "14": "MORONA SANTIAGO",
    "15": "NAPO",
    "16": "PASTAZA",
    "17": "PICHINCHA",
    "18": "TUNGURAHUA",
    "19": "ZAMORA CHINCHIPE",
    "20": "GALAPAGOS",
    "21": "SUCUMBIOS",
    "22": "ORELLANA",
    "23": "SANTO DOMINGO DE LOS TSACHILAS",
    "24": "SANTA ELENA",
}

# --------------------------------------------------
# CONSTRUCCIÓN DEL DATASET UNIFICADO
# --------------------------------------------------
def construir_dataset_unificado(
    ruta_dengue: str | Path,
    ruta_lluvia: str | Path,
    ruta_clima: str | Path,
) -> pd.DataFrame:
    ruta_dengue = ruta_segura(ruta_dengue)
    ruta_lluvia = ruta_segura(ruta_lluvia)
    ruta_clima = ruta_segura(ruta_clima)

    # ----------------------------
    # DENGUE
    # ----------------------------
    hojas_dengue = []
    excel_dengue = pd.ExcelFile(ruta_dengue)

    for hoja in excel_dengue.sheet_names:
        df_hoja = pd.read_excel(excel_dengue, sheet_name=hoja)
        df_hoja = normalizar_columnas(df_hoja)

        col_anio = _primera_columna_disponible(df_hoja, ["anio", "ano"])
        col_semana = _primera_columna_disponible(df_hoja, ["semana", "se"])
        col_provincia = _primera_columna_disponible(df_hoja, ["provincia", "prov_domic"])
        col_total = _primera_columna_disponible(df_hoja, ["total"])

        if not all([col_anio, col_semana, col_provincia]):
            continue

        df_hoja["anio"] = pd.to_numeric(df_hoja[col_anio], errors="coerce")
        df_hoja["semana_epi"] = pd.to_numeric(df_hoja[col_semana], errors="coerce")
        df_hoja["provincia_std"] = limpiar_texto_ubicacion(df_hoja[col_provincia]).apply(estandarizar_nombre_provincia)
        df_hoja["casos_dengue"] = (
            pd.to_numeric(df_hoja[col_total], errors="coerce").fillna(1)
            if col_total else 1
        )

        hojas_dengue.append(df_hoja[["anio", "semana_epi", "provincia_std", "casos_dengue"]])

    if not hojas_dengue:
        raise ValueError("No se pudieron identificar columnas válidas en el archivo de dengue.")

    dengue = pd.concat(hojas_dengue, ignore_index=True)
    dengue = dengue.dropna(subset=["anio", "semana_epi", "provincia_std"])
    dengue["anio"] = dengue["anio"].astype(int)
    dengue["semana_epi"] = dengue["semana_epi"].astype(int)

    dengue = (
        dengue.groupby(["anio", "semana_epi", "provincia_std"], as_index=False)["casos_dengue"]
        .sum()
        .drop_duplicates()
        .sort_values(["provincia_std", "anio", "semana_epi"])
    )

    # ----------------------------
    # LLUVIA
    # ----------------------------
    lluvia = pd.read_csv(ruta_lluvia)
    lluvia = normalizar_columnas(lluvia)

    if "date" not in lluvia.columns or "pcode" not in lluvia.columns or "rfh" not in lluvia.columns:
        raise ValueError("El dataset de lluvia no contiene las columnas esperadas: date, pcode, rfh.")

    lluvia["date"] = pd.to_datetime(lluvia["date"], errors="coerce")
    lluvia["rfh"] = pd.to_numeric(lluvia["rfh"], errors="coerce")
    lluvia["pcode"] = lluvia["pcode"].astype(str).str.upper()

    lluvia["codigo_provincia"] = lluvia["pcode"].str.extract(r"EC(\d{2})")
    lluvia["provincia_std"] = lluvia["codigo_provincia"].map(CODIGOS_PROVINCIA_ECUADOR)
    lluvia["provincia_std"] = lluvia["provincia_std"].apply(lambda x: estandarizar_nombre_provincia(x) if pd.notna(x) else x)

    lluvia["anio"] = lluvia["date"].dt.year
    lluvia["semana_epi"] = lluvia["date"].dt.isocalendar().week.astype("Int64")

    lluvia = lluvia.dropna(subset=["anio", "semana_epi", "provincia_std", "rfh"])
    lluvia = (
        lluvia.groupby(["anio", "semana_epi", "provincia_std"], as_index=False)["rfh"]
        .sum()
        .rename(columns={"rfh": "precipitacion_total"})
        .drop_duplicates()
    )

    # ----------------------------
    # CLIMA
    # ----------------------------
    clima = pd.read_csv(ruta_clima)
    clima = normalizar_columnas(clima)

    if "date" not in clima.columns:
        raise ValueError("El dataset climático no contiene la columna 'date'.")

    clima["date"] = pd.to_datetime(clima["date"], errors="coerce")

    # Intento de usar Ecuador; si no existe, usa South America
    if "country_name" in clima.columns and "ecuador" in clima["country_name"].astype(str).str.lower().unique():
        clima_base = clima[clima["country_name"].astype(str).str.lower() == "ecuador"].copy()
        fuente_clima = "Ecuador"
    elif "region" in clima.columns:
        clima_base = clima[clima["region"].astype(str).str.lower() == "south america"].copy()
        fuente_clima = "South America"
    else:
        clima_base = clima.copy()
        fuente_clima = "Global"

    clima_base["anio"] = clima_base["date"].dt.year
    clima_base["semana_epi"] = clima_base["date"].dt.isocalendar().week.astype("Int64")

    columnas_clima = [c for c in ["temperature_celsius", "vector_disease_risk_score", "precipitation_mm"] if c in clima_base.columns]
    if not columnas_clima:
        raise ValueError("No se encontraron columnas climáticas útiles.")

    clima_base = (
        clima_base.groupby(["anio", "semana_epi"], as_index=False)[columnas_clima]
        .mean()
        .rename(columns={
            "temperature_celsius": "temperatura_promedio",
            "vector_disease_risk_score": "riesgo_vectorial_climatico",
            "precipitation_mm": "precipitacion_climatica_mm",
        })
    )
    clima_base["fuente_clima"] = fuente_clima

    # ----------------------------
    # UNIÓN FINAL
    # ----------------------------
    base_unificada = dengue.merge(
        lluvia,
        on=["anio", "semana_epi", "provincia_std"],
        how="left"
    ).merge(
        clima_base,
        on=["anio", "semana_epi"],
        how="left"
    )

    # Imputación / limpieza
    base_unificada["precipitacion_total"] = pd.to_numeric(base_unificada["precipitacion_total"], errors="coerce")
    base_unificada["temperatura_promedio"] = pd.to_numeric(base_unificada["temperatura_promedio"], errors="coerce")
    base_unificada["riesgo_vectorial_climatico"] = pd.to_numeric(base_unificada.get("riesgo_vectorial_climatico", np.nan), errors="coerce")
    base_unificada["precipitacion_climatica_mm"] = pd.to_numeric(base_unificada.get("precipitacion_climatica_mm", np.nan), errors="coerce")

    base_unificada["precipitacion_total"] = base_unificada["precipitacion_total"].fillna(base_unificada["precipitacion_total"].median())
    base_unificada["temperatura_promedio"] = base_unificada["temperatura_promedio"].fillna(base_unificada["temperatura_promedio"].median())
    base_unificada["riesgo_vectorial_climatico"] = base_unificada["riesgo_vectorial_climatico"].fillna(base_unificada["riesgo_vectorial_climatico"].median())
    base_unificada["precipitacion_climatica_mm"] = base_unificada["precipitacion_climatica_mm"].fillna(base_unificada["precipitacion_climatica_mm"].median())

    # Rangos válidos
    base_unificada["semana_epi"] = pd.to_numeric(base_unificada["semana_epi"], errors="coerce").clip(1, 53)
    base_unificada["casos_dengue"] = pd.to_numeric(base_unificada["casos_dengue"], errors="coerce").fillna(0).clip(lower=0)

    return base_unificada.sort_values(["provincia_std", "anio", "semana_epi"]).reset_index(drop=True)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["provincia_std", "anio", "semana_epi"]).reset_index(drop=True)

    # índice temporal simple
    df["time_index"] = df["anio"] * 100 + df["semana_epi"]

    # lags
    for lag in [1, 2, 3, 4]:
        df[f"casos_lag_{lag}"] = df.groupby("provincia_std")["casos_dengue"].shift(lag)

    # medias móviles
    df["media_3_sem"] = (
        df.groupby("provincia_std")["casos_dengue"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df["media_4_sem_lluvia"] = (
        df.groupby("provincia_std")["precipitacion_total"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    )

    # ratios e interacciones
    df["ratio_lluvia_temp"] = df["precipitacion_total"] / (df["temperatura_promedio"].replace(0, np.nan))
    df["interaccion_temp_lluvia"] = df["temperatura_promedio"] * df["precipitacion_total"]
    df["interaccion_hist_clima"] = df["casos_lag_1"] * df["temperatura_promedio"]

    # temporalidad cíclica
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_epi"] / 52.0)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_epi"] / 52.0)

    # target: casos futuros 1 semana adelante
    df["casos_futuros"] = df.groupby("provincia_std")["casos_dengue"].shift(-1)

    # clasificación del riesgo futuro
    q1 = df["casos_futuros"].quantile(0.33)
    q2 = df["casos_futuros"].quantile(0.66)

    def clasif(c):
        if pd.isna(c):
            return np.nan
        if c <= q1:
            return "BAJO"
        elif c <= q2:
            return "MEDIO"
        return "ALTO"

    df["riesgo_futuro_clase"] = df["casos_futuros"].apply(clasif)

    # sanitización final
    df = df.replace([np.inf, -np.inf], np.nan)

    cols_fill = [
        "casos_lag_1", "casos_lag_2", "casos_lag_3", "casos_lag_4",
        "media_3_sem", "media_4_sem_lluvia",
        "ratio_lluvia_temp", "interaccion_temp_lluvia", "interaccion_hist_clima"
    ]
    for c in cols_fill:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


# --------------------------------------------------
# SELECCIÓN DE CARACTERÍSTICAS
# --------------------------------------------------
def seleccionar_features(df_modelo: pd.DataFrame):
    features = [
        "temperatura_promedio",
        "precipitacion_total",
        "riesgo_vectorial_climatico",
        "precipitacion_climatica_mm",
        "semana_epi",
        "semana_sin",
        "semana_cos",
        "casos_lag_1",
        "casos_lag_2",
        "casos_lag_3",
        "casos_lag_4",
        "media_3_sem",
        "media_4_sem_lluvia",
        "ratio_lluvia_temp",
        "interaccion_temp_lluvia",
        "interaccion_hist_clima",
    ]

    existentes = [f for f in features if f in df_modelo.columns]

    # selección inicial por correlación con casos futuros
    corr_base = df_modelo[existentes + ["casos_futuros"]].copy()
    corr = corr_base.corr(numeric_only=True)["casos_futuros"].drop("casos_futuros").abs().sort_values(ascending=False)

    top_features = corr.head(min(10, len(corr))).index.tolist()
    return top_features, corr


# --------------------------------------------------
# ENTRENAMIENTO / MODELADO
# --------------------------------------------------
def entrenar_modelos(df_full: pd.DataFrame):
    df = df_full.copy()
    df = df.dropna(subset=["riesgo_futuro_clase", "casos_futuros"])

    selected_features, corr = seleccionar_features(df)

    # split temporal
    tiempo_ordenado = df[["anio", "semana_epi"]].drop_duplicates().sort_values(["anio", "semana_epi"])
    corte = int(len(tiempo_ordenado) * 0.80)
    tiempo_train = tiempo_ordenado.iloc[:corte]
    tiempo_test = tiempo_ordenado.iloc[corte:]

    train_keys = set(zip(tiempo_train["anio"], tiempo_train["semana_epi"]))
    test_keys = set(zip(tiempo_test["anio"], tiempo_test["semana_epi"]))

    df_train = df[df[["anio", "semana_epi"]].apply(tuple, axis=1).isin(train_keys)].copy()
    df_test = df[df[["anio", "semana_epi"]].apply(tuple, axis=1).isin(test_keys)].copy()

    X_train = df_train[selected_features].copy()
    y_train = df_train["riesgo_futuro_clase"].copy()

    X_test = df_test[selected_features].copy()
    y_test = df_test["riesgo_futuro_clase"].copy()

    # Modelo 1: Random Forest
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [4, 8, None],
        "min_samples_split": [2, 5]
    }
    grid_rf = GridSearchCV(
        rf,
        rf_params,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)

    # Modelo 2: Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        "n_estimators": [100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3]
    }
    grid_gb = GridSearchCV(
        gb,
        gb_params,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )
    grid_gb.fit(X_train, y_train)

    modelos = {
        "Random Forest": grid_rf.best_estimator_,
        "Gradient Boosting": grid_gb.best_estimator_,
    }

    resultados = {}
    mejor_modelo_nombre = None
    mejor_f1 = -1
    mejor_modelo = None

    for nombre, modelo in modelos.items():
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="macro", zero_division=0)
        rec = recall_score(y_test, pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, pred, average="macro", zero_division=0)

        resultados[nombre] = {
            "modelo": modelo,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "y_test": y_test,
            "pred": pred,
            "X_test": X_test,
            "X_train": X_train,
            "best_params": grid_rf.best_params_ if nombre == "Random Forest" else grid_gb.best_params_,
        }

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_modelo_nombre = nombre
            mejor_modelo = modelo

    return {
        "selected_features": selected_features,
        "corr": corr,
        "df_train": df_train,
        "df_test": df_test,
        "resultados": resultados,
        "mejor_modelo_nombre": mejor_modelo_nombre,
        "mejor_modelo": mejor_modelo
    }


# --------------------------------------------------
# INTERPRETABILIDAD
# --------------------------------------------------
def obtener_importancias(modelo, X_test: pd.DataFrame) -> pd.DataFrame:
    if hasattr(modelo, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": X_test.columns,
            "importance": modelo.feature_importances_
        }).sort_values("importance", ascending=False)
        return imp

    perm = permutation_importance(modelo, X_test, modelo.predict(X_test), n_repeats=5, random_state=42)
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)
    return imp


# --------------------------------------------------
# MAPA / RESUMEN PROVINCIAL
# --------------------------------------------------
def construir_prediccion_provincial(df_modelo: pd.DataFrame, modelo, selected_features: list[str]) -> pd.DataFrame:
    df = df_modelo.copy()

    # usar último registro disponible por provincia
    ultimo = (
        df.sort_values(["provincia_std", "anio", "semana_epi"])
          .groupby("provincia_std", as_index=False)
          .tail(1)
          .copy()
    )

    X_pred = ultimo[selected_features].copy()
    pred = modelo.predict(X_pred)

    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_pred)
        conf = proba.max(axis=1) * 100
    else:
        conf = np.full(len(ultimo), 0.0)

    mapping_riesgo_num = {"BAJO": 33, "MEDIO": 66, "ALTO": 100}
    ultimo["nivel_predicho"] = pred
    ultimo["riesgo_total"] = pd.Series(pred).map(mapping_riesgo_num).values
    ultimo["confianza_modelo"] = conf

    resumen_casos = (
        df.groupby("provincia_std", as_index=False)["casos_dengue"]
        .sum()
        .rename(columns={"casos_dengue": "casos_dengue_periodo"})
    )

    ultimo = ultimo.merge(resumen_casos, on="provincia_std", how="left")
    return ultimo


def crear_mapa_provincias(df_riesgo: pd.DataFrame, geojson: dict):
    df_mapa = df_riesgo.copy()
    df_mapa["provincia_std"] = df_mapa["provincia_std"].apply(normalizar_texto_simple)

    fig = px.choropleth_mapbox(
        df_mapa,
        geojson=geojson,
        locations="provincia_std",
        featureidkey="properties.provincia_std",
        color="riesgo_total",
        color_continuous_scale=[
            [0.00, "#22c55e"],
            [0.50, "#f59e0b"],
            [1.00, "#ef4444"]
        ],
        range_color=(0, 100),
        mapbox_style="carto-positron",
        zoom=4.8,
        center={"lat": -1.6, "lon": -78.3},
        opacity=0.78,
        hover_name="provincia_std",
        hover_data={
            "nivel_predicho": True,
            "confianza_modelo": ":.1f",
            "anio": True,
            "semana_epi": True,
            "casos_dengue_periodo": True,
            "temperatura_promedio": ":.1f",
            "precipitacion_total": ":.1f",
            "provincia_std": False
        },
    )

    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=540,
        coloraxis_colorbar=dict(title="Riesgo")
    )
    return fig


# --------------------------------------------------
# TÍTULO
# --------------------------------------------------
st.markdown('<div class="main-title">🦟 Predicción de Riesgo: Dengue</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Pipeline completo: integración, preparación, ingeniería de características, modelado, evaluación, despliegue e interpretabilidad.</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------------------------------------
# RUTAS LOCALES
# --------------------------------------------------
rutas_locales = {
    "dengue": Path("dataset/Datos_Dengue_MSP_Ene2021_Ago2025.xlsx"),
    "lluvia": Path("dataset/ecu-rainfall-subnat-full.csv"),
    "clima": Path("dataset/global_climate_health_impact_tracker_2015_2025.csv"),
    "geojson": Path("dataset/ecuador_provincias.geojson"),
}

# --------------------------------------------------
# CARGA Y PREPARACIÓN
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_y_preparar():
    base = construir_dataset_unificado(
        rutas_locales["dengue"],
        rutas_locales["lluvia"],
        rutas_locales["clima"],
    )
    base = agregar_features(base)
    return base

base_modelo = None

try:
    base_modelo = cargar_y_preparar()
    st.success("Base unificada, limpiada y enriquecida correctamente.")
except ImportError:
    st.error("Hace falta instalar openpyxl para leer el Excel.")
except Exception as exc:
    st.error(f"No se pudo construir la base unificada: {exc}")

if base_modelo is None or base_modelo.empty:
    st.stop()

# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def entrenar_pipeline(df):
    return entrenar_modelos(df)

with st.spinner("Entrenando modelos y evaluando pipeline..."):
    entrenamiento = entrenar_pipeline(base_modelo)

mejor_modelo = entrenamiento["mejor_modelo"]
mejor_modelo_nombre = entrenamiento["mejor_modelo_nombre"]
selected_features = entrenamiento["selected_features"]
resultados = entrenamiento["resultados"]
resultado_mejor = resultados[mejor_modelo_nombre]

# --------------------------------------------------
# FILTROS
# --------------------------------------------------
st.subheader("Filtros de análisis")

anios_disponibles = sorted(base_modelo["anio"].dropna().unique().tolist())
opciones_anio = ["Todos"] + [int(a) for a in anios_disponibles]
anio_sel = st.selectbox("Filtrar por año", opciones_anio)

if anio_sel == "Todos":
    base_filtrada = base_modelo.copy()
else:
    base_filtrada = base_modelo[base_modelo["anio"] == anio_sel].copy()

provincias_disponibles = sorted(base_filtrada["provincia_std"].dropna().unique().tolist())
provincia_sel = st.selectbox("Provincia", provincias_disponibles)

df_filtrado = base_filtrada[base_filtrada["provincia_std"] == provincia_sel].copy()

if not df_filtrado.empty:
    fila = df_filtrado.sort_values(["anio", "semana_epi"]).iloc[-1]
else:
    fila = base_modelo.sort_values(["anio", "semana_epi"]).iloc[-1]

temp = float(fila.get("temperatura_promedio", 28.0))
precip = float(fila.get("precipitacion_total", 80.0))
semana_epi = int(fila.get("semana_epi", 12))
casos_lag_1 = int(fila.get("casos_lag_1", 0))
casos_lag_2 = int(fila.get("casos_lag_2", 0))
casos_lag_3 = int(fila.get("casos_lag_3", 0))
casos_lag_4 = int(fila.get("casos_lag_4", 0))
riesgo_vectorial = float(fila.get("riesgo_vectorial_climatico", 0))
precip_climatica = float(fila.get("precipitacion_climatica_mm", 0))

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Parámetros de simulación")

temp = st.sidebar.slider("Temperatura (°C)", 15.0, 40.0, float(temp), 0.5)
precip = st.sidebar.slider("Precipitación (mm/semana)", 0.0, 500.0, float(min(500.0, precip)), 5.0)
semana_epi = st.sidebar.slider("Semana epidemiológica", 1, 52, int(semana_epi), 1)
casos_lag_1 = st.sidebar.slider("Casos lag 1", 0, 300, int(min(300, casos_lag_1)), 1)
casos_lag_2 = st.sidebar.slider("Casos lag 2", 0, 300, int(min(300, casos_lag_2)), 1)
casos_lag_3 = st.sidebar.slider("Casos lag 3", 0, 300, int(min(300, casos_lag_3)), 1)
casos_lag_4 = st.sidebar.slider("Casos lag 4", 0, 300, int(min(300, casos_lag_4)), 1)

st.sidebar.markdown("---")
st.sidebar.info("La app ya entrena modelos reales y usa el mejor para inferencia de riesgo.")

# --------------------------------------------------
# INFERENCIA INDIVIDUAL
# --------------------------------------------------
registro_pred = {
    "temperatura_promedio": temp,
    "precipitacion_total": precip,
    "riesgo_vectorial_climatico": riesgo_vectorial if pd.notna(riesgo_vectorial) else 0,
    "precipitacion_climatica_mm": precip_climatica if pd.notna(precip_climatica) else precip,
    "semana_epi": semana_epi,
    "semana_sin": np.sin(2 * np.pi * semana_epi / 52.0),
    "semana_cos": np.cos(2 * np.pi * semana_epi / 52.0),
    "casos_lag_1": casos_lag_1,
    "casos_lag_2": casos_lag_2,
    "casos_lag_3": casos_lag_3,
    "casos_lag_4": casos_lag_4,
    "media_3_sem": np.mean([casos_lag_1, casos_lag_2, casos_lag_3]),
    "media_4_sem_lluvia": precip,
    "ratio_lluvia_temp": precip / temp if temp != 0 else 0,
    "interaccion_temp_lluvia": temp * precip,
    "interaccion_hist_clima": casos_lag_1 * temp,
}

X_nuevo = pd.DataFrame([registro_pred])[selected_features]
pred_nivel = mejor_modelo.predict(X_nuevo)[0]

if hasattr(mejor_modelo, "predict_proba"):
    proba_individual = mejor_modelo.predict_proba(X_nuevo)[0]
    confianza = float(np.max(proba_individual) * 100)
else:
    confianza = 0.0

map_num = {"BAJO": 25, "MEDIO": 60, "ALTO": 90}
riesgo_total = map_num.get(pred_nivel, 0)

if pred_nivel == "BAJO":
    color = "#22c55e"
    interpretacion = "Condiciones relativamente favorables, con menor probabilidad de incremento de casos."
elif pred_nivel == "MEDIO":
    color = "#f59e0b"
    interpretacion = "Se detecta un riesgo intermedio; conviene reforzar monitoreo y acciones preventivas."
else:
    color = "#ef4444"
    interpretacion = "Alta probabilidad de incremento de casos; se recomienda intervención prioritaria."

# --------------------------------------------------
# MÉTRICAS SUPERIORES
# --------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filas filtradas", len(base_filtrada))
m2.metric("Provincias visibles", base_filtrada["provincia_std"].nunique())
m3.metric("Mejor modelo", mejor_modelo_nombre)
m4.metric("F1-score", f"{resultado_mejor['f1']:.3f}")

st.markdown(
    f"""
    <div class="metric-card">
        <b>Predicción actual:</b> <span style="color:{color}; font-weight:700;">{pred_nivel}</span><br>
        <span class="small-text">{interpretacion}</span><br>
        <span class="small-text">Confianza del modelo: {confianza:.1f}%</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# --------------------------------------------------
# GRÁFICOS PRINCIPALES
# --------------------------------------------------
col_grafico1, col_grafico2 = st.columns(2)

with col_grafico1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=riesgo_total,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"Nivel de Riesgo: {pred_nivel}", "font": {"size": 22, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 35], "color": "rgba(34, 197, 94, 0.20)"},
                {"range": [35, 70], "color": "rgba(245, 158, 11, 0.20)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.20)"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": riesgo_total
            }
        }
    ))

    fig_gauge.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_grafico2:
    importancias = obtener_importancias(mejor_modelo, resultado_mejor["X_test"]).head(8)
    fig_imp = px.bar(
        importancias.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="Importancia de variables"
    )
    fig_imp.update_layout(
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Importancia",
        yaxis_title="Variable"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# MÉTRICAS DE EVALUACIÓN
# --------------------------------------------------
st.subheader("Evaluación del modelo")

tabla_eval = pd.DataFrame([
    {
        "Modelo": nombre,
        "Accuracy": round(res["accuracy"], 4),
        "Precision": round(res["precision"], 4),
        "Recall": round(res["recall"], 4),
        "F1-score": round(res["f1"], 4),
    }
    for nombre, res in resultados.items()
]).sort_values("F1-score", ascending=False)

st.dataframe(tabla_eval, use_container_width=True, hide_index=True)

col_cm1, col_cm2 = st.columns(2)

with col_cm1:
    cm = confusion_matrix(resultado_mejor["y_test"], resultado_mejor["pred"], labels=["BAJO", "MEDIO", "ALTO"])
    df_cm = pd.DataFrame(cm, index=["Real BAJO", "Real MEDIO", "Real ALTO"], columns=["Pred BAJO", "Pred MEDIO", "Pred ALTO"])
    st.write("**Matriz de confusión**")
    st.dataframe(df_cm, use_container_width=True)

with col_cm2:
    st.write("**Mejores hiperparámetros**")
    st.json(resultado_mejor["best_params"])

st.markdown("---")

# --------------------------------------------------
# MAPA DE RIESGO PROVINCIAL
# --------------------------------------------------
st.subheader("Mapa de riesgo por provincia")

try:
    with open(ruta_segura(rutas_locales["geojson"]), "r", encoding="utf-8") as f:
        geojson_ecuador = json.load(f)

    geojson_ecuador = preparar_geojson_provincias(geojson_ecuador)
    df_pred_prov = construir_prediccion_provincial(base_filtrada, mejor_modelo, selected_features)

    fig_mapa = crear_mapa_provincias(df_pred_prov, geojson_ecuador)
    st.plotly_chart(fig_mapa, use_container_width=True)

    if not df_pred_prov.empty:
        provincia_mayor_riesgo = df_pred_prov.sort_values("riesgo_total", ascending=False).iloc[0]
        texto_periodo = "en todo el período analizado" if anio_sel == "Todos" else f"en el año {anio_sel}"
        st.info(
            f"Provincia con mayor riesgo {texto_periodo}: "
            f"**{provincia_mayor_riesgo['provincia_std']}** "
            f"con nivel **{provincia_mayor_riesgo['nivel_predicho']}** "
            f"y confianza de **{provincia_mayor_riesgo['confianza_modelo']:.1f}%**."
        )

except FileNotFoundError:
    st.warning("No se encontró el archivo `dataset/ecuador_provincias.geojson`.")
except Exception as exc:
    st.warning(f"No se pudo generar el mapa provincial: {exc}")

# --------------------------------------------------
# CASOS POR AÑO
# --------------------------------------------------
st.subheader("Casos de dengue por año")

casos_por_anio = (
    base_modelo.groupby("anio", as_index=False)["casos_dengue"]
    .sum()
    .sort_values("anio")
)

fig_casos_anio = px.bar(
    casos_por_anio,
    x="anio",
    y="casos_dengue",
    text="casos_dengue",
    title="Total de casos de dengue por año"
)

fig_casos_anio.update_layout(
    height=420,
    xaxis_title="Año",
    yaxis_title="Casos",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white")
)

st.plotly_chart(fig_casos_anio, use_container_width=True)

# --------------------------------------------------
# FEATURES SELECCIONADAS
# --------------------------------------------------
st.subheader("Selección de características")

corr_df = entrenamiento["corr"].reset_index()
corr_df.columns = ["feature", "correlacion_abs_con_casos_futuros"]

st.write("**Variables seleccionadas para el entrenamiento**")
st.dataframe(
    pd.DataFrame({"feature_seleccionada": selected_features}),
    use_container_width=True,
    hide_index=True
)

fig_corr = px.bar(
    corr_df.head(12),
    x="feature",
    y="correlacion_abs_con_casos_futuros",
    title="Relevancia inicial de variables"
)
fig_corr.update_layout(
    height=380,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    xaxis_title="Variable",
    yaxis_title="Correlación absoluta"
)
st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# SHAP OPCIONAL
# --------------------------------------------------
st.subheader("Interpretabilidad avanzada")

try:
    import shap

    muestra_shap = resultado_mejor["X_test"].head(50).copy()
    explainer = shap.Explainer(mejor_modelo, resultado_mejor["X_train"])
    shap_values = explainer(muestra_shap)

    shap_mean = np.abs(shap_values.values).mean(axis=0)
    df_shap = pd.DataFrame({
        "feature": muestra_shap.columns,
        "shap_mean_abs": shap_mean
    }).sort_values("shap_mean_abs", ascending=False)

    fig_shap = px.bar(
        df_shap.head(10).sort_values("shap_mean_abs", ascending=True),
        x="shap_mean_abs",
        y="feature",
        orientation="h",
        title="Importancia promedio SHAP"
    )
    fig_shap.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis_title="Impacto SHAP medio",
        yaxis_title="Variable"
    )
    st.plotly_chart(fig_shap, use_container_width=True)

except Exception:
    st.info("SHAP no está disponible. La app sigue usando importancia de variables del modelo.")

# --------------------------------------------------
# TABLA DE DETALLE DE LA PREDICCIÓN ACTUAL
# --------------------------------------------------
st.subheader("Detalle de la predicción actual")

detalle = pd.DataFrame({
    "Variable": [
        "Temperatura",
        "Precipitación",
        "Semana epidemiológica",
        "Casos lag 1",
        "Casos lag 2",
        "Casos lag 3",
        "Casos lag 4",
        "Media 3 semanas",
        "Ratio lluvia/temp",
        "Predicción final",
        "Confianza"
    ],
    "Valor": [
        round(temp, 2),
        round(precip, 2),
        semana_epi,
        casos_lag_1,
        casos_lag_2,
        casos_lag_3,
        casos_lag_4,
        round(np.mean([casos_lag_1, casos_lag_2, casos_lag_3]), 2),
        round((precip / temp if temp != 0 else 0), 4),
        pred_nivel,
        f"{confianza:.1f}%"
    ]
})
st.dataframe(detalle, use_container_width=True, hide_index=True)

# --------------------------------------------------
# RECOMENDACIÓN FINAL
# --------------------------------------------------
st.subheader("Recomendación automática")

if pred_nivel == "BAJO":
    st.success("Mantener vigilancia rutinaria, seguimiento semanal y monitoreo climático.")
elif pred_nivel == "MEDIO":
    st.warning("Reforzar monitoreo semanal, campañas preventivas y control focalizado.")
else:
    st.error("Priorizar intervención sanitaria, control vectorial y vigilancia intensiva.")

# --------------------------------------------------
# PREVIEW DEL DATASET UNIFICADO
# --------------------------------------------------
st.markdown("---")
st.subheader("Vista previa del dataset unificado")

col_f1, col_f2 = st.columns(2)

with col_f1:
    opciones_preview_anio = ["Todos"] + sorted(base_modelo["anio"].dropna().unique().tolist())
    preview_anio = st.selectbox("Filtrar preview por año", opciones_preview_anio, key="preview_anio")

with col_f2:
    opciones_orden = ["Más recientes primero", "Más antiguos primero"]
    orden_preview = st.selectbox("Orden de visualización", opciones_orden, key="orden_preview")

df_preview = base_modelo.copy()

if preview_anio != "Todos":
    df_preview = df_preview[df_preview["anio"] == preview_anio].copy()

ascendente = orden_preview == "Más antiguos primero"

df_preview = df_preview.sort_values(
    ["anio", "semana_epi", "provincia_std"],
    ascending=ascendente
)

m1, m2, m3 = st.columns(3)
m1.metric("Filas visibles", len(df_preview))
m2.metric("Provincias visibles", df_preview["provincia_std"].nunique())
m3.metric("Años visibles", df_preview["anio"].nunique())

st.dataframe(df_preview, use_container_width=True, height=350)