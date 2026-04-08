import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import unicodedata
import json
from pathlib import Path

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
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FUNCIONES DE NEGOCIO
# --------------------------------------------------
def calcular_riesgo_temperatura(temp: float) -> float:
    if 26 <= temp <= 30:
        return 25.0
    elif temp < 26:
        return max(0.0, 25.0 - (26 - temp) * 2.5)
    else:
        return max(0.0, 25.0 - (temp - 30) * 2.5)


def calcular_riesgo_lluvia(precip: float) -> float:
    return min(25.0, (precip / 200.0) * 25.0)


def calcular_riesgo_estacionalidad(semana_epi: int) -> float:
    if 1 <= semana_epi <= 52:
        distancia = abs(12 - semana_epi)
        riesgo = 20.0 - distancia * 1.3
        return max(5.0, min(20.0, riesgo))
    return 5.0


def calcular_riesgo_historial(casos_previos: int) -> float:
    return min(30.0, (casos_previos / 50.0) * 30.0)


def clasificar_riesgo(riesgo_total: float):
    if riesgo_total < 35:
        return "BAJO", "#22c55e", "Condiciones relativamente favorables, vigilancia estándar."
    elif riesgo_total < 70:
        return "MODERADO", "#f59e0b", "Condiciones de atención; conviene reforzar monitoreo y prevención."
    else:
        return "ALTO", "#ef4444", "Alta probabilidad de incremento; se recomienda respuesta preventiva intensiva."

# --------------------------------------------------
# FUNCIONES DE LIMPIEZA
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
        "SANTO DOMINGO DE LOS TSACHILAS": "SANTO DOMINGO DE LOS TSACHILAS",
        "SANTO DOMINGO DE LOS TSÁCHILAS": "SANTO DOMINGO DE LOS TSACHILAS",
        "MORONA-SANTIAGO": "MORONA SANTIAGO",
        "ZAMORA-CHINCHIPE": "ZAMORA CHINCHIPE",
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


def obtener_resumen_provincial(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["temperatura_promedio"] = pd.to_numeric(df["temperatura_promedio"], errors="coerce").fillna(28.0)
    df["precipitacion_total"] = pd.to_numeric(df["precipitacion_total"], errors="coerce").fillna(80.0)
    df["casos_previos"] = pd.to_numeric(df["casos_previos"], errors="coerce").fillna(0)
    df["semana_epi"] = pd.to_numeric(df["semana_epi"], errors="coerce").fillna(12).astype(int)
    df["casos_dengue"] = pd.to_numeric(df["casos_dengue"], errors="coerce").fillna(0)

    df["riesgo_temp"] = df["temperatura_promedio"].apply(calcular_riesgo_temperatura)
    df["riesgo_lluvia"] = df["precipitacion_total"].apply(calcular_riesgo_lluvia)
    df["riesgo_clima"] = df["riesgo_temp"] + df["riesgo_lluvia"]
    df["riesgo_estacionalidad"] = df["semana_epi"].apply(calcular_riesgo_estacionalidad)
    df["riesgo_historial"] = df["casos_previos"].apply(calcular_riesgo_historial)
    df["riesgo_total"] = (
        df["riesgo_clima"] + df["riesgo_estacionalidad"] + df["riesgo_historial"]
    ).clip(upper=100)

    df["provincia_std"] = df["provincia_std"].apply(estandarizar_nombre_provincia)

    # Último registro disponible por provincia dentro del filtro actual
    ultimo = (
        df.sort_values(["provincia_std", "anio", "semana_epi"])
          .groupby("provincia_std", as_index=False)
          .tail(1)
          .copy()
    )

    ultimo["nivel"] = ultimo["riesgo_total"].apply(lambda x: clasificar_riesgo(x)[0])

    # También sumamos casos por provincia dentro del filtro actual
    casos = (
        df.groupby("provincia_std", as_index=False)["casos_dengue"]
          .sum()
          .rename(columns={"casos_dengue": "casos_dengue_periodo"})
    )

    resumen = ultimo.merge(casos, on="provincia_std", how="left")
    return resumen


def crear_mapa_provincias(df_resumen: pd.DataFrame, geojson: dict):
    fig = px.choropleth_mapbox(
        df_resumen,
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
            "riesgo_total": ":.1f",
            "nivel": True,
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
        margin=dict(l=0, r=0, t=50, b=0),
        height=540,
        coloraxis_colorbar=dict(title="Riesgo %")
    )

    return fig

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
# CONSTRUCCIÓN AUTOMÁTICA DEL DATASET UNIFICADO
# --------------------------------------------------
def construir_dataset_unificado(
    ruta_dengue: str | Path,
    ruta_lluvia: str | Path,
    ruta_clima: str | Path,
) -> pd.DataFrame:
    ruta_dengue = Path(ruta_dengue)
    ruta_lluvia = Path(ruta_lluvia)
    ruta_clima = Path(ruta_clima)

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
            if col_total
            else 1
        )

        hojas_dengue.append(
            df_hoja[["anio", "semana_epi", "provincia_std", "casos_dengue"]]
        )

    if not hojas_dengue:
        raise ValueError("No se pudieron identificar columnas válidas en el archivo de dengue.")

    dengue = pd.concat(hojas_dengue, ignore_index=True)
    dengue = dengue.dropna(subset=["anio", "semana_epi", "provincia_std"])
    dengue["anio"] = dengue["anio"].astype(int)
    dengue["semana_epi"] = dengue["semana_epi"].astype(int)

    dengue = (
        dengue.groupby(["anio", "semana_epi", "provincia_std"], as_index=False)["casos_dengue"]
        .sum()
        .sort_values(["provincia_std", "anio", "semana_epi"])
    )

    dengue["casos_previos"] = (
        dengue.groupby(["provincia_std"])["casos_dengue"].shift(1).fillna(0)
    )

    # ----------------------------
    # LLUVIA
    # ----------------------------
    lluvia = pd.read_csv(ruta_lluvia)
    lluvia = normalizar_columnas(lluvia)
    lluvia["date"] = pd.to_datetime(lluvia["date"], errors="coerce")
    lluvia["rfh"] = pd.to_numeric(lluvia["rfh"], errors="coerce")
    lluvia["pcode"] = lluvia["pcode"].astype(str).str.upper()
    lluvia["codigo_provincia"] = lluvia["pcode"].str.extract(r"EC(\d{2})")
    lluvia["provincia_std"] = lluvia["codigo_provincia"].map(CODIGOS_PROVINCIA_ECUADOR).apply(estandarizar_nombre_provincia)
    lluvia["anio"] = lluvia["date"].dt.year
    lluvia["semana_epi"] = lluvia["date"].dt.isocalendar().week.astype("Int64")

    lluvia = lluvia.dropna(subset=["anio", "semana_epi", "provincia_std", "rfh"])
    lluvia = (
        lluvia.groupby(["anio", "semana_epi", "provincia_std"], as_index=False)["rfh"]
        .sum()
        .rename(columns={"rfh": "precipitacion_total"})
    )

    # ----------------------------
    # CLIMA
    # ----------------------------
    clima = pd.read_csv(ruta_clima)
    clima = normalizar_columnas(clima)
    clima["date"] = pd.to_datetime(clima["date"], errors="coerce")

    if "country_name" in clima.columns and "ecuador" in clima["country_name"].astype(str).str.lower().unique():
        clima_base = clima[clima["country_name"].astype(str).str.lower() == "ecuador"].copy()
        fuente_clima = "Ecuador"
    else:
        clima_base = clima[clima["region"].astype(str).str.lower() == "south america"].copy()
        fuente_clima = "South America"

    clima_base["anio"] = clima_base["date"].dt.year
    clima_base["semana_epi"] = clima_base["date"].dt.isocalendar().week.astype("Int64")

    columnas_clima = [c for c in ["temperature_celsius", "vector_disease_risk_score", "precipitation_mm"] if c in clima_base.columns]

    clima_base = (
        clima_base.groupby(["anio", "semana_epi"], as_index=False)[columnas_clima]
        .mean()
        .rename(
            columns={
                "temperature_celsius": "temperatura_promedio",
                "vector_disease_risk_score": "riesgo_vectorial_climatico",
                "precipitation_mm": "precipitacion_climatica_mm",
            }
        )
    )
    clima_base["fuente_clima"] = fuente_clima

    # ----------------------------
    # UNIÓN FINAL
    # ----------------------------
    base_unificada = dengue.merge(
        lluvia,
        on=["anio", "semana_epi", "provincia_std"],
        how="left",
    ).merge(
        clima_base,
        on=["anio", "semana_epi"],
        how="left",
    )

    return base_unificada.sort_values(["anio", "semana_epi", "provincia_std"]).reset_index(drop=True)


# --------------------------------------------------
# MAPA DE RIESGO POR PROVINCIA
# --------------------------------------------------
def construir_riesgo_por_provincia(base_modelo: pd.DataFrame) -> pd.DataFrame:
    df = base_modelo.copy()

    df["temperatura_promedio"] = pd.to_numeric(df["temperatura_promedio"], errors="coerce").fillna(28.0)
    df["precipitacion_total"] = pd.to_numeric(df["precipitacion_total"], errors="coerce").fillna(80.0)
    df["semana_epi"] = pd.to_numeric(df["semana_epi"], errors="coerce").fillna(12).astype(int)
    df["casos_previos"] = pd.to_numeric(df["casos_previos"], errors="coerce").fillna(0)

    df["riesgo_temp"] = df["temperatura_promedio"].apply(calcular_riesgo_temperatura)
    df["riesgo_lluvia"] = df["precipitacion_total"].apply(calcular_riesgo_lluvia)
    df["riesgo_clima"] = df["riesgo_temp"] + df["riesgo_lluvia"]
    df["riesgo_estacionalidad"] = df["semana_epi"].apply(calcular_riesgo_estacionalidad)
    df["riesgo_historial"] = df["casos_previos"].apply(calcular_riesgo_historial)

    df["riesgo_total"] = (
        df["riesgo_clima"] + df["riesgo_estacionalidad"] + df["riesgo_historial"]
    ).clip(upper=100)

    # Último registro por provincia
    df = (
        df.sort_values(["provincia_std", "anio", "semana_epi"])
          .groupby("provincia_std", as_index=False)
          .tail(1)
          .copy()
    )

    df["nivel"] = df["riesgo_total"].apply(lambda x: clasificar_riesgo(x)[0])
    return df


def cargar_geojson_provincias(ruta_geojson: str | Path):
    ruta_geojson = Path(ruta_geojson)
    with open(ruta_geojson, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    for feature in geojson["features"]:
        props = feature.get("properties", {})

        nombre = (
            props.get("name")
            or props.get("NAME")
            or props.get("provincia")
            or props.get("PROVINCIA")
            or props.get("DPA_DESPRO")
            or ""
        )

        feature["properties"]["provincia_std"] = normalizar_texto_simple(nombre)

    return geojson


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
        opacity=0.75,
        hover_name="provincia_std",
        hover_data={
            "riesgo_total": ":.1f",
            "casos_dengue": True,
            "temperatura_promedio": ":.1f",
            "precipitacion_total": ":.1f",
            "provincia_std": False
        },
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
        coloraxis_colorbar=dict(title="Riesgo %")
    )

    return fig

# --------------------------------------------------
# TÍTULO
# --------------------------------------------------
st.markdown('<div class="main-title">🦟 Predicción de Riesgo: Dengue</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dashboard conceptual con consolidación automática de 3 datasets y visualización provincial del riesgo.</div>',
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
# CARGA AUTOMÁTICA
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_base_modelo():
    return construir_dataset_unificado(
        rutas_locales["dengue"],
        rutas_locales["lluvia"],
        rutas_locales["clima"],
    )

base_modelo = None

try:
    base_modelo = cargar_base_modelo()
    st.success("Base unificada generada automáticamente desde los archivos locales.")
except ImportError:
    st.error("Hace falta instalar `openpyxl` para leer el archivo Excel de dengue.")
except Exception as exc:
    st.error(f"No se pudo construir la base unificada: {exc}")

if base_modelo is None or base_modelo.empty:
    st.stop()

# --------------------------------------------------
# SELECCIÓN DE PROVINCIA
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

    temp = float(fila["temperatura_promedio"]) if pd.notna(fila["temperatura_promedio"]) else 28.0
    precip = float(fila["precipitacion_total"]) if pd.notna(fila["precipitacion_total"]) else 80.0
    semana_epi = int(fila["semana_epi"])
    casos_previos = int(fila["casos_previos"]) if pd.notna(fila["casos_previos"]) else 0
else:
    temp = 28.0
    precip = 80.0
    semana_epi = 12
    casos_previos = 20

# --------------------------------------------------
# SIDEBAR - CONTROLES
# --------------------------------------------------
st.sidebar.header("Parámetros de simulación")

temp = st.sidebar.slider("Temperatura (°C)", 15.0, 35.0, float(temp), 0.5)
precip = st.sidebar.slider("Precipitación (mm/semana)", 0.0, 300.0, float(min(300.0, precip)), 5.0)
semana_epi = st.sidebar.slider("Semana epidemiológica", 1, 52, int(semana_epi), 1)
casos_previos = st.sidebar.slider("Casos previos (t-1)", 0, 100, int(min(100, casos_previos)), 1)

st.sidebar.markdown("---")
st.sidebar.info("El tablero usa la base unificada local y ya no requiere carga manual de archivos.")

# --------------------------------------------------
# CÁLCULOS
# --------------------------------------------------
riesgo_temp = calcular_riesgo_temperatura(temp)
riesgo_lluvia = calcular_riesgo_lluvia(precip)
riesgo_clima = riesgo_temp + riesgo_lluvia

riesgo_estacionalidad = calcular_riesgo_estacionalidad(semana_epi)
riesgo_historial = calcular_riesgo_historial(casos_previos)

riesgo_total = min(100.0, riesgo_clima + riesgo_estacionalidad + riesgo_historial)
nivel, color, interpretacion = clasificar_riesgo(riesgo_total)

# --------------------------------------------------
# MÉTRICAS SUPERIORES
# --------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filas filtradas", len(base_filtrada))
m2.metric("Provincias visibles", base_filtrada["provincia_std"].nunique())
m3.metric("Años disponibles", base_modelo["anio"].nunique())
m4.metric("Historial reciente", f"{riesgo_historial:.1f}%")

st.markdown(
    f"""
    <div class="metric-card">
        <b>Interpretación:</b> <span style="color:{color}; font-weight:700;">{nivel}</span><br>
        <span class="small-text">{interpretacion}</span>
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
        title={"text": f"Nivel de Riesgo: {nivel}", "font": {"size": 22, "color": color}},
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
    fig_bar = go.Figure(go.Bar(
        x=[riesgo_clima, riesgo_estacionalidad, riesgo_historial],
        y=["Clima", "Estacionalidad", "Historial"],
        orientation="h",
        marker=dict(color=["#3b82f6", "#a855f7", "#10b981"]),
        text=[
            f"{riesgo_clima:.1f}%",
            f"{riesgo_estacionalidad:.1f}%",
            f"{riesgo_historial:.1f}%"
        ],
        textposition="outside"
    ))

    fig_bar.update_layout(
        title="Contribución de factores al riesgo",
        xaxis=dict(title="Puntaje", range=[0, 50]),
        yaxis=dict(title="Factor"),
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --------------------------------------------------
# MAPA DE ECUADOR POR PROVINCIAS
# --------------------------------------------------
st.subheader("Mapa de riesgo por provincia")

try:
    with open(rutas_locales["geojson"], "r", encoding="utf-8") as f:
        geojson_ecuador = json.load(f)

    geojson_ecuador = preparar_geojson_provincias(geojson_ecuador)

    df_resumen_prov = obtener_resumen_provincial(base_filtrada)

    fig_mapa = crear_mapa_provincias(df_resumen_prov, geojson_ecuador)
    st.plotly_chart(fig_mapa, use_container_width=True)

    if not df_resumen_prov.empty:
        provincia_mayor_riesgo = df_resumen_prov.sort_values("riesgo_total", ascending=False).iloc[0]

        if anio_sel == "Todos":
            texto_periodo = "en todo el período analizado"
        else:
            texto_periodo = f"en el año {anio_sel}"

        st.info(
            f"Provincia con mayor riesgo {texto_periodo}: "
            f"**{provincia_mayor_riesgo['provincia_std']}** "
            f"con **{provincia_mayor_riesgo['riesgo_total']:.1f}%**."
        )

except FileNotFoundError:
    st.warning(
        "No se encontró el archivo `dataset/ecuador_provincias.geojson`. "
        "Agrégalo a esa carpeta para visualizar el mapa por provincias."
    )
except Exception as exc:
    st.warning(f"No se pudo generar el mapa provincial: {exc}")


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
# TABLA DE DETALLE
# --------------------------------------------------
st.subheader("Detalle del cálculo")

detalle = pd.DataFrame({
    "Factor": ["Temperatura", "Precipitación", "Clima total", "Estacionalidad", "Historial", "Riesgo total"],
    "Valor de entrada": [f"{temp} °C", f"{precip} mm/sem", "-", f"Semana {semana_epi}", f"{casos_previos} casos", "-"],
    "Puntaje": [
        round(riesgo_temp, 2),
        round(riesgo_lluvia, 2),
        round(riesgo_clima, 2),
        round(riesgo_estacionalidad, 2),
        round(riesgo_historial, 2),
        round(riesgo_total, 2)
    ]
})

st.dataframe(detalle, use_container_width=True, hide_index=True)

# --------------------------------------------------
# RECOMENDACIÓN FINAL
# --------------------------------------------------
st.subheader("Recomendación automática")

if nivel == "BAJO":
    st.success("Mantener vigilancia de rutina y seguimiento de indicadores climáticos.")
elif nivel == "MODERADO":
    st.warning("Reforzar monitoreo semanal, campañas de prevención y revisión de zonas críticas.")
else:
    st.error("Priorizar intervención preventiva, control vectorial y vigilancia intensiva.")

# --------------------------------------------------
# PREVIEW DEL DATASET UNIFICADO AL FINAL
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