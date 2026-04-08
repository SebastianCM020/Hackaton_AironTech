import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import unicodedata
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


def convertir_fecha(df: pd.DataFrame, col_fecha: str) -> pd.DataFrame:
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    return df


def convertir_numerico(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def preparar_dengue(df: pd.DataFrame, fecha_col: str, provincia_col: str, canton_col: str | None):
    df = normalizar_columnas(df)
    fecha_col = fecha_col.lower()
    provincia_col = provincia_col.lower()
    canton_col = canton_col.lower() if canton_col else None

    df = df.drop_duplicates()
    df = convertir_fecha(df, fecha_col)

    df["provincia_std"] = limpiar_texto_ubicacion(df[provincia_col])

    if canton_col:
        df["canton_std"] = limpiar_texto_ubicacion(df[canton_col])
    else:
        df["canton_std"] = "SIN_CANTON"

    df = df.dropna(subset=[fecha_col, "provincia_std"])

    df["anio"] = df[fecha_col].dt.year
    df["semana_epi"] = df[fecha_col].dt.isocalendar().week.astype(int)

    # Cada fila cuenta como un caso
    dengue_semanal = (
        df.groupby(["anio", "semana_epi", "provincia_std", "canton_std"], as_index=False)
          .size()
          .rename(columns={"size": "casos_dengue"})
    )

    dengue_semanal = dengue_semanal.sort_values(
        ["provincia_std", "canton_std", "anio", "semana_epi"]
    )

    dengue_semanal["casos_previos"] = (
        dengue_semanal.groupby(["provincia_std", "canton_std"])["casos_dengue"].shift(1).fillna(0)
    )

    return dengue_semanal


def preparar_temperatura(df: pd.DataFrame, fecha_col: str, provincia_col: str, canton_col: str | None, temp_col: str):
    df = normalizar_columnas(df)
    fecha_col = fecha_col.lower()
    provincia_col = provincia_col.lower()
    canton_col = canton_col.lower() if canton_col else None
    temp_col = temp_col.lower()

    df = df.drop_duplicates()
    df = convertir_fecha(df, fecha_col)
    df = convertir_numerico(df, temp_col)

    df["provincia_std"] = limpiar_texto_ubicacion(df[provincia_col])

    if canton_col:
        df["canton_std"] = limpiar_texto_ubicacion(df[canton_col])
    else:
        df["canton_std"] = "SIN_CANTON"

    df = df.dropna(subset=[fecha_col, "provincia_std", temp_col])

    # Limpieza de valores absurdos de temperatura
    df.loc[(df[temp_col] < -10) | (df[temp_col] > 60), temp_col] = np.nan
    df = df.dropna(subset=[temp_col])

    df["anio"] = df[fecha_col].dt.year
    df["semana_epi"] = df[fecha_col].dt.isocalendar().week.astype(int)

    temp_semanal = (
        df.groupby(["anio", "semana_epi", "provincia_std", "canton_std"], as_index=False)[temp_col]
          .mean()
          .rename(columns={temp_col: "temperatura_promedio"})
    )

    return temp_semanal


def preparar_precipitacion(df: pd.DataFrame, fecha_col: str, provincia_col: str, canton_col: str | None, precip_col: str):
    df = normalizar_columnas(df)
    fecha_col = fecha_col.lower()
    provincia_col = provincia_col.lower()
    canton_col = canton_col.lower() if canton_col else None
    precip_col = precip_col.lower()

    df = df.drop_duplicates()
    df = convertir_fecha(df, fecha_col)
    df = convertir_numerico(df, precip_col)

    df["provincia_std"] = limpiar_texto_ubicacion(df[provincia_col])

    if canton_col:
        df["canton_std"] = limpiar_texto_ubicacion(df[canton_col])
    else:
        df["canton_std"] = "SIN_CANTON"

    df = df.dropna(subset=[fecha_col, "provincia_std", precip_col])

    # Limpieza de valores absurdos de precipitación
    df.loc[df[precip_col] < 0, precip_col] = np.nan
    df.loc[df[precip_col] > 1000, precip_col] = np.nan
    df = df.dropna(subset=[precip_col])

    df["anio"] = df[fecha_col].dt.year
    df["semana_epi"] = df[fecha_col].dt.isocalendar().week.astype(int)

    precip_semanal = (
        df.groupby(["anio", "semana_epi", "provincia_std", "canton_std"], as_index=False)[precip_col]
          .sum()
          .rename(columns={precip_col: "precipitacion_total"})
    )

    return precip_semanal


def unir_datasets(dengue_df, temp_df, precip_df):
    base = dengue_df.merge(
        temp_df,
        on=["anio", "semana_epi", "provincia_std", "canton_std"],
        how="left"
    ).merge(
        precip_df,
        on=["anio", "semana_epi", "provincia_std", "canton_std"],
        how="left"
    )

    return base


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


def _primera_columna_disponible(df: pd.DataFrame, candidatas: list[str]) -> str | None:
    for columna in candidatas:
        if columna in df.columns:
            return columna
    return None


def construir_dataset_unificado(
    ruta_dengue: str | Path,
    ruta_lluvia: str | Path,
    ruta_clima: str | Path,
) -> pd.DataFrame:
    ruta_dengue = Path(ruta_dengue)
    ruta_lluvia = Path(ruta_lluvia)
    ruta_clima = Path(ruta_clima)

    hojas_dengue = []
    excel_dengue = pd.ExcelFile(ruta_dengue)

    for hoja in excel_dengue.sheet_names:
        df_hoja = pd.read_excel(excel_dengue, sheet_name=hoja)
        df_hoja = normalizar_columnas(df_hoja)

        col_anio = _primera_columna_disponible(df_hoja, ["anio", "ano"])
        col_semana = _primera_columna_disponible(df_hoja, ["semana", "se"])
        col_provincia = _primera_columna_disponible(df_hoja, ["provincia", "prov_domic"])
        col_canton = _primera_columna_disponible(df_hoja, ["canton", "canton_domic"])
        col_total = _primera_columna_disponible(df_hoja, ["total"])

        if not all([col_anio, col_semana, col_provincia]):
            continue

        df_hoja["anio"] = pd.to_numeric(df_hoja[col_anio], errors="coerce")
        df_hoja["semana_epi"] = pd.to_numeric(df_hoja[col_semana], errors="coerce")
        df_hoja["provincia_std"] = limpiar_texto_ubicacion(df_hoja[col_provincia])
        df_hoja["canton_std"] = (
            limpiar_texto_ubicacion(df_hoja[col_canton])
            if col_canton
            else "SIN_CANTON"
        )
        df_hoja["casos_dengue"] = (
            pd.to_numeric(df_hoja[col_total], errors="coerce").fillna(1)
            if col_total
            else 1
        )

        hojas_dengue.append(
            df_hoja[["anio", "semana_epi", "provincia_std", "canton_std", "casos_dengue"]]
        )

    if not hojas_dengue:
        raise ValueError("No se pudieron identificar columnas validas en el archivo de dengue.")

    dengue = pd.concat(hojas_dengue, ignore_index=True)
    dengue = dengue.dropna(subset=["anio", "semana_epi", "provincia_std"])
    dengue["anio"] = dengue["anio"].astype(int)
    dengue["semana_epi"] = dengue["semana_epi"].astype(int)
    dengue["canton_std"] = "SIN_CANTON"

    dengue = (
        dengue.groupby(["anio", "semana_epi", "provincia_std", "canton_std"], as_index=False)["casos_dengue"]
        .sum()
        .sort_values(["provincia_std", "anio", "semana_epi"])
    )
    dengue["casos_previos"] = (
        dengue.groupby(["provincia_std", "canton_std"])["casos_dengue"].shift(1).fillna(0)
    )

    lluvia = pd.read_csv(ruta_lluvia)
    lluvia = normalizar_columnas(lluvia)
    lluvia["date"] = pd.to_datetime(lluvia["date"], errors="coerce")
    lluvia["rfh"] = pd.to_numeric(lluvia["rfh"], errors="coerce")
    lluvia["pcode"] = lluvia["pcode"].astype(str).str.upper()
    lluvia["codigo_provincia"] = lluvia["pcode"].str.extract(r"EC(\d{2})")
    lluvia["provincia_std"] = lluvia["codigo_provincia"].map(CODIGOS_PROVINCIA_ECUADOR)
    lluvia["anio"] = lluvia["date"].dt.year
    lluvia["semana_epi"] = lluvia["date"].dt.isocalendar().week.astype("Int64")

    lluvia = lluvia.dropna(subset=["anio", "semana_epi", "provincia_std", "rfh"])
    lluvia["canton_std"] = "SIN_CANTON"
    lluvia = (
        lluvia.groupby(["anio", "semana_epi", "provincia_std", "canton_std"], as_index=False)["rfh"]
        .sum()
        .rename(columns={"rfh": "precipitacion_total"})
    )

    clima = pd.read_csv(ruta_clima)
    clima = normalizar_columnas(clima)
    clima["date"] = pd.to_datetime(clima["date"], errors="coerce")

    if "ecuador" in clima["country_name"].astype(str).str.lower().unique():
        clima_base = clima[clima["country_name"].astype(str).str.lower() == "ecuador"].copy()
        fuente_clima = "Ecuador"
    else:
        clima_base = clima[clima["region"].astype(str).str.lower() == "south america"].copy()
        fuente_clima = "South America"

    clima_base["anio"] = clima_base["date"].dt.year
    clima_base["semana_epi"] = clima_base["date"].dt.isocalendar().week.astype("Int64")

    clima_base = (
        clima_base.groupby(["anio", "semana_epi"], as_index=False)[
            ["temperature_celsius", "vector_disease_risk_score", "precipitation_mm"]
        ]
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

    base_unificada = dengue.merge(
        lluvia,
        on=["anio", "semana_epi", "provincia_std", "canton_std"],
        how="left",
    ).merge(
        clima_base,
        on=["anio", "semana_epi"],
        how="left",
    )

    return base_unificada.sort_values(["anio", "semana_epi", "provincia_std"]).reset_index(drop=True)

# --------------------------------------------------
# TÍTULO
# --------------------------------------------------
st.markdown('<div class="main-title">🦟 Predicción de Riesgo: Dengue</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Dashboard conceptual con limpieza y consolidación de 3 datasets.</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------------------------------------
# CARGA DE ARCHIVOS
# --------------------------------------------------
st.subheader("1. Carga de datasets")

rutas_locales = {
    "dengue": Path("dataset/Datos_Dengue_MSP_Ene2021_Ago2025.xlsx"),
    "lluvia": Path("dataset/ecu-rainfall-subnat-full.csv"),
    "clima": Path("dataset/global_climate_health_impact_tracker_2015_2025.csv"),
}

if st.button("Unir automaticamente los 3 datasets locales", use_container_width=True):
    try:
        st.session_state["base_modelo_unificada"] = construir_dataset_unificado(
            rutas_locales["dengue"],
            rutas_locales["lluvia"],
            rutas_locales["clima"],
        )
        st.session_state["fuente_base_modelo"] = "automatica"
    except ImportError:
        st.error("Hace falta instalar `openpyxl` para leer el archivo Excel de dengue.")
    except Exception as exc:
        st.error(f"No se pudo construir la base unificada: {exc}")

base_modelo = st.session_state.get("base_modelo_unificada")
fuente_base_modelo = st.session_state.get("fuente_base_modelo")

col_up1, col_up2, col_up3 = st.columns(3)

with col_up1:
    archivo_dengue = st.file_uploader("Sube dataset de dengue", type=["csv", "xlsx"], key="dengue")

with col_up2:
    archivo_temp = st.file_uploader("Sube dataset de temperatura", type=["csv", "xlsx"], key="temp")

with col_up3:
    archivo_precip = st.file_uploader("Sube dataset de precipitación", type=["csv", "xlsx"], key="precip")


def leer_archivo(archivo):
    if archivo is None:
        return None
    if archivo.name.endswith(".csv"):
        return pd.read_csv(archivo)
    return pd.read_excel(archivo)


df_dengue_raw = leer_archivo(archivo_dengue)
df_temp_raw = leer_archivo(archivo_temp)
df_precip_raw = leer_archivo(archivo_precip)

# --------------------------------------------------
# MAPEO DE COLUMNAS
# --------------------------------------------------
if df_dengue_raw is not None and df_temp_raw is not None and df_precip_raw is not None:
    st.subheader("2. Mapeo de columnas")

    df_dengue_raw = normalizar_columnas(df_dengue_raw)
    df_temp_raw = normalizar_columnas(df_temp_raw)
    df_precip_raw = normalizar_columnas(df_precip_raw)

    with st.expander("Selecciona las columnas correctas para cada dataset", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("### Dengue")
            fecha_dengue = st.selectbox("Fecha dengue", df_dengue_raw.columns, key="fecha_dengue")
            prov_dengue = st.selectbox("Provincia dengue", df_dengue_raw.columns, key="prov_dengue")
            cant_dengue = st.selectbox("Cantón dengue (opcional)", [""] + list(df_dengue_raw.columns), key="cant_dengue")

        with c2:
            st.markdown("### Temperatura")
            fecha_temp = st.selectbox("Fecha temperatura", df_temp_raw.columns, key="fecha_temp")
            prov_temp = st.selectbox("Provincia temperatura", df_temp_raw.columns, key="prov_temp")
            cant_temp = st.selectbox("Cantón temperatura (opcional)", [""] + list(df_temp_raw.columns), key="cant_temp")
            col_temp = st.selectbox("Columna temperatura", df_temp_raw.columns, key="col_temp")

        with c3:
            st.markdown("### Precipitación")
            fecha_precip = st.selectbox("Fecha precipitación", df_precip_raw.columns, key="fecha_precip")
            prov_precip = st.selectbox("Provincia precipitación", df_precip_raw.columns, key="prov_precip")
            cant_precip = st.selectbox("Cantón precipitación (opcional)", [""] + list(df_precip_raw.columns), key="cant_precip")
            col_precip = st.selectbox("Columna precipitación", df_precip_raw.columns, key="col_precip")

    # --------------------------------------------------
    # PROCESO DE LIMPIEZA
    # --------------------------------------------------
    st.subheader("3. Limpieza y consolidación")

    dengue_limpio = preparar_dengue(
        df_dengue_raw, fecha_dengue, prov_dengue, cant_dengue if cant_dengue else None
    )

    temp_limpio = preparar_temperatura(
        df_temp_raw, fecha_temp, prov_temp, cant_temp if cant_temp else None, col_temp
    )

    precip_limpio = preparar_precipitacion(
        df_precip_raw, fecha_precip, prov_precip, cant_precip if cant_precip else None, col_precip
    )

    base_modelo = unir_datasets(dengue_limpio, temp_limpio, precip_limpio)
    st.session_state["base_modelo_unificada"] = base_modelo
    st.session_state["fuente_base_modelo"] = "manual"

    st.success("Datasets limpiados y unidos correctamente.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filas dengue semanal", len(dengue_limpio))
    m2.metric("Filas temperatura semanal", len(temp_limpio))
    m3.metric("Filas precipitación semanal", len(precip_limpio))
    m4.metric("Filas base final", len(base_modelo))

    st.markdown("### Vista previa de la base final")
    st.dataframe(base_modelo.head(20), use_container_width=True)

    # --------------------------------------------------
    # FILTRO PARA SIMULACIÓN
    # --------------------------------------------------
    st.subheader("4. Selección de registro para simulación")

    provincias_disponibles = sorted(base_modelo["provincia_std"].dropna().unique().tolist())
    provincia_sel = st.selectbox("Provincia", provincias_disponibles)

    df_filtrado = base_modelo[base_modelo["provincia_std"] == provincia_sel].copy()

    cantones_disponibles = sorted(df_filtrado["canton_std"].dropna().unique().tolist())
    canton_sel = st.selectbox("Cantón", cantones_disponibles)

    df_filtrado = df_filtrado[df_filtrado["canton_std"] == canton_sel].copy()

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

elif base_modelo is not None:
    st.subheader("2. Base consolidada disponible")
    mensaje_fuente = "desde los archivos locales en /dataset" if fuente_base_modelo == "automatica" else "desde la carga manual"
    st.success(f"Base unificada generada correctamente {mensaje_fuente}.")
    st.markdown("### Vista previa de la base final")
    st.dataframe(base_modelo.head(20), use_container_width=True)

    st.subheader("4. Selección de registro para simulación")

    provincias_disponibles = sorted(base_modelo["provincia_std"].dropna().unique().tolist())
    provincia_sel = st.selectbox("Provincia", provincias_disponibles, key="provincia_auto")

    df_filtrado = base_modelo[base_modelo["provincia_std"] == provincia_sel].copy()

    cantones_disponibles = sorted(df_filtrado["canton_std"].dropna().unique().tolist())
    canton_sel = st.selectbox("Cantón", cantones_disponibles, key="canton_auto")

    df_filtrado = df_filtrado[df_filtrado["canton_std"] == canton_sel].copy()

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

else:
    st.info("Sube los 3 archivos o usa el botón de carga automática para activar la consolidación de datos.")
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
st.sidebar.info(
    "Ahora el tablero puede usar valores calculados desde tus datasets limpios."
)

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
m1.metric("Riesgo total", f"{riesgo_total:.1f}%")
m2.metric("Nivel", nivel)
m3.metric("Riesgo climático", f"{riesgo_clima:.1f}%")
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

st.caption(
    "Nota: este dashboard usa una lógica heurística conceptual. "
    "El siguiente paso ideal es reemplazarla por un modelo entrenado con tus datasets reales."
)
