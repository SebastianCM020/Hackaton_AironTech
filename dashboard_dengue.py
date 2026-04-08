import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Predicción Dengue Nacional", layout="wide")
st.title("🦟 Sistema Nacional de Predicción de Dengue")
st.markdown("Modelo Híbrido (Random Forest + Red Neuronal) para todas las provincias del Ecuador.")

# --- 1. CARGA DE ARCHIVO POR INTERFAZ ---
st.sidebar.header("1. Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV de casos", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Por favor, arrastra y suelta tu archivo CSV en el panel izquierdo para comenzar.")
else:
    # --- 2. PREPARACIÓN DE DATOS (TODAS LAS PROVINCIAS) ---
    @st.cache_data
    def load_and_prepare_data(file):
        # Leer el CSV subido
        df = pd.read_csv(file)
        
        # Renombrar columnas para que coincidan con el código
        df = df.rename(columns={
            'adm_1_name': 'Provincia',
            'year': 'Año',
            'semana_epidemiologica': 'Semana'
        })
        
        # Crear la columna 'Total' (cada fila es un caso, entonces sumamos 1 por cada caso)
        df['Total'] = 1
        
        # Agrupar casos por Provincia, Año y Semana
        weekly = df.groupby(['Provincia', 'Año', 'Semana'], as_index=False)['Total'].sum()
        
        # Ordenar cronológicamente
        weekly = weekly.sort_values(by=['Provincia', 'Año', 'Semana'])
        
        # FEATURE ENGINEERING: Casos de la semana anterior (t-1) POR PROVINCIA
        weekly['Casos_t_1'] = weekly.groupby('Provincia')['Total'].shift(1).fillna(0)
        
        # CLIMA SINTÉTICO (en producción, conectar con API del INAMHI)
        np.random.seed(42)
        weekly['Precipitacion_mm'] = weekly['Semana'].apply(
            lambda x: np.random.normal(200, 40) if 5 <= x <= 20 else np.random.normal(60, 20))
        weekly['Temperatura_C'] = weekly['Semana'].apply(
            lambda x: np.random.normal(28, 1.5) if 5 <= x <= 20 else np.random.normal(24, 2))
        
        # Eliminar filas con valores nulos
        df_model = weekly.dropna().reset_index(drop=True)
        provincias_disponibles = sorted(df_model['Provincia'].unique())
        
        # Calcular casos totales por provincia para el mapa de calor
        casos_totales_provincia = df_model.groupby('Provincia')['Total'].sum().reset_index()
        casos_totales_provincia.columns = ['Provincia', 'Casos_Historicos_Total']
        
        # Mostrar estadísticas en sidebar
        st.sidebar.info(f"📊 Datos cargados: {len(df)} casos históricos")
        st.sidebar.info(f"📍 Provincias: {len(provincias_disponibles)}")
        st.sidebar.write("Provincias:", ", ".join(provincias_disponibles[:5]) + 
                        ("..." if len(provincias_disponibles) > 5 else ""))
        
        return df_model, provincias_disponibles, casos_totales_provincia

    # --- 3. ENTRENAMIENTO DE IA GLOBAL ---
    @st.cache_resource
    def train_models(df_model):
        X = df_model[['Semana', 'Precipitacion_mm', 'Temperatura_C', 'Casos_t_1']]
        y = df_model['Total']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=8, 
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf.fit(X_scaled, y)
        
        dl = MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            max_iter=500, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        dl.fit(X_scaled, y)
        
        # Métricas
        y_pred_rf = rf.predict(X_scaled)
        y_pred_dl = dl.predict(X_scaled)
        
        mae_rf = mean_absolute_error(y, y_pred_rf)
        mae_dl = mean_absolute_error(y, y_pred_dl)
        r2_rf = r2_score(y, y_pred_rf)
        r2_dl = r2_score(y, y_pred_dl)
        
        return scaler, rf, dl, (mae_rf, mae_dl, r2_rf, r2_dl)

    with st.spinner('🔄 Procesando datos y entrenando modelos de IA...'):
        df_historico, lista_provincias, casos_totales = load_and_prepare_data(uploaded_file)
        
        if len(df_historico) < 10:
            st.error("❌ No hay suficientes datos históricos para entrenar los modelos.")
            st.stop()
            
        scaler, rf_model, dl_model, metrics = train_models(df_historico)
    
    st.sidebar.success(f"✅ IA entrenada exitosamente!")
    
    # Mostrar métricas del modelo
    with st.sidebar.expander("📈 Rendimiento del Modelo"):
        st.metric("Random Forest - MAE", f"{metrics[0]:.2f} casos")
        st.metric("Red Neuronal - MAE", f"{metrics[1]:.2f} casos")
        st.metric("Random Forest - R²", f"{metrics[2]:.3f}")
        st.metric("Red Neuronal - R²", f"{metrics[3]:.3f}")

    # --- 4. MAPA DE CALOR COROPLÉTICO DE ECUADOR (MEJORADO) ---
    st.markdown("---")
    st.subheader("🗺️ Mapa de Calor Epidemiológico del Ecuador")
    st.markdown("**Casos totales históricos por provincia - Intensidad de color según número de casos**")
    
    # GeoJSON simplificado de Ecuador con polígonos reales de provincias
    # Este GeoJSON contiene las coordenadas aproximadas de cada provincia
    geojson_ecuador = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"PROVINCIA": "Azuay"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.5, -3.5], [-78.5, -3.5], [-78.5, -2.5], [-79.5, -2.5], [-79.5, -3.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Bolívar"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.5, -2.0], [-78.8, -2.0], [-78.8, -1.3], [-79.5, -1.3], [-79.5, -2.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Cañar"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.2, -2.8], [-78.8, -2.8], [-78.8, -2.2], [-79.2, -2.2], [-79.2, -2.8]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Carchi"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.5, 0.5], [-77.5, 0.5], [-77.5, 1.0], [-78.5, 1.0], [-78.5, 0.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Chimborazo"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.0, -2.2], [-78.2, -2.2], [-78.2, -1.5], [-79.0, -1.5], [-79.0, -2.2]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Cotopaxi"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.0, -1.2], [-78.5, -1.2], [-78.5, -0.8], [-79.0, -0.8], [-79.0, -1.2]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "El Oro"}, "geometry": {"type": "Polygon", "coordinates": [[[-80.2, -3.8], [-79.5, -3.8], [-79.5, -3.0], [-80.2, -3.0], [-80.2, -3.8]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Esmeraldas"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.8, 0.5], [-78.8, 0.5], [-78.8, 1.2], [-79.8, 1.2], [-79.8, 0.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Galápagos"}, "geometry": {"type": "Polygon", "coordinates": [[[-91.0, -1.0], [-89.0, -1.0], [-89.0, 0.5], [-91.0, 0.5], [-91.0, -1.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Guayas"}, "geometry": {"type": "Polygon", "coordinates": [[[-80.5, -2.5], [-79.5, -2.5], [-79.5, -1.8], [-80.5, -1.8], [-80.5, -2.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Imbabura"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.5, 0.2], [-78.0, 0.2], [-78.0, 0.6], [-78.5, 0.6], [-78.5, 0.2]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Loja"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.5, -4.5], [-78.8, -4.5], [-78.8, -3.8], [-79.5, -3.8], [-79.5, -4.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Los Ríos"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.8, -1.8], [-79.0, -1.8], [-79.0, -1.2], [-79.8, -1.2], [-79.8, -1.8]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Manabí"}, "geometry": {"type": "Polygon", "coordinates": [[[-80.8, -1.5], [-79.5, -1.5], [-79.5, -0.5], [-80.8, -0.5], [-80.8, -1.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Morona Santiago"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.2, -3.0], [-77.5, -3.0], [-77.5, -2.0], [-78.2, -2.0], [-78.2, -3.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Napo"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.0, -1.0], [-77.5, -1.0], [-77.5, -0.5], [-78.0, -0.5], [-78.0, -1.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Orellana"}, "geometry": {"type": "Polygon", "coordinates": [[[-77.0, -1.0], [-76.5, -1.0], [-76.5, -0.5], [-77.0, -0.5], [-77.0, -1.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Pastaza"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.0, -2.0], [-77.0, -2.0], [-77.0, -1.0], [-78.0, -1.0], [-78.0, -2.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Pichincha"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.8, -0.5], [-78.2, -0.5], [-78.2, 0.2], [-78.8, 0.2], [-78.8, -0.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Santa Elena"}, "geometry": {"type": "Polygon", "coordinates": [[[-81.0, -2.5], [-80.5, -2.5], [-80.5, -1.8], [-81.0, -1.8], [-81.0, -2.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Santo Domingo de los Tsáchilas"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.5, -0.5], [-78.8, -0.5], [-78.8, 0.0], [-79.5, 0.0], [-79.5, -0.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Sucumbíos"}, "geometry": {"type": "Polygon", "coordinates": [[[-76.5, 0.0], [-75.8, 0.0], [-75.8, 0.5], [-76.5, 0.5], [-76.5, 0.0]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Tungurahua"}, "geometry": {"type": "Polygon", "coordinates": [[[-78.8, -1.5], [-78.2, -1.5], [-78.2, -1.0], [-78.8, -1.0], [-78.8, -1.5]]]}},
            {"type": "Feature", "properties": {"PROVINCIA": "Zamora Chinchipe"}, "geometry": {"type": "Polygon", "coordinates": [[[-79.0, -4.2], [-78.5, -4.2], [-78.5, -3.5], [-79.0, -3.5], [-79.0, -4.2]]]}}
        ]
    }
    
    # Normalizar los nombres de provincias en los datos para que coincidan con el GeoJSON
    casos_totales['Provincia_Normalizada'] = casos_totales['Provincia'].str.strip()
    
    # Crear mapa coroplético (choropleth) - ESTE ES EL MAPA DE CALOR VERDADERO
    fig_choropleth = px.choropleth(
        casos_totales,
        geojson=geojson_ecuador,
        locations='Provincia_Normalizada',
        color='Casos_Historicos_Total',
        featureidkey="properties.PROVINCIA",
        color_continuous_scale=[
            (0.0, '#ffffb2'),    # Amarillo muy claro
            (0.2, '#fed976'),
            (0.4, '#feb24c'),
            (0.6, '#fd8d3c'),
            (0.8, '#f03b20'),
            (1.0, '#bd0026')     # Rojo oscuro
        ],
        range_color=(0, casos_totales['Casos_Historicos_Total'].max()),
        labels={'Casos_Historicos_Total': 'Casos Totales'},
        title="<b>INTENSIDAD DE DENGUE POR PROVINCIA</b>"
    )
    
    # Personalizar la apariencia del mapa
    fig_choropleth.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator"
    )
    
    fig_choropleth.update_layout(
        height=600,
        margin={"r":0, "t":50, "l":0, "b":0},
        coloraxis_colorbar={
            "title": "CASOS<br>TOTALES",
            "thickness": 25,
            "len": 0.7,
            "tickformat": ",.0f",
            "title_font": {"size": 12, "weight": "bold"}
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Mejorar el hover tooltip
    fig_choropleth.update_traces(
        hovertemplate="<b>%{location}</b><br>" +
                      "📊 Casos Totales: <b>%{z:,.0f}</b><br>" +
                      "<extra></extra>",
        marker_line_width=0.5,
        marker_line_color='darkgray',
        selector=dict(type='choropleth')
    )
    
    # Agregar anotación con el total nacional
    total_nacional = casos_totales['Casos_Historicos_Total'].sum()
    fig_choropleth.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Total Nacional:</b> {total_nacional:,.0f} casos",
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    st.plotly_chart(fig_choropleth, use_container_width=True)
    
    # Agregar explicación de la escala de colores
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("🟡 **Muy Bajo**")
    with col2:
        st.markdown("🟠 **Bajo**")
    with col3:
        st.markdown("🟠🟠 **Medio**")
    with col4:
        st.markdown("🔴 **Alto**")
    with col5:
        st.markdown("🔴🔴 **Muy Alto**")
    
    # Mostrar tabla de casos totales por provincia
    with st.expander("📊 Ver tabla completa de casos totales por provincia"):
        # Agregar columna de porcentaje
        casos_totales['% del Total'] = (casos_totales['Casos_Historicos_Total'] / total_nacional * 100).round(2)
        casos_totales['% del Total'] = casos_totales['% del Total'].apply(lambda x: f"{x}%")
        
        st.dataframe(
            casos_totales[['Provincia', 'Casos_Historicos_Total', '% del Total']].sort_values('Casos_Historicos_Total', ascending=False),
            use_container_width=True,
            column_config={
                "Provincia": "Provincia",
                "Casos_Historicos_Total": st.column_config.NumberColumn("Casos Totales", format="%.0f"),
                "% del Total": "Porcentaje"
            }
        )
    
    # --- 5. GRÁFICO DE BARRAS COMPLEMENTARIO ---
    st.subheader("📊 Top 10 Provincias con Mayor Incidencia")
    
    top_provincias = casos_totales.nlargest(10, 'Casos_Historicos_Total')
    
    fig_barras = px.bar(
        top_provincias,
        x='Casos_Historicos_Total',
        y='Provincia',
        orientation='h',
        color='Casos_Historicos_Total',
        color_continuous_scale='Reds',
        title="Provincias con Más Casos Acumulados",
        labels={'Casos_Historicos_Total': 'Número de Casos', 'Provincia': ''},
        text='Casos_Historicos_Total'
    )
    
    fig_barras.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Casos: %{x:,.0f}<extra></extra>'
    )
    
    fig_barras.update_layout(
        height=500,
        xaxis_title="Número de Casos",
        yaxis_title="",
        showlegend=False,
        xaxis=dict(tickformat=",.0f")
    )
    
    st.plotly_chart(fig_barras, use_container_width=True)
    
    # --- 6. INTERFAZ DE SIMULACIÓN Y PREDICCIÓN ---
    st.markdown("---")
    st.subheader("🎯 Panel de Simulación Provincial")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        provincia_seleccionada = st.selectbox("📍 Seleccione la Provincia:", lista_provincias)
    
    # Mostrar estadísticas rápidas de la provincia seleccionada
    casos_provincia = casos_totales[casos_totales['Provincia'] == provincia_seleccionada]['Casos_Historicos_Total'].values[0]
    st.info(f"📌 **{provincia_seleccionada}** - Histórico total: **{casos_provincia:,.0f} casos**")
    
    # Obtener datos históricos de la provincia seleccionada
    datos_provincia = df_historico[df_historico['Provincia'] == provincia_seleccionada]
    ultimos_casos = datos_provincia['Total'].tail(3).mean() if len(datos_provincia) > 0 else 50
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        semana_in = st.number_input("📅 Semana Epidemiológica", min_value=1, max_value=52, value=15)
    with col2:
        temp_in = st.number_input("🌡️ Temperatura (°C)", min_value=10.0, max_value=40.0, value=25.0)
    with col3:
        precip_in = st.number_input("☔ Precipitación (mm)", min_value=0.0, max_value=500.0, value=80.0)
    with col4:
        casos_prev_in = st.number_input("📊 Casos Semana Previa", min_value=0, max_value=500, 
                                        value=int(ultimos_casos), step=5)

    # --- 7. INFERENCIA HÍBRIDA ---
    input_data = pd.DataFrame({
        'Semana': [semana_in],
        'Precipitacion_mm': [precip_in],
        'Temperatura_C': [temp_in],
        'Casos_t_1': [casos_prev_in]
    })

    input_scaled = scaler.transform(input_data)

    pred_rf = max(0, rf_model.predict(input_scaled)[0])
    pred_dl = max(0, dl_model.predict(input_scaled)[0])
    
    # Ensemble con pesos
    peso_rf = 0.6
    peso_dl = 0.4
    pred_final = int(pred_rf * peso_rf + pred_dl * peso_dl)

    st.markdown("---")
    st.subheader(f"🔔 Alerta Temprana para {provincia_seleccionada}")
    
    # Determinar nivel de alerta
    incremento = ((pred_final - casos_prev_in) / casos_prev_in * 100) if casos_prev_in > 0 else 100
    
    if incremento > 50:
        nivel_alerta = "🔴 ALTA"
        color_alerta = "#e74c3c"
    elif incremento > 20:
        nivel_alerta = "🟡 MEDIA"
        color_alerta = "#f1c40f"
    else:
        nivel_alerta = "🟢 BAJA"
        color_alerta = "#2ecc71"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="📊 Predicción de Casos", 
                  value=f"{pred_final} casos", 
                  delta=f"{pred_final - casos_prev_in} vs semana anterior",
                  delta_color="inverse")
    
    with col2:
        st.metric(label="⚠️ Nivel de Alerta", value=nivel_alerta, delta=f"{incremento:.1f}% incremento")
    
    with col3:
        st.metric(label="🤖 Modelo", value="Ensemble", delta="RF + DL")

    # Gráfico de Gauge
    fig_gauge = go.Figure()
    
    max_valor = max(100, pred_final + 50, casos_prev_in * 2)
    
    fig_gauge.add_trace(go.Indicator(
        mode="number+gauge",
        value=pred_final,
        domain={'x': [0.1, 0.9], 'y': [0, 1]},
        title={'text': "Nivel de Riesgo Epidemiológico"},
        gauge={
            'axis': {'range': [0, max_valor], 'tickwidth': 1},
            'bar': {'color': color_alerta},
            'steps': [
                {'range': [0, casos_prev_in * 0.8], 'color': "rgba(46, 204, 113, 0.2)"},
                {'range': [casos_prev_in * 0.8, casos_prev_in * 1.5], 'color': "rgba(241, 196, 15, 0.2)"},
                {'range': [casos_prev_in * 1.5, max_valor], 'color': "rgba(231, 76, 60, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': casos_prev_in
            }
        }
    ))
    
    fig_gauge.update_layout(height=350, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # --- 8. INTERPRETACIÓN Y RECOMENDACIONES ---
    st.markdown("---")
    st.subheader("📋 Recomendaciones")
    
    if pred_final > casos_prev_in * 1.5:
        st.error("""
        🚨 **ALERTA EPIDEMIOLÓGICA** 
        - Activar comités de respuesta inmediata
        - Intensificar fumigación en zonas críticas
        - Activar campañas de eliminación de criaderos
        - Preparar unidades de salud para posible aumento de casos
        """)
    elif pred_final > casos_prev_in * 1.2:
        st.warning("""
        ⚠️ **TENDENCIA AL ALZA**
        - Reforzar vigilancia epidemiológica
        - Realizar jornadas de limpieza comunitaria
        - Mantener monitoreo de casos febriles
        """)
    else:
        st.info("""
        ℹ️ **SITUACIÓN CONTROLADA**
        - Continuar con vigilancia rutinaria
        - Mantener programas de prevención
        - Reportar casos sospechosos oportunamente
        """)
    
    # Mostrar datos históricos de la provincia
    with st.expander("📜 Ver datos históricos de la provincia"):
        datos_hist_prov = df_historico[df_historico['Provincia'] == provincia_seleccionada].tail(10)
        if len(datos_hist_prov) > 0:
            st.dataframe(datos_hist_prov[['Año', 'Semana', 'Total', 'Casos_t_1', 'Temperatura_C', 'Precipitacion_mm']])
        else:
            st.info("No hay datos históricos suficientes para esta provincia")