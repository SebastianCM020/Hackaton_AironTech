import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

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
        # Si cada fila representa un paciente, entonces agrupamos por provincia/año/semana
        # y contamos cuántos pacientes hay
        df['Total'] = 1  # Cada registro es un caso
        
        # Agrupar casos por Provincia, Año y Semana
        weekly = df.groupby(['Provincia', 'Año', 'Semana'], as_index=False)['Total'].sum()
        
        # Ordenar cronológicamente para calcular bien el historial
        weekly = weekly.sort_values(by=['Provincia', 'Año', 'Semana'])
        
        # FEATURE ENGINEERING: Casos de la semana anterior (t-1) POR PROVINCIA
        weekly['Casos_t_1'] = weekly.groupby('Provincia')['Total'].shift(1).fillna(0)
        
        # CLIMA SINTÉTICO (en producción, conectar con API del INAMHI)
        np.random.seed(42)
        # Patrón estacional: más lluvia en semanas 5-20 (febrero-mayo)
        weekly['Precipitacion_mm'] = weekly['Semana'].apply(
            lambda x: np.random.normal(200, 40) if 5 <= x <= 20 else np.random.normal(60, 20))
        weekly['Temperatura_C'] = weekly['Semana'].apply(
            lambda x: np.random.normal(28, 1.5) if 5 <= x <= 20 else np.random.normal(24, 2))
        
        # Eliminar filas con valores nulos (primeras semanas sin historial)
        df_model = weekly.dropna().reset_index(drop=True)
        provincias_disponibles = sorted(df_model['Provincia'].unique())
        
        # Mostrar estadísticas en sidebar
        st.sidebar.info(f"📊 Datos cargados: {len(df)} casos históricos")
        st.sidebar.info(f"📍 Provincias: {len(provincias_disponibles)}")
        st.sidebar.write("Provincias:", ", ".join(provincias_disponibles[:5]) + 
                        ("..." if len(provincias_disponibles) > 5 else ""))
        
        return df_model, provincias_disponibles

    # --- 3. ENTRENAMIENTO DE IA GLOBAL ---
    @st.cache_resource
    def train_models(df_model):
        # Variables predictoras
        X = df_model[['Semana', 'Precipitacion_mm', 'Temperatura_C', 'Casos_t_1']]
        y = df_model['Total']
        
        # Escalado de características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=8, 
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        rf.fit(X_scaled, y)
        
        # Red Neuronal (Deep Learning)
        dl = MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            max_iter=500, 
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        dl.fit(X_scaled, y)
        
        # Calcular métricas de rendimiento
        y_pred_rf = rf.predict(X_scaled)
        y_pred_dl = dl.predict(X_scaled)
        
        mae_rf = mean_absolute_error(y, y_pred_rf)
        mae_dl = mean_absolute_error(y, y_pred_dl)
        r2_rf = r2_score(y, y_pred_rf)
        r2_dl = r2_score(y, y_pred_dl)
        
        return scaler, rf, dl, (mae_rf, mae_dl, r2_rf, r2_dl)

    with st.spinner('🔄 Procesando datos y entrenando modelos de IA...'):
        df_historico, lista_provincias = load_and_prepare_data(uploaded_file)
        
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

    # --- 4. INTERFAZ DE SIMULACIÓN Y PREDICCIÓN ---
    st.markdown("---")
    st.subheader("2. Panel de Simulación Provincial")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        provincia_seleccionada = st.selectbox("📍 Seleccione la Provincia:", lista_provincias)
    
    # Obtener datos históricos de la provincia seleccionada para valores por defecto
    datos_provincia = df_historico[df_historico['Provincia'] == provincia_seleccionada]
    ultimos_casos = datos_provincia['Total'].tail(3).mean() if len(datos_provincia) > 0 else 50
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        semana_in = st.number_input("📅 Semana Epidemiológica", min_value=1, max_value=52, value=15)
    with col2:
        temp_in = st.number_input("🌡️ Temperatura (°C)", min_value=10.0, max_value=40.0, value=25.0, 
                                   help="Temperatura promedio de la semana")
    with col3:
        precip_in = st.number_input("☔ Precipitación (mm)", min_value=0.0, max_value=500.0, value=80.0,
                                     help="Precipitación acumulada en mm")
    with col4:
        casos_prev_in = st.number_input("📊 Casos Semana Previa", min_value=0, max_value=500, 
                                        value=int(ultimos_casos), step=5)

    # --- 5. INFERENCIA HÍBRIDA ---
    input_data = pd.DataFrame({
        'Semana': [semana_in],
        'Precipitacion_mm': [precip_in],
        'Temperatura_C': [temp_in],
        'Casos_t_1': [casos_prev_in]
    })

    input_scaled = scaler.transform(input_data)

    pred_rf = max(0, rf_model.predict(input_scaled)[0])
    pred_dl = max(0, dl_model.predict(input_scaled)[0])
    
    # Ensemble con pesos (puedes ajustar los pesos)
    peso_rf = 0.6
    peso_dl = 0.4
    pred_final = int(pred_rf * peso_rf + pred_dl * peso_dl)

    st.markdown("---")
    st.subheader(f"3. 🔔 Alerta Temprana para {provincia_seleccionada}")
    
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

    # Gráfico de Gauge (Medidor)
    fig = go.Figure()
    
    # Determinar límites para el gauge
    max_valor = max(100, pred_final + 50, casos_prev_in * 2)
    
    fig.add_trace(go.Indicator(
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
                'value': casos_prev_in,
                'title': {"text": "Base"}
            }
        }
    ))
    
    fig.update_layout(height=350, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 6. INTERPRETACIÓN Y RECOMENDACIONES ---
    st.markdown("---")
    st.subheader("4. 📋 Recomendaciones")
    
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
        ⚠️ **T省NDENCIA AL ALZA**
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
        st.dataframe(datos_hist_prov[['Año', 'Semana', 'Total', 'Casos_t_1', 'Temperatura_C', 'Precipitacion_mm']])