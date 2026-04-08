import streamlit as st
import plotly.graph_objects as go

# Configuración de la página (Tema oscuro por defecto en Streamlit)
st.set_page_config(page_title="Predicción de Riesgo: Dengue", layout="wide")

st.title("Predicción de Riesgo: Dengue")
st.markdown("---")

# --- 1. CONTROLES DE ENTRADA (Sliders) ---
st.subheader("Parámetros de Simulación")
col1, col2 = st.columns(2)

with col1:
    temp = st.slider("Temperatura (°C)", min_value=15.0, max_value=35.0, value=22.0, step=0.5)
    semana_epi = st.slider("Semana Epi.", min_value=1, max_value=52, value=12, step=1)

with col2:
    precip = st.slider("Precipitación (mm/sem)", min_value=0.0, max_value=300.0, value=50.0, step=5.0)
    casos_previos = st.slider("Casos Previos (t-1)", min_value=0, max_value=100, value=15, step=1)

# --- 2. LÓGICA DEL MODELO CONCEPTUAL ---
# Cálculo del riesgo de Clima (Máximo 50%)
# Temperatura ideal: 26-30°C
if 26 <= temp <= 30:
    riesgo_temp = 25.0
elif temp < 26:
    riesgo_temp = max(0.0, 25.0 - (26 - temp) * 2.5)
else:
    riesgo_temp = max(0.0, 25.0 - (temp - 30) * 2.5)

riesgo_lluvia = min(25.0, (precip / 200.0) * 25.0)
riesgo_clima = riesgo_temp + riesgo_lluvia

# Cálculo de Estacionalidad (Máximo 20%)
# Semanas de mayor riesgo (época lluviosa típica: semanas 5 a 20)
if 5 <= semana_epi <= 20:
    riesgo_estacionalidad = 20.0 - abs(12 - semana_epi) * 1.5
else:
    riesgo_estacionalidad = 5.0

# Cálculo de Historial de Casos (Máximo 30%)
riesgo_historial = min(30.0, (casos_previos / 50.0) * 30.0)

# Riesgo Total
riesgo_total = riesgo_clima + riesgo_estacionalidad + riesgo_historial

# Clasificación
if riesgo_total < 35:
    nivel = "BAJO"
    color = "#2ecc71" # Verde
elif riesgo_total < 70:
    nivel = "MODERADO"
    color = "#f1c40f" # Amarillo
else:
    nivel = "ALTO"
    color = "#e74c3c" # Rojo

st.markdown("---")

# --- 3. VISUALIZACIÓN DE RESULTADOS ---
col_grafico1, col_grafico2 = st.columns(2)

with col_grafico1:
    # Gráfico de Medidor (Gauge)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = riesgo_total,
        number = {'suffix': "%", 'font': {'size': 40}},
        title = {'text': f"NIVEL: {nivel}", 'font': {'size': 20, 'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 35], 'color': "rgba(46, 204, 113, 0.2)"},
                {'range': [35, 70], 'color': "rgba(241, 196, 15, 0.2)"},
                {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.2)"}
            ]
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_grafico2:
    # Gráfico de Barras Horizontales
    fig_bar = go.Figure(go.Bar(
        x=[riesgo_clima, riesgo_estacionalidad, riesgo_historial],
        y=['Clima', 'Estacionalidad', 'Historial'],
        orientation='h',
        marker=dict(color=['#3498db', '#9b59b6', '#2ecc71']),
        text=[f"{riesgo_clima:.1f}%", f"{riesgo_estacionalidad:.1f}%", f"{riesgo_historial:.1f}%"],
        textposition='auto'
    ))
    fig_bar.update_layout(
        title="Contribución al Riesgo (%)",
        xaxis=dict(range=[0, 50]),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)