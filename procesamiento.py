import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Silenciar advertencias para mantener limpio el dashboard
warnings.filterwarnings('ignore')

# 1. Configuración de la página de Streamlit
st.set_page_config(page_title="Dashboard Dengue", layout="wide", page_icon="🦠")

st.title("🦠 Dashboard Epidemiológico: Riesgo de Dengue")
st.markdown("""
Esta herramienta simula y analiza el riesgo de brotes de dengue, con un enfoque particular en 
las dinámicas de **valles interandinos y zonas costeras**.
""")

# 2. Generación de Datos (Cacheado para no recalcular en cada interacción)
@st.cache_data
def generate_synthetic_data(n_samples):
    np.random.seed(42)
    data = {
        'semana_epidemiologica': np.random.randint(1, 53, n_samples),
        'temperatura': np.random.uniform(14, 32, n_samples),
        'precipitacion': np.random.uniform(0, 500, n_samples),
        'casos_previos': np.random.randint(0, 100, n_samples)
    }
    df = pd.DataFrame(data)
    score = (df['temperatura'] * 0.5) + (df['precipitacion'] * 0.05) + (df['casos_previos'] * 0.3)
    
    def label_risk(s):
        if s < 25: return 'Bajo'
        elif s < 45: return 'Medio'
        else: return 'Alto'
    
    df['nivel_riesgo'] = score.apply(label_risk)
    return df

# Sidebar para controles interactivos
st.sidebar.header("Parámetros del Modelo")
n_samples = st.sidebar.slider("Tamaño del Dataset Sintético", 500, 5000, 1000, 100)

df = generate_synthetic_data(n_samples)

# 3. Preprocesamiento y División de Datos
X = df.drop('nivel_riesgo', axis=1)
y = df['nivel_riesgo']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

numeric_features = ['semana_epidemiologica', 'temperatura', 'precipitacion', 'casos_previos']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

# 4. Entrenamiento de Modelos
configs = [
    {'name': 'Config 1: Baseline', 'params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}},
    {'name': 'Config 2: Profundo', 'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05}},
    {'name': 'Config 3: Agresivo', 'params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.2}}
]

results = []
last_model = None

for config in configs:
    # Nota: se eliminó use_label_encoder=False para evitar el warning
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**config['params'], eval_metric='mlogloss'))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    metrics = {
        'Configuración': config['name'],
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1-Score': round(f1_score(y_test, y_pred, average='weighted'), 4)
    }
    results.append(metrics)
    last_model = clf

# 5. Interfaz del Dashboard (Columnas)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📈 Rendimiento de los Modelos (XGBoost)")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

with col2:
    st.subheader("Importancia de Variables (Modelo Agresivo)")
    model_xgb = last_model.named_steps['classifier']
    
    # Gráfico de XGBoost adaptado para Streamlit
    fig, ax = plt.subplots(figsize=(8, 5))
    xgb.plot_importance(model_xgb, importance_type='weight', ax=ax, title="Feature Importance (Weight)")
    
    # Reemplazamos los nombres genéricos "f0, f1..." por los reales
    yticklabels = [numeric_features[int(t.get_text().replace('f', ''))] for t in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels)
    st.pyplot(fig)

# 6. Explicabilidad con SHAP
st.markdown("---")
st.subheader("🧠 Análisis de Explicabilidad (SHAP)")

X_test_transformed = last_model.named_steps['preprocessor'].transform(X_test)
X_test_df = pd.DataFrame(X_test_transformed, columns=numeric_features)

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test_df)

col3, col4 = st.columns([1, 1])

with col3:
    st.write("**Impacto de las variables en la predicción global**")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # Para problemas multiclase, shap_values es una matriz 3D. Promediamos sobre las clases para el summary_plot
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    st.pyplot(fig2)

with col4:
    st.info("""
    **📚 Explicación Epidemiológica:**
    
    1. **Temperatura:** Actúa como un interruptor biológico. El *Aedes aegypti* requiere temperaturas mínimas constantes para completar su ciclo de vida. Un aumento leve puede disparar la tasa de replicación viral.
    2. **Rezagos Temporales (Casos Previos):** Indican la presencia de reservorios humanos del virus. La introducción de un caso sumado a condiciones climáticas favorables es el precursor directo de un brote.
    """)