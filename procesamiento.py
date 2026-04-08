import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Configuración de semilla para reproducibilidad
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    """
    Genera un dataset sintético epidemiológico para el Dengue en Ecuador.
    En zonas andinas, la temperatura y los rezagos (casos previos) son críticos 
    debido a que el mosquito Aedes aegypti tiene límites térmicos de supervivencia.
    """
    data = {
        'semana_epidemiologica': np.random.randint(1, 53, n_samples),
        'temperatura': np.random.uniform(14, 32, n_samples), # Rango común en valles interandinos y costa
        'precipitacion': np.random.uniform(0, 500, n_samples), # mm mensuales
        'casos_previos': np.random.randint(0, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Lógica sintética para nivel de riesgo (Simulando dinámica epidemiológica)
    # El riesgo aumenta con temp > 20°C y precipitaciones moderadas
    score = (df['temperatura'] * 0.5) + (df['precipitacion'] * 0.05) + (df['casos_previos'] * 0.3)
    
    def label_risk(s):
        if s < 25: return 'Bajo'
        elif s < 45: return 'Medio'
        else: return 'Alto'
    
    df['nivel_riesgo'] = score.apply(label_risk)
    return df

# 1. Generar Dataset
df = generate_synthetic_data(1000)
X = df.drop('nivel_riesgo', axis=1)
y = df['nivel_riesgo']

# Codificar la variable objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Clases: 0: Alto, 1: Bajo, 2: Medio (según orden alfabético de LabelEncoder)

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Definir Pipeline de Preprocesamiento
numeric_features = ['semana_epidemiologica', 'temperatura', 'precipitacion', 'casos_previos']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 4. Implementar Modelos XGBoost con 3 configuraciones
configs = [
    {'name': 'Config 1: Baseline', 'params': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}},
    {'name': 'Config 2: Profundo', 'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05}},
    {'name': 'Config 3: Agresivo', 'params': {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.2}}
]

results = []

print("--- Evaluación de Modelos XGBoost ---")
for config in configs:
    # Crear pipeline con el modelo
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**config['params'], use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    # Entrenar
    clf.fit(X_train, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test)
    
    # Métricas
    metrics = {
        'Configuración': config['name'],
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }
    results.append(metrics)
    
    # Guardar el último modelo para SHAP
    last_model = clf

# 5. Tabla Comparativa
df_results = pd.DataFrame(results)
print("\nTabla Comparativa de Métricas:")
print(df_results.to_markdown(index=False))

# 6. Explicabilidad con SHAP
print("\n--- Generando Análisis de Explicabilidad (SHAP) ---")
# Extraer el modelo entrenado del pipeline y los datos transformados
model_xgb = last_model.named_steps['classifier']
X_test_transformed = last_model.named_steps['preprocessor'].transform(X_test)
X_test_df = pd.DataFrame(X_test_transformed, columns=numeric_features)

explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_test_df)

# Visualización de importancia de variables (Feature Importance tradicional como alternativa rápida)
plt.figure(figsize=(10, 6))
xgb.plot_importance(model_xgb, importance_type='weight')
plt.title('Importancia de Variables (XGBoost Weight)')
plt.savefig('feature_importance.png')
print("Gráfico de importancia guardado como 'feature_importance.png'")

# Comentario experto:
"""
EXPLICACIÓN EPIDEMIOLÓGICA (ZONAS ANDINAS):
1. Temperatura: En los valles interandinos de Ecuador (ej. valles de Quito, Cuenca o Loja), la temperatura 
   actúa como un interruptor biológico. El Aedes aegypti requiere temperaturas mínimas constantes (aprox. 15-17°C) 
   para completar su ciclo de vida. Un aumento leve puede disparar la tasa de picaduras y replicación viral.
2. Rezagos Temporales (Casos Previos): El dengue no aparece de la nada. Los casos previos indican la 
   presencia de reservorios humanos del virus. En zonas altas, donde la densidad del mosquito es menor que 
   en la costa, la introducción de un caso importado sumado a condiciones climáticas favorables es el 
   precursor directo de un brote.
"""

print("\nScript ejecutado con éxito. Listo para copiar a Google Colab.")