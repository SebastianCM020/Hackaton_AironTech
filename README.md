# Hackaton_AironTech
Repositorio para el hackaton Airon Tech 2026

## 🛠️ Instalación y Configuración

Para ejecutar este proyecto de forma local, necesitas tener instalado **Python 3.x**. 

Las librerías integradas (`json`, `pathlib`, `unicodedata`) ya vienen con Python. Solo necesitas instalar las dependencias externas.

### 1. Instalar requerimientos
Abre tu terminal en la carpeta del proyecto y ejecuta el siguiente comando para instalar todas las librerías necesarias:

```bash
pip install streamlit
pip install plotly
pip install pandas
pip install numpy
pip install scikit-learn
pip install openpyxl 
```
### 2. Iniciar la Interfaz
```bash
streamlit run dashboard_dengue.py
```

### 3. Variables utilizadas para el entrenamiento
## Selección de características

Para entrenar el modelo de predicción de dengue se utilizaron variables derivadas del historial de casos y de condiciones climáticas. Estas variables permiten capturar patrones temporales, estacionales y ambientales que influyen en la propagación del dengue.

### Variables utilizadas

**casos_lag_1**  
Cantidad de casos de dengue registrados en la semana anterior (t-1). Esta variable suele ser una de las más importantes porque el dengue presenta un comportamiento autoregresivo: si hubo casos la semana pasada, es probable que continúen en la siguiente.

**media_3_sem**  
Promedio de casos registrados durante las últimas tres semanas. Esta variable permite suavizar la variabilidad semanal y ayuda al modelo a identificar tendencias recientes en la evolución de los casos.

**casos_lag_2**  
Número de casos registrados hace dos semanas (t-2). Esta variable permite capturar el efecto de continuidad en la transmisión de la enfermedad.

**interaccion_hist_clima**  
Variable que representa la interacción entre el historial de casos y las condiciones climáticas. Permite modelar situaciones donde factores ambientales pueden amplificar brotes existentes.

**casos_lag_3**  
Número de casos registrados hace tres semanas (t-3). Esta variable amplía la memoria temporal del modelo para detectar patrones más prolongados.

**casos_lag_4**  
Número de casos registrados hace cuatro semanas (t-4). Es relevante porque el ciclo biológico del mosquito transmisor del dengue suele desarrollarse dentro de este rango temporal.

**media_4_sem_lluvia**  
Promedio de precipitación durante las últimas cuatro semanas. La acumulación de lluvias favorece la formación de criaderos del mosquito.

**interaccion_temp_lluvia**  
Interacción entre temperatura y precipitación. Estas dos variables combinadas representan condiciones climáticas favorables para la reproducción del mosquito.

**precipitacion_total**  
Cantidad total de lluvia registrada en la semana actual. Influye directamente en la disponibilidad de criaderos temporales.

**semana_sin**  
Transformación senoidal de la semana epidemiológica utilizada para representar la estacionalidad del dengue a lo largo del año. Esta transformación permite capturar ciclos anuales sin generar discontinuidades entre la última y la primera semana del año.

### Conclusión

El modelo utiliza una combinación de variables históricas, climáticas y estacionales para mejorar la capacidad de predicción de los brotes de dengue. Esta estrategia permite capturar patrones temporales, efectos ambientales y dinámicas epidemiológicas que influyen en la aparición y evolución de la enfermedad.

