# 🦟 Predicción de Brotes de Dengue en Ecuador

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción del Proyecto

Este proyecto fue desarrollado para el **Hackathon AironTech** con el objetivo de construir
un sistema de predicción temprana de brotes de dengue en Ecuador utilizando técnicas de
Machine Learning.

El dengue es una enfermedad viral transmitida por el mosquito *Aedes aegypti* que representa
una amenaza significativa para la salud pública en Ecuador y toda América Latina.

## 📊 Contexto: El Dengue en Ecuador

- Ecuador reporta entre **30 000 y 80 000 casos** anuales de dengue
- Las provincias de la **Costa** (Guayas, Manabí, Los Ríos) concentran más del 70% de los casos
- La **temporada lluviosa** (enero–mayo) coincide con los picos epidémicos
- Las zonas por debajo de **2 000 msnm** son endémicas del vector *Aedes aegypti*
- El cambio climático está ampliando el rango geográfico del mosquito

## 🗂️ Estructura del Proyecto

```
Hackaton_AironTech/
├── config/
│   └── config.yaml              # Hiperparámetros y rutas de datos
├── data/
│   ├── raw/                     # Datos crudos (CSV sintético)
│   └── processed/               # Datos preprocesados (train/test splits)
├── models/                      # Modelos entrenados (.pkl)
├── notebooks/
│   ├── 01_EDA.ipynb             # Análisis Exploratorio de Datos
│   └── 02_Modelado.ipynb        # Pipeline completa de modelado
├── reports/                     # Gráficos y reportes generados
├── src/
│   ├── data/
│   │   ├── generate_synthetic_data.py   # Generador de datos sintéticos
│   │   └── preprocess.py                # Limpieza y partición de datos
│   ├── features/
│   │   └── build_features.py            # Ingeniería de características
│   ├── models/
│   │   ├── train.py                     # Entrenamiento con CV
│   │   └── evaluate.py                  # Métricas y análisis SHAP
│   └── visualization/
│       └── plots.py                     # Todas las visualizaciones
├── tests/
│   ├── test_data.py             # Tests de datos y preprocesamiento
│   └── test_models.py           # Tests de modelos y métricas
├── environment.yml              # Entorno Conda
├── requirements.txt             # Dependencias pip
└── README.md
```

## ⚙️ Instalación

### Opción 1: pip

```bash
git clone https://github.com/tu-usuario/Hackaton_AironTech.git
cd Hackaton_AironTech
pip install -r requirements.txt
```

### Opción 2: Conda

```bash
conda env create -f environment.yml
conda activate dengue-prediction
```

## 🚀 Uso

### 1. Generar datos sintéticos

```bash
python src/data/generate_synthetic_data.py
```

### 2. Preprocesar datos

```bash
python src/data/preprocess.py data/raw/dengue_ecuador_sintetico.csv
```

### 3. Entrenar modelos

```bash
# Clasificación binaria con Random Forest
python src/models/train.py --model_type rf --task classification

# Regresión con XGBoost
python src/models/train.py --model_type xgb --task regression

# Clasificación con LightGBM
python src/models/train.py --model_type lgbm --task classification
```

### 4. Evaluar modelos

```bash
python src/models/evaluate.py models/rf_classification.pkl classification
```

### 5. Ejecutar notebooks

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Modelado.ipynb
```

### 6. Ejecutar pruebas

```bash
pytest tests/ -v
```

## 📐 Descripción de Variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `canton_id` | int | Identificador único del cantón |
| `canton_name` | str | Nombre del cantón |
| `provincia` | str | Provincia a la que pertenece |
| `year` | int | Año (2018–2023) |
| `semana_epidemiologica` | int | Semana epidemiológica (1–52) |
| `temperatura_promedio` | float | Temperatura media semanal (°C) |
| `precipitacion_mm` | float | Precipitación acumulada semanal (mm) |
| `casos_semana_anterior` | float | Casos de dengue la semana previa |
| `casos_acumulados_mes` | float | Casos acumulados en el mes en curso |
| `indice_aedes` | float | Índice de densidad del vector (0–1) |
| `altitud_msnm` | float | Altitud sobre el nivel del mar (m) |
| `densidad_poblacional` | float | Densidad poblacional (hab/km²) |
| `cobertura_salud` | float | Cobertura del sistema de salud (0–1) |
| `brote` | int | **Target binario**: 1 si hay brote, 0 si no |
| `nivel_riesgo` | str | **Target multiclase**: bajo / medio / alto / muy_alto |
| `casos_dengue` | float | **Target regresión**: número de casos en la semana |

## 🏆 Rendimiento de los Modelos

| Modelo | Tarea | Accuracy | F1 | ROC-AUC |
|--------|-------|----------|----|---------|
| Random Forest | Clasificación | — | — | — |
| XGBoost | Clasificación | — | — | — |
| LightGBM | Clasificación | — | — | — |

*Las métricas se completan después del entrenamiento.*

## 👥 Equipo

| Nombre | Rol |
|--------|-----|
| *(Tu nombre aquí)* | ML Engineer |
| *(Nombre compañero)* | Data Engineer |

## 📚 Referencias

1. OPS/OMS. (2023). *Situación epidemiológica del dengue en las Américas.*
2. Ministerio de Salud Pública del Ecuador. *Boletines epidemiológicos semanales.*
3. Bhatt, S. et al. (2013). The global distribution and burden of dengue. *Nature*, 496, 504–507.
4. Mordecai, E.A. et al. (2017). Detecting the impact of temperature on transmission of Zika, dengue, and chikungunya using mechanistic models. *PLOS Neglected Tropical Diseases*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

Repositorio para el hackaton Airon Tech 2026
