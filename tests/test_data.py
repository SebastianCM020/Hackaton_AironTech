"""
Tests para generación de datos sintéticos y preprocesamiento.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_synthetic_data import generate_synthetic_data
from src.data.preprocess import load_data, preprocess_data, split_data
from src.features.build_features import build_all_features


EXPECTED_COLUMNS = [
    "canton_id", "canton_name", "provincia", "year",
    "semana_epidemiologica", "temperatura_promedio", "precipitacion_mm",
    "casos_semana_anterior", "casos_acumulados_mes", "indice_aedes",
    "altitud_msnm", "densidad_poblacional", "cobertura_salud",
    "brote", "nivel_riesgo", "casos_dengue",
]


@pytest.fixture(scope="module")
def synthetic_df():
    """Dataset sintético generado una sola vez para todos los tests."""
    return generate_synthetic_data()


def test_generate_synthetic_data_shape(synthetic_df):
    """El dataset debe tener al menos 60 000 filas.

    Estimación: 215 cantones × 6 años (2018-2023) × 52 semanas ≈ 67 080 registros.
    """
    assert synthetic_df.shape[0] > 60_000, (
        f"Se esperaban >60 000 filas, se obtuvieron {synthetic_df.shape[0]}"
    )
    assert synthetic_df.shape[1] == len(EXPECTED_COLUMNS), (
        f"Se esperaban {len(EXPECTED_COLUMNS)} columnas, hay {synthetic_df.shape[1]}"
    )


def test_generate_synthetic_data_columns(synthetic_df):
    """Todas las columnas esperadas deben estar presentes."""
    for col in EXPECTED_COLUMNS:
        assert col in synthetic_df.columns, f"Columna faltante: {col}"


def test_generate_synthetic_data_values(synthetic_df):
    """Los valores deben estar en rangos razonables."""
    assert synthetic_df["temperatura_promedio"].between(0, 45).all()
    assert synthetic_df["precipitacion_mm"].ge(0).all()
    assert synthetic_df["indice_aedes"].between(0, 1).all()
    assert synthetic_df["cobertura_salud"].between(0, 1).all()
    assert synthetic_df["brote"].isin([0, 1]).all()
    assert synthetic_df["nivel_riesgo"].isin(["bajo", "medio", "alto", "muy_alto"]).all()
    assert synthetic_df["semana_epidemiologica"].between(1, 52).all()
    assert synthetic_df["year"].between(2018, 2023).all()


def test_preprocess_handles_missing_values(synthetic_df):
    """El preprocesamiento debe eliminar NaN en columnas numéricas."""
    # Introducir NaN artificialmente
    df_dirty = synthetic_df.copy()
    rng = np.random.default_rng(0)
    nan_idx = rng.choice(len(df_dirty), size=200, replace=False)
    df_dirty.loc[nan_idx, "temperatura_promedio"] = np.nan
    df_dirty.loc[nan_idx[:50], "precipitacion_mm"] = np.nan

    df_clean = preprocess_data(df_dirty)
    assert df_clean["temperatura_promedio"].isna().sum() == 0
    assert df_clean["precipitacion_mm"].isna().sum() == 0


def test_preprocess_adds_encoded_columns(synthetic_df):
    """El preprocesamiento debe agregar columnas codificadas."""
    df_proc = preprocess_data(synthetic_df)
    assert "provincia_encoded" in df_proc.columns
    assert "nivel_riesgo_encoded" in df_proc.columns


def test_feature_engineering_adds_columns(synthetic_df):
    """La ingeniería de características debe agregar columnas de lag y rolling."""
    # build_all_features necesita canton_id, year, semana_epidemiologica, casos_dengue
    df_feat = build_all_features(synthetic_df.copy())

    new_cols = set(df_feat.columns) - set(synthetic_df.columns)
    assert len(new_cols) > 0, "No se agregaron columnas nuevas"

    # Verificar columnas específicas
    assert "semana_sin" in df_feat.columns
    assert "semana_cos" in df_feat.columns
    assert "indice_riesgo_compuesto" in df_feat.columns
    assert any("lag" in c for c in df_feat.columns), "No se encontraron columnas de lag"
    assert any("roll" in c for c in df_feat.columns), "No se encontraron columnas rolling"


def test_train_test_split_proportions(synthetic_df):
    """La partición debe respetar las proporciones indicadas."""
    df_proc = preprocess_data(synthetic_df)
    test_sizes = [0.2, 0.3]

    for ts in test_sizes:
        X_train, X_test, y_train, y_test = split_data(
            df_proc, target_col="brote", test_size=ts, random_state=42
        )
        total = len(X_train) + len(X_test)
        actual_ratio = len(X_test) / total
        assert abs(actual_ratio - ts) < 0.01, (
            f"Ratio esperado {ts}, obtenido {actual_ratio:.3f}"
        )
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
