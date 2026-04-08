"""
Tests para entrenamiento y evaluación de modelos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.evaluate import evaluate_classification, evaluate_regression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """Datos binarios de clasificación sintéticos."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def regression_data():
    """Datos de regresión sintéticos."""
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=6,
        noise=5,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


# ---------------------------------------------------------------------------
# Tests de entrenamiento
# ---------------------------------------------------------------------------

def test_random_forest_classification_trains(classification_data):
    """Random Forest debe entrenar sin errores sobre datos de clasificación."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})


def test_xgboost_classification_trains(classification_data):
    """XGBoost debe entrenar sin errores sobre datos de clasificación."""
    X, y = classification_data
    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})


def test_random_forest_regression_trains(regression_data):
    """Random Forest debe entrenar sin errores sobre datos de regresión."""
    X, y = regression_data
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X.values, y.values)
    preds = model.predict(X.values)
    assert preds.shape == y.shape


# ---------------------------------------------------------------------------
# Tests de métricas
# ---------------------------------------------------------------------------

def test_classification_metrics(classification_data):
    """Las métricas de clasificación deben estar en rangos válidos."""
    X, y = classification_data
    X_train, X_test = X.iloc[:400], X.iloc[400:]
    y_train, y_test = y.iloc[:400], y.iloc[400:]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train.values, y_train.values)

    metrics = evaluate_classification(model, X_test, y_test, save_report=False)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    if metrics["roc_auc"] is not None:
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_regression_metrics(regression_data):
    """Las métricas de regresión deben ser finitas y no negativas donde corresponde."""
    X, y = regression_data
    X_train, X_test = X.iloc[:400], X.iloc[400:]
    y_train, y_test = y.iloc[:400], y.iloc[400:]

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train.values, y_train.values)

    metrics = evaluate_regression(model, X_test, y_test, save_report=False)

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert np.isfinite(metrics["r2"])
    assert metrics["mape"] >= 0


def test_prediction_shape(classification_data):
    """La forma de las predicciones debe coincidir con el conjunto de prueba."""
    X, y = classification_data
    X_train, X_test = X.iloc[:400], X.iloc[400:]
    y_train, y_test = y.iloc[:400], y.iloc[400:]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train.values, y_train.values)
    preds = model.predict(X_test.values)

    assert preds.shape[0] == X_test.shape[0], (
        f"Shape esperada ({X_test.shape[0]},), obtenida {preds.shape}"
    )


def test_feature_importance_available(classification_data):
    """Random Forest debe exponer feature_importances_ después del entrenamiento."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X.values, y.values)

    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == X.shape[1]
    assert np.isclose(model.feature_importances_.sum(), 1.0, atol=1e-5)
