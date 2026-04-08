"""
Evaluación de modelos para predicción de dengue.
Genera métricas, matrices de confusión, importancia de variables y valores SHAP.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

REPORTS_DIR = Path("reports")


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_classification(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_report: bool = True,
) -> dict:
    """
    Evalúa un modelo de clasificación.

    Parameters
    ----------
    model : estimador sklearn
    X_test : pd.DataFrame
    y_test : pd.Series
    save_report : bool
        Si True, guarda el reporte en reports/.

    Returns
    -------
    dict con accuracy, precision, recall, f1, roc_auc.
    """
    _ensure_reports_dir()
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    y_pred = model.predict(X_arr)

    metrics = {
        "accuracy":  float(accuracy_score(y_arr, y_pred)),
        "precision": float(precision_score(y_arr, y_pred, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_arr, y_pred, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_arr, y_pred, average="weighted", zero_division=0)),
    }

    # ROC-AUC (solo para clasificación binaria con predict_proba)
    if hasattr(model, "predict_proba") and len(np.unique(y_arr)) == 2:
        y_prob = model.predict_proba(X_arr)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_arr, y_prob))
    else:
        metrics["roc_auc"] = None

    print("\n=== Métricas de Clasificación ===")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:12s}: {v:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(y_arr, y_pred, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_arr, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    if save_report:
        with open(REPORTS_DIR / "classification_report.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nReporte guardado en: {REPORTS_DIR}/classification_report.json")

    return metrics


def evaluate_regression(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_report: bool = True,
) -> dict:
    """
    Evalúa un modelo de regresión.

    Parameters
    ----------
    model : estimador sklearn
    X_test : pd.DataFrame
    y_test : pd.Series
    save_report : bool

    Returns
    -------
    dict con MAE, RMSE, R², MAPE.
    """
    _ensure_reports_dir()
    X_arr = X_test.values if hasattr(X_test, "values") else X_test
    y_arr = y_test.values if hasattr(y_test, "values") else y_test

    y_pred = model.predict(X_arr)

    # MAPE protegido contra división por cero
    mask = y_arr != 0
    mape = float(np.mean(np.abs((y_arr[mask] - y_pred[mask]) / y_arr[mask])) * 100) if mask.any() else 0.0

    metrics = {
        "mae":  float(mean_absolute_error(y_arr, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_arr, y_pred))),
        "r2":   float(r2_score(y_arr, y_pred)),
        "mape": mape,
    }

    print("\n=== Métricas de Regresión ===")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    # Gráfico predicciones vs real
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_arr, y_pred, alpha=0.3, s=10, color="steelblue")
    lim = max(y_arr.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Línea perfecta")
    ax.set_xlabel("Casos reales")
    ax.set_ylabel("Casos predichos")
    ax.set_title(f"Predicciones vs Real  (R²={metrics['r2']:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "predictions_vs_actual.png", dpi=120)
    plt.close()

    if save_report:
        with open(REPORTS_DIR / "regression_report.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nReporte guardado en: {REPORTS_DIR}/regression_report.json")

    return metrics


def plot_shap_values(
    model,
    X_test: pd.DataFrame,
    feature_names: list,
    max_samples: int = 500,
) -> None:
    """
    Genera y guarda el gráfico de valores SHAP (importancia global).

    Parameters
    ----------
    model : estimador entrenado
    X_test : pd.DataFrame
    feature_names : list
        Nombres de las características.
    max_samples : int
        Máximo de filas para acelerar el cálculo.
    """
    try:
        import shap
    except ImportError:
        print("SHAP no disponible. Instala con: pip install shap")
        return

    _ensure_reports_dir()
    X_arr = X_test.values[:max_samples] if hasattr(X_test, "values") else X_test[:max_samples]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_arr)

        # Para clasificación binaria TreeExplainer puede devolver lista
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        idx = np.argsort(mean_abs_shap)[::-1][:20]

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(
            range(len(idx)),
            mean_abs_shap[idx],
            color="steelblue",
            edgecolor="white",
        )
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx], fontsize=10)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Importancia de Variables (SHAP)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=120)
        plt.close()
        print(f"Gráfico SHAP guardado en: {REPORTS_DIR}/shap_importance.png")

    except Exception as e:
        print(f"No se pudo calcular SHAP: {e}")


if __name__ == "__main__":
    import joblib
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/rf_classification.pkl"
    task = sys.argv[2] if len(sys.argv) > 2 else "classification"

    model = joblib.load(model_path)
    X_test = pd.read_csv("data/processed/X_test.csv").select_dtypes(include=[np.number])
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    if task == "classification":
        evaluate_classification(model, X_test, y_test)
    else:
        evaluate_regression(model, X_test, y_test)

    plot_shap_values(model, X_test, list(X_test.columns))
