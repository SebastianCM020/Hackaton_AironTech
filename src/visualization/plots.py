"""
Módulo de visualizaciones para el análisis de dengue en Ecuador.
Genera gráficos de series temporales, mapas de calor, correlaciones,
importancia de variables y mapas de riesgo geográfico.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores para niveles de riesgo
RISK_COLORS = {
    "bajo":     "#2ecc71",
    "medio":    "#f1c40f",
    "alto":     "#e67e22",
    "muy_alto": "#e74c3c",
}


def plot_dengue_timeseries(
    df: pd.DataFrame,
    province: str = None,
    save: bool = True,
) -> plt.Figure:
    """
    Grafica la serie temporal de casos de dengue.

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener 'year', 'semana_epidemiologica', 'casos_dengue', 'provincia'.
    province : str, optional
        Filtra por provincia. Si es None, usa todas.
    save : bool
        Guarda la figura en reports/.

    Returns
    -------
    plt.Figure
    """
    data = df.copy()
    if province:
        data = data[data["provincia"] == province]
        title = f"Serie Temporal de Casos de Dengue — {province}"
    else:
        title = "Serie Temporal de Casos de Dengue — Ecuador"

    # Agregar por año y semana
    ts = (
        data.groupby(["year", "semana_epidemiologica"])["casos_dengue"]
        .sum()
        .reset_index()
    )
    ts["periodo"] = ts["year"].astype(str) + "-S" + ts["semana_epidemiologica"].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 5))
    for yr, grp in ts.groupby("year"):
        ax.plot(grp["semana_epidemiologica"], grp["casos_dengue"], label=str(yr), linewidth=1.5)
    ax.set_xlabel("Semana Epidemiológica")
    ax.set_ylabel("Casos de Dengue")
    ax.set_title(title, fontsize=13)
    ax.legend(title="Año", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        fname = f"timeseries_{province or 'ecuador'}.png".replace(" ", "_")
        fig.savefig(REPORTS_DIR / fname, dpi=120)
    return fig


def plot_cases_heatmap(
    df: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """
    Mapa de calor de casos de dengue por cantón y semana epidemiológica.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int
        Número de cantones más afectados a mostrar.
    save : bool

    Returns
    -------
    plt.Figure
    """
    pivot = (
        df.groupby(["canton_name", "semana_epidemiologica"])["casos_dengue"]
        .mean()
        .unstack(fill_value=0)
    )
    # Seleccionar top cantones por total de casos
    top_cantons = pivot.sum(axis=1).nlargest(top_n).index
    pivot = pivot.loc[top_cantons]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.1,
        cbar_kws={"label": "Casos promedio"},
    )
    ax.set_title(f"Mapa de Calor: Casos de Dengue por Cantón y Semana (Top {top_n})", fontsize=13)
    ax.set_xlabel("Semana Epidemiológica")
    ax.set_ylabel("Cantón")
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_DIR / "heatmap_casos.png", dpi=120)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Matriz de correlación de las variables numéricas.

    Parameters
    ----------
    df : pd.DataFrame
    save : bool

    Returns
    -------
    plt.Figure
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Excluir IDs y targets encodados
    exclude = ["canton_id", "nivel_riesgo_encoded", "brote"]
    num_cols = [c for c in num_cols if c not in exclude]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title("Matriz de Correlación", fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=120)
    return fig


def plot_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """
    Gráfico de barras horizontal con la importancia de características.

    Parameters
    ----------
    feature_names : list
    importances : np.ndarray
    top_n : int
    save : bool

    Returns
    -------
    plt.Figure
    """
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(names)), vals[::-1], color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {top_n} Variables Más Importantes", fontsize=13)
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
    return fig


def plot_risk_map(
    df: pd.DataFrame,
    year: int = None,
    save: bool = True,
) -> plt.Figure:
    """
    Mapa de riesgo geográfico simplificado usando scatter plot
    (altitud vs temperatura como proxy geográfico).

    Parameters
    ----------
    df : pd.DataFrame
    year : int, optional
        Filtra por año.
    save : bool

    Returns
    -------
    plt.Figure
    """
    data = df.copy()
    if year:
        data = data[data["year"] == year]

    risk_summary = (
        data.groupby(["canton_name", "provincia"])
        .agg(
            altitud=("altitud_msnm", "mean"),
            temperatura=("temperatura_promedio", "mean"),
            nivel_riesgo=("nivel_riesgo", lambda x: x.mode()[0] if len(x) > 0 else "bajo"),
            casos_total=("casos_dengue", "sum"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for nivel, color in RISK_COLORS.items():
        sub = risk_summary[risk_summary["nivel_riesgo"] == nivel]
        if not sub.empty:
            ax.scatter(
                sub["temperatura"],
                sub["altitud"],
                c=color,
                s=np.sqrt(sub["casos_total"] + 1) * 3,
                alpha=0.7,
                label=nivel,
                edgecolors="white",
                linewidth=0.4,
            )

    ax.set_xlabel("Temperatura Promedio (°C)")
    ax.set_ylabel("Altitud (msnm)")
    ax.set_title(f"Mapa de Riesgo por Cantón{' — ' + str(year) if year else ''}", fontsize=13)
    ax.legend(title="Nivel de Riesgo")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_DIR / "risk_map.png", dpi=120)
    return fig


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """
    Gráfico de dispersión entre valores reales y predichos.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    save : bool

    Returns
    -------
    plt.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Línea de regresión
    coef = np.polyfit(y_true, y_pred, 1)
    poly = np.poly1d(coef)
    x_line = np.linspace(y_true.min(), y_true.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=12, color="steelblue", label="Predicciones")
    lim = max(y_true.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Predicción perfecta")
    ax.plot(x_line, poly(x_line), "g-", linewidth=1.5, alpha=0.7, label="Regresión lineal")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Valores Predichos")
    ax.set_title("Predicciones vs Valores Reales", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if save:
        fig.savefig(REPORTS_DIR / "predictions_vs_actual.png", dpi=120)
    return fig
