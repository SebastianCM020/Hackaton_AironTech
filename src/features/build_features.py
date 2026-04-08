"""
Ingeniería de características para el dataset de dengue.
Agrega variables de rezago, promedios móviles, interacciones y estacionalidad.
"""

import numpy as np
import pandas as pd

# Epsilon para evitar división por cero en normalizaciones
_EPSILON = 1e-9


def add_lag_features(
    df: pd.DataFrame,
    lag_cols: list,
    lags: list = [1, 2, 4],
) -> pd.DataFrame:
    """
    Agrega características de rezago temporal por cantón.

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener 'canton_id', 'year', 'semana_epidemiologica'.
    lag_cols : list
        Columnas a las que aplicar rezago.
    lags : list
        Número de semanas de rezago.

    Returns
    -------
    pd.DataFrame con columnas nuevas `{col}_lag{n}`.
    """
    df = df.copy()
    df = df.sort_values(["canton_id", "year", "semana_epidemiologica"])

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = (
                df.groupby("canton_id")[col]
                .shift(lag)
                .fillna(0)
            )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    roll_cols: list,
    windows: list = [4, 8],
) -> pd.DataFrame:
    """
    Agrega promedios móviles por cantón.

    Parameters
    ----------
    df : pd.DataFrame
    roll_cols : list
        Columnas a las que aplicar promedios móviles.
    windows : list
        Tamaños de ventana en semanas.

    Returns
    -------
    pd.DataFrame con columnas `{col}_roll{w}`.
    """
    df = df.copy()
    df = df.sort_values(["canton_id", "year", "semana_epidemiologica"])

    def _lagged_rolling_mean(series: pd.Series, window: int) -> pd.Series:
        """Calcula la media móvil sobre valores pasados (shift(1)) para evitar
        filtración de datos del período actual en el promedio histórico."""
        return series.shift(1).rolling(window, min_periods=1).mean()

    for col in roll_cols:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_roll{w}"] = (
                df.groupby("canton_id")[col]
                .transform(_lagged_rolling_mean, w)
                .fillna(0)
            )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características de interacción entre variables climáticas.

    Returns
    -------
    pd.DataFrame con columnas nuevas de interacción.
    """
    df = df.copy()

    if "temperatura_promedio" in df.columns and "precipitacion_mm" in df.columns:
        df["temp_precip_interaccion"] = (
            df["temperatura_promedio"] * df["precipitacion_mm"]
        )

    if "indice_aedes" in df.columns and "densidad_poblacional" in df.columns:
        df["aedes_densidad"] = df["indice_aedes"] * np.log1p(df["densidad_poblacional"])

    if "temperatura_promedio" in df.columns and "indice_aedes" in df.columns:
        df["temp_aedes"] = df["temperatura_promedio"] * df["indice_aedes"]

    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica la semana epidemiológica con seno y coseno (encoding cíclico).

    Returns
    -------
    pd.DataFrame con columnas 'semana_sin' y 'semana_cos'.
    """
    df = df.copy()

    if "semana_epidemiologica" in df.columns:
        semana = df["semana_epidemiologica"]
        df["semana_sin"] = np.sin(2 * np.pi * semana / 52)
        df["semana_cos"] = np.cos(2 * np.pi * semana / 52)

    return df


def add_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un índice de riesgo compuesto normalizado (0-1).

    Combina temperatura, precipitación, índice Aedes e inverso de cobertura
    de salud en un único índice.

    Returns
    -------
    pd.DataFrame con columna 'indice_riesgo_compuesto'.
    """
    df = df.copy()

    required = ["temperatura_promedio", "precipitacion_mm", "indice_aedes", "cobertura_salud"]
    if not all(c in df.columns for c in required):
        return df

    temp_norm = (df["temperatura_promedio"] - df["temperatura_promedio"].min()) / (
        df["temperatura_promedio"].max() - df["temperatura_promedio"].min() + _EPSILON
    )
    precip_norm = (df["precipitacion_mm"] - df["precipitacion_mm"].min()) / (
        df["precipitacion_mm"].max() - df["precipitacion_mm"].min() + _EPSILON
    )
    aedes_norm = df["indice_aedes"]
    salud_inv = 1 - df["cobertura_salud"]

    df["indice_riesgo_compuesto"] = (
        0.30 * temp_norm
        + 0.25 * precip_norm
        + 0.30 * aedes_norm
        + 0.15 * salud_inv
    )
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica toda la pipeline de ingeniería de características.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset preprocesado.

    Returns
    -------
    pd.DataFrame con todas las características adicionales.
    """
    lag_cols = ["casos_dengue", "indice_aedes", "temperatura_promedio", "precipitacion_mm"]
    roll_cols = ["casos_dengue", "indice_aedes", "precipitacion_mm"]

    df = add_lag_features(df, lag_cols=lag_cols, lags=[1, 2, 4])
    df = add_rolling_features(df, roll_cols=roll_cols, windows=[4, 8])
    df = add_interaction_features(df)
    df = add_seasonal_features(df)
    df = add_risk_index(df)

    return df


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/processed/X_train.csv"
    df = pd.read_csv(filepath)
    df_feat = build_all_features(df)
    print(f"Características originales: {len(pd.read_csv(filepath).columns)}")
    print(f"Características finales   : {len(df_feat.columns)}")
    print("Nuevas columnas:", [c for c in df_feat.columns if c not in pd.read_csv(filepath).columns])
