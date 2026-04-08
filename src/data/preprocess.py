"""
Preprocesamiento del dataset de dengue.
Funciones para carga, limpieza, codificación y partición de datos.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Parameters
    ----------
    filepath : str
        Ruta al archivo CSV.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    df = pd.read_csv(path)
    print(f"Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y preprocesa el DataFrame:
    - Elimina duplicados
    - Imputa valores faltantes
    - Codifica variables categóricas

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo.

    Returns
    -------
    pd.DataFrame procesado con columnas adicionales de codificación.
    """
    df = df.copy()

    # Eliminar duplicados
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"Duplicados eliminados: {n_dup}")

    # Imputar numéricos con mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputado {col}: {missing} valores con mediana={median_val:.2f}")

    # Imputar categóricos con moda
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  Imputado {col}: {missing} valores con moda='{mode_val}'")

    # Codificar provincia con LabelEncoder
    if "provincia" in df.columns:
        le = LabelEncoder()
        df["provincia_encoded"] = le.fit_transform(df["provincia"].astype(str))

    # Codificar nivel_riesgo como ordinal
    if "nivel_riesgo" in df.columns:
        risk_order = {"bajo": 0, "medio": 1, "alto": 2, "muy_alto": 3}
        df["nivel_riesgo_encoded"] = df["nivel_riesgo"].map(risk_order)

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Nombre de la columna objetivo.
    test_size : float
        Proporción del conjunto de prueba (0-1).
    random_state : int

    Returns
    -------
    Tuple (X_train, X_test, y_train, y_test)
    """
    # Excluir columnas no predictoras
    drop_cols = [
        "canton_id", "canton_name", "provincia",
        "brote", "nivel_riesgo", "nivel_riesgo_encoded", "casos_dengue",
    ]
    # Mantener el target en el DataFrame antes de separar
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Asegurar que target_col esté disponible
    if target_col not in df.columns:
        raise ValueError(f"Columna objetivo no encontrada: {target_col}")

    X = df[[c for c in feature_cols if c != target_col]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """
    Guarda los conjuntos preprocesados como archivos CSV.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    output_dir : str
        Directorio de salida.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)
    print(f"Datos procesados guardados en: {output_dir}")


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/raw/dengue_ecuador_sintetico.csv"
    df = load_data(filepath)
    df_proc = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_proc, target_col="brote")
    save_processed(X_train, X_test, y_train, y_test, "data/processed/")
