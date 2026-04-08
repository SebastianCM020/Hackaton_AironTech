"""
Entrenamiento de modelos para predicción de dengue.
Soporta Random Forest, XGBoost y LightGBM para tareas de clasificación y regresión.

Uso:
    python src/models/train.py --model_type rf --task classification
    python src/models/train.py --model_type xgb --task regression
    python src/models/train.py --model_type lgbm --task classification
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore")

CONFIG_PATH = Path("config/config.yaml")
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def load_config() -> dict:
    """Carga la configuración desde config/config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_processed_data(task: str) -> tuple:
    """
    Carga los datos procesados desde data/processed/.

    Returns
    -------
    Tuple (X_train, X_test, y_train, y_test)
    """
    config = load_config()

    if not (DATA_DIR / "X_train.csv").exists():
        print("Datos procesados no encontrados. Generando y preprocesando...")
        _prepare_data(config, task)

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def _prepare_data(config: dict, task: str):
    """Genera datos sintéticos y los preprocesa si no existen."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.generate_synthetic_data import generate_synthetic_data
    from src.data.preprocess import preprocess_data, split_data, save_processed

    raw_path = Path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        df = generate_synthetic_data()
        df.to_csv(raw_path, index=False)

    df = pd.read_csv(raw_path)
    df_proc = preprocess_data(df)

    target_map = {
        "classification": config["features"]["target_classification"],
        "regression": config["features"]["target_regression"],
    }
    target_col = target_map.get(task, "brote")

    X_train, X_test, y_train, y_test = split_data(
        df_proc,
        target_col=target_col,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
    )
    save_processed(X_train, X_test, y_train, y_test, config["data"]["processed_path"])


def get_model(model_type: str, task: str, config: dict):
    """
    Instancia el modelo según tipo y tarea.

    Parameters
    ----------
    model_type : str
        'rf', 'xgb' o 'lgbm'
    task : str
        'classification' o 'regression'
    config : dict

    Returns
    -------
    Estimador de scikit-learn compatible.
    """
    rf_params = config["models"]["random_forest"]
    xgb_params = config["models"]["xgboost"]
    lgbm_params = config["models"]["lightgbm"]

    models = {
        ("rf", "classification"):   RandomForestClassifier(**rf_params),
        ("rf", "regression"):       RandomForestRegressor(**rf_params),
        ("xgb", "classification"):  XGBClassifier(**xgb_params, eval_metric="logloss",
                                                   use_label_encoder=False,
                                                   verbosity=0),
        ("xgb", "regression"):      XGBRegressor(**xgb_params, verbosity=0),
        ("lgbm", "classification"): LGBMClassifier(**lgbm_params, verbose=-1),
        ("lgbm", "regression"):     LGBMRegressor(**lgbm_params, verbose=-1),
    }

    key = (model_type, task)
    if key not in models:
        raise ValueError(f"Combinación no soportada: model_type={model_type}, task={task}")
    return models[key]


def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                         task: str, n_splits: int = 5) -> dict:
    """
    Valida el modelo con K-Fold cross-validation.

    Returns
    -------
    dict con métricas promedio de validación cruzada.
    """
    if task == "classification":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        metric_name = "accuracy"
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        metric_name = "rmse"

    scores = []
    X_arr = X_train.values
    y_arr = y_train.values

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr, y_arr), 1):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        if task == "classification":
            score = accuracy_score(y_val, preds)
        else:
            score = np.sqrt(mean_squared_error(y_val, preds))

        scores.append(score)
        print(f"  Fold {fold}/{n_splits}: {metric_name}={score:.4f}")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    print(f"  CV {metric_name}: {mean_score:.4f} ± {std_score:.4f}")
    return {metric_name: mean_score, f"{metric_name}_std": std_score}


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series,
                     task: str) -> dict:
    """Evalúa el modelo en el conjunto de prueba."""
    preds = model.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        }
    else:
        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds),
        }

    print("\nMétricas en conjunto de prueba:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def save_model(model, model_type: str, task: str) -> Path:
    """Guarda el modelo entrenado con joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_type}_{task}.pkl"
    joblib.dump(model, model_path)
    print(f"\nModelo guardado: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos de predicción de dengue")
    parser.add_argument("--model_type", choices=["rf", "xgb", "lgbm"], default="rf",
                        help="Tipo de modelo: rf, xgb, lgbm")
    parser.add_argument("--task", choices=["classification", "regression"],
                        default="classification",
                        help="Tarea: classification o regression")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Número de folds para cross-validation")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Entrenando: {args.model_type.upper()} | Tarea: {args.task}")
    print(f"{'='*60}\n")

    config = load_config()
    X_train, X_test, y_train, y_test = load_processed_data(args.task)

    # Mantener solo columnas numéricas
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test[X_train.columns]

    model = get_model(args.model_type, args.task, config)

    print(f"Cross-validation ({args.cv_folds} folds):")
    cv_metrics = cross_validate_model(model, X_train, y_train,
                                      args.task, n_splits=args.cv_folds)

    # Reentrenar con todos los datos de entrenamiento
    model.fit(X_train.values, y_train.values)

    test_metrics = evaluate_on_test(model, X_test, y_test, args.task)
    save_model(model, args.model_type, args.task)

    print(f"\n{'='*60}")
    print("Entrenamiento completado.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
