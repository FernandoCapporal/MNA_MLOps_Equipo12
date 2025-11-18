"""
Pruebas automatizadas para el módulo `src/model_training.py`.

Incluye:
- Pruebas unitarias del pipeline de modelado (construcción, ajuste y evaluación).
- Prueba de integración que valida el flujo de entrenamiento completo utilizando
  la función `train_logistic_regression` con hiperparámetros de ejemplo.

Las pruebas están diseñadas para ejecutarse con:
    pytest -q
desde la raíz del proyecto.
"""

import numpy as np
import pandas as pd
import pytest

from src.model_training import ModelTrainingPipeline, train_logistic_regression


@pytest.fixture
def dummy_dataset():
    """
    Genera un dataset sintético pequeño con variables numéricas y categóricas.

    Este dataset se utiliza para validar que el pipeline:
    - Detecta correctamente tipos de variables.
    - Soporta el flujo de imputación, escalado, codificación y modelado.
    """
    X = pd.DataFrame(
        {
            "age": [25, 35, 45, 30, 50, 40],
            "income": [30000, 50000, 70000, 45000, 80000, 60000],
            "city": ["A", "B", "A", "B", "A", "B"],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 1], name="target")
    return X, y


def test_build_creates_pipeline(dummy_dataset):
    """
    Verifica que el método `build` construye y almacena correctamente el pipeline.

    Criterios de éxito:
    - `build` regresa un objeto de pipeline no nulo.
    - El atributo `pipeline_` del objeto `ModelTrainingPipeline` queda asignado.
    """
    X, _ = dummy_dataset

    trainer = ModelTrainingPipeline()
    pipeline = trainer.build(X)

    assert pipeline is not None, "build() debe regresar un pipeline válido."
    assert trainer.pipeline_ is pipeline, (
        "El atributo pipeline_ debe almacenar la instancia del pipeline creada."
    )


def test_fit_and_evaluate_return_valid_metrics(dummy_dataset):
    """
    Valida que el pipeline completo (fit + evaluate) genera métricas coherentes.

    Criterios de éxito:
    - `evaluate` regresa un diccionario.
    - El diccionario contiene las llaves 'accuracy' y 'f1'.
    - Los valores de las métricas se encuentran en el rango [0, 1].
    """
    from sklearn.model_selection import train_test_split

    X, y = dummy_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    trainer = ModelTrainingPipeline(n_estimators=50, max_depth=3)
    trainer.build(X_train)
    trainer.fit(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)

    assert isinstance(metrics, dict), "evaluate debe regresar un diccionario de métricas."
    assert "accuracy" in metrics and "f1" in metrics, (
        "El diccionario de métricas debe contener 'accuracy' y 'f1'."
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0, "accuracy debe estar entre 0 y 1."
    assert 0.0 <= metrics["f1"] <= 1.0, "f1 debe estar entre 0 y 1."


def test_train_logistic_regression_runs_end_to_end(dummy_dataset, monkeypatch):
    """
    Prueba de integración de la función `train_logistic_regression`.

    Valida el flujo extremo a extremo del pipeline de entrenamiento:
    carga de datos sintéticos → construcción del pipeline → entrenamiento → evaluación.

    Para evitar efectos colaterales, se hace monkeypatch de las funciones de MLflow
    (`start_run` y `log_metrics`), de modo que:
    - No se creen runs reales.
    - No se escriban métricas en un backend externo.

    Criterios de éxito:
    - La función corre sin lanzar excepciones.
    - Regresa un diccionario con 'accuracy' y 'f1' en el rango [0, 1].
    """
    from sklearn.model_selection import train_test_split
    import mlflow

    X, y = dummy_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Mock de MLflow para evitar efectos externos durante las pruebas
    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(mlflow, "start_run", lambda run_name=None: DummyRun())
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics: None)

    metrics = train_logistic_regression(X_train, X_test, y_train, y_test)

    assert isinstance(metrics, dict), "La función debe regresar un diccionario de métricas."
    assert "accuracy" in metrics and "f1" in metrics, (
        "El diccionario debe contener al menos 'accuracy' y 'f1'."
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0, "accuracy debe estar entre 0 y 1."
    assert 0.0 <= metrics["f1"] <= 1.0, "f1 debe estar entre 0 y 1."
