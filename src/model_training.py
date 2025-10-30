from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Para undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def train_logistic_regression(X_train, X_test, y_train, y_test, C=1.0):
    trainer = ModelTrainingPipeline(
        n_estimators=300,
        max_depth=10,
        criterion='gini',
        random_state=42
    )
    trainer.build(X_train)
    trainer.fit(X_train, y_train)
    metrics = trainer.evaluate(X_test, y_test)
    metrics
    return metrics


class ModelTrainingPipeline:
    """Pipeline de modelado con imputación, escalado, codificación, undersampling y RandomForest.
    Usa los hiperparámetros óptimos previamente encontrados por el usuario.
    """
    def __init__(self, n_estimators=300, max_depth=10, criterion='gini', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.pipeline_ = None

    def build(self, X):
        # Detectar tipos de columnas (numéricas vs categóricas)
        num_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        cat_features = X.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ])

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state
        )

        # ImbPipeline para incluir undersampling dentro del flujo
        self.pipeline_ = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("undersample", RandomUnderSampler(random_state=self.random_state)),
            ("model", rf)
        ])
        return self.pipeline_

    def fit(self, X_train, y_train):
        if self.pipeline_ is None:
            self.build(X_train)
        self.pipeline_.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline_.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(values_format='d')
        plt.title("Matriz de confusión")
        plt.tight_layout()
        plt.show()

        return {"accuracy": acc, "f1": f1}
