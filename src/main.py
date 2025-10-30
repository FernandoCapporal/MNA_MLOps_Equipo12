from data_preprocessing import load_data, preprocess_data
from model_training import train_logistic_regression
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import os
import numpy as np

def main():
    # 1. Cargar datos

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # sube un nivel desde /src
    data_path = os.path.join(BASE_DIR, "data", "processed", "dataset_tratado_20251012_000533.csv")
    df = load_data(data_path)

    # 2. Preprocesar
    print("Columnas disponibles en el dataset:")
    print(df.columns.tolist())
    X, y, scaler = preprocess_data(df, target="A86")

    # 3. Separar datos
    print(np.isnan(X).sum())  # Debe imprimir 0
    print(type(X), type(y))
    print(X[:5])
    print(y[:5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # 4. Entrenar modelo
    model = train_logistic_regression(X_train, y_train)

    # 5. Evaluar modelo
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
