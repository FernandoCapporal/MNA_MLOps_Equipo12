from data_preprocessing import load_data, preprocess_data
from model_training import train_logistic_regression
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import os
import numpy as np
import mlflow
import mlflow.sklearn

def main():

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # sube un nivel desde /src
    data_path = os.path.join(BASE_DIR, "data", "raw", "Datasets", "insurance_company_modified.csv")
    print("Cargando y preprocesando datos del path...")
    print( data_path)
    # Configuración de MLflow
    mlruns_path = os.path.join(BASE_DIR, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_path}")  # tracking local en ./mlruns
    mlflow.set_experiment("coil_insurance_experiment")  # nombre de tu experimento

    # Habilitar autolog para sklearn (hazlo antes del .fit)
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    mlflow.start_run(run_name="random-forest-classifier")
    metric = train_logistic_regression(X_train, X_test, y_train, y_test, C=1.0)
    mlflow.end_run()

if __name__ == "__main__":
    main()
