from data_preprocessing import load_data, preprocess_data
from model_training import train_logistic_regression
from evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import os
import numpy as np

def main():


    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # sube un nivel desde /src
    data_path = os.path.join(BASE_DIR, "data", "raw", "Datasets", "insurance_company_modified.csv")
    print("Cargando y preprocesando datos del path...")
    print( data_path)

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    metric = train_logistic_regression(X_train, X_test, y_train, y_test, C=1.0)

if __name__ == "__main__":
    main()
