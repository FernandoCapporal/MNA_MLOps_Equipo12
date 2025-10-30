import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df, target):
    # Verifica que la columna objetivo exista
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el DataFrame.")

    # Separar variables predictoras y objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Verificar que no queden NaN
    if np.isnan(X_scaled).any():
        raise ValueError("Aún quedan NaN en los datos después del preprocesamiento.")

    return X_scaled, y, scaler