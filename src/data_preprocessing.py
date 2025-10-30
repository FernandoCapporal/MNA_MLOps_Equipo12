#from matplotlib.path import Path
from pathlib import Path
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

def preprocess_data(data_path):
    # Ajusta esta ruta a tu archivo real
    cleaner = DataCleaningPipeline(input_path=data_path)
    df = cleaner.load_with_official_columns()
    df = cleaner.structural_cleanup()

    print("Tamaño del dataset limpio:", df.shape)
    print("\nConteo de la variable objetivo:")
    print(df["CARAVAN"].value_counts(dropna=False))

    X_train, X_test, y_train, y_test = cleaner.split(test_size=0.2, random_state=42)
    X_train.shape, X_test.shape

    return X_train, X_test, y_train, y_test

class DataCleaningPipeline:
    """Pipeline de limpieza estructural del dataset Caravan.
    No imputa nulos; esa responsabilidad pertenece al pipeline de modelado.
    """
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.df_ = None

    def load_with_official_columns(self):
        column_names = [
            "MOSTYPE","MAANTHUI","MGEMOMV","MGEMLEEF","MOSHOOFD","MGODRK","MGODPR","MGODOV","MGODGE",
            "MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND","MFWEKIND","MOPLHOOG","MOPLMIDD","MOPLLAAG",
            "MBERHOOG","MBERZELF","MBERBOER","MBERMIDD","MBERARBG","MBERARBO","MSKA","MSKB1","MSKB2",
            "MSKC","MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS","MZPART","MINKM30",
            "MINK3045","MINK4575","MINK7512","MINK123M","MINKGEM","MKOOPKLA","PWAPART","PWABEDR",
            "PWALAND","PPERSAUT","PBESAUT","PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR","PWERKT","PBROM",
            "PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS","PINBOED",
            "PBYSTAND","AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT","AMOTSCO","AVRAAUT",
            "AAANHANG","ATRACTOR","AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG","ABRAND",
            "AZEILPL","APLEZIER","AFIETS","AINBOED","ABYSTAND","CARAVAN"
        ]
        # Cargar CSV sin encabezado usando los nombres oficiales
        df = pd.read_csv(self.input_path, sep=',', header=None, names=column_names)
        # En algunos casos el CSV trae una primera fila tipo encabezado textual: la eliminamos si no es numérica
        # Revisamos si la primera fila contiene letras; si sí, la removemos
        if df.iloc[0].astype(str).str.contains('[A-Za-z]', regex=True).any():
            df = df.iloc[1:].reset_index(drop=True)
        self.df_ = df
        return self.df_

    def structural_cleanup(self):
        # Eliminar columna final si está completamente vacía (por seguridad)
        if self.df_.iloc[:, -1].isnull().all():
            self.df_ = self.df_.iloc[:, :-1]

        # Asegurar tipos numéricos cuando aplique
        for col in self.df_.columns:
            # Intentar convertir a numérico cuando sea posible
            self.df_[col] = pd.to_numeric(self.df_[col], errors='ignore')

        # Validar y asegurar binariedad del target
        assert "CARAVAN" in self.df_.columns, "No se encontró la columna objetivo 'CARAVAN'."
        # Forzar a enteros donde sea seguro
        try:
            self.df_["CARAVAN"] = pd.to_numeric(self.df_["CARAVAN"], errors='coerce').fillna(0).astype(int).clip(0, 1)
        except Exception as e:
            raise ValueError("No se pudo convertir CARAVAN a 0/1. Revisa el archivo de entrada.") from e

        return self.df_

    def split(self, test_size=0.2, random_state=42):
        target = "CARAVAN"
        X = self.df_.drop(columns=[target])
        y = self.df_[target].astype(int)
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
