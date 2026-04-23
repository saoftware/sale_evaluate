import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================
# Chargement des données
# =========================
def load_data(file_path):
    df = pd.read_csv(file_path, encoding="latin1")
    return df


# =========================
# Nettoyage
# =========================
def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()

    # garder transactions valides
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ]

    return df


# =========================
# Création variable cible
# =========================
def create_target(df):
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    return df


# =========================
# Features temporelles
# =========================
def create_time_features(df):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["WeekDay"] = df["InvoiceDate"].dt.weekday

    return df


# =========================
# Features ventes
# =========================
def prepare_sales_features(df):
    features = [
        "Quantity",
        "UnitPrice",
        "Month",
        "Day",
        "Hour",
        "WeekDay"
    ]

    X = df[features]
    y = df["Sales"]

    return X, y


# =========================
# Features fraude
# =========================
def prepare_fraud_features(df):
    features = [
        "Quantity",
        "UnitPrice",
        "Sales"
    ]

    X = df[features]

    return X


# =========================
# Evaluation régression
# =========================
def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


# =========================
# Affichage métriques
# =========================
def print_metrics(metrics, model_name="Model"):
    print(f"\n===== {model_name.upper()} =====")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"R2   : {metrics['R2']:.4f}")