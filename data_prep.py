import pandas as pd
import numpy as np

COLS = [
    "Class","Age","Sex","Steroid","Antivirals","Fatigue","Malaise",
    "Anorexia","Liver_Big","Liver_Firm","Spleen_Palpable","Spiders",
    "Ascites","Varices","Bilirubin","ALK_Phosphate","SGOT","Albumin",
    "Protime","Histology"
]

NUMERIC_COLS = ["Age","Bilirubin","ALK_Phosphate","SGOT","Albumin","Protime"]

def load_and_clean_data(path="hepatitis_dataset/hepatitis.data"):
    df = pd.read_csv(path, names=COLS)

    # replace ? with NaN
    df = df.replace("?", np.nan)

    # convert numeric columns
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # define X and y
    y = df["Class"].map({2: 1, 1: 0})
    X = df.drop(columns=["Class"])

    # find numeric vs other columns
    numeric_cols = X.select_dtypes(include=["float64","int64"]).columns
    other_cols = [c for c in X.columns if c not in numeric_cols]

    # fill missing values
    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())

    for c in other_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0])

    return X, y
