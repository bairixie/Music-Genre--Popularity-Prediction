# datapreprocessing_classification_pca.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

SRC = "musicData_genreOneHot.csv"
OUT_TRAIN = "train_classification_PCA.csv"
OUT_VAL   = "val_classification_PCA.csv"
OUT_TEST  = "test_classification_PCA.csv"
IMP_PATH  = "imputer.pkl"
SCL_PATH  = "scaler.pkl"
PCA_PATH  = "pca.pkl"
COLS_PATH = "feature_columns_before_pca.txt"

df = pd.read_csv(SRC)

genre_cols = [c for c in df.columns if c.startswith("genre_")]
if not genre_cols:
    raise ValueError("No genre_ one-hot columns found. Please run OHE first.")

y = df[genre_cols].copy()
X = df.drop(columns=genre_cols).copy()

mask_has_label = y.sum(axis=1) > 0
if (~mask_has_label).any():
    n_drop = (~mask_has_label).sum()
    X = X.loc[mask_has_label].reset_index(drop=True)
    y = y.loc[mask_has_label].reset_index(drop=True)

y_strat = y.idxmax(axis=1).str.replace("^genre_", "", regex=True)

for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)
    elif X[c].dtype == object:
        X[c] = pd.to_numeric(X[c], errors="coerce")

all_nan_cols = X.columns[X.isna().all()]
if len(all_nan_cols) > 0:
    X = X.drop(columns=all_nan_cols)

with open(COLS_PATH, "w", encoding="utf-8") as f:
    for col in X.columns:
        f.write(col + "\n")

X_train, X_temp, y_train, y_temp, y_strat_train, y_strat_temp = train_test_split(
    X, y, y_strat, test_size=0.30, random_state=42, shuffle=True, stratify=y_strat
)
X_val, X_test, y_val, y_test, y_strat_val, y_strat_test = train_test_split(
    X_temp, y_temp, y_strat_temp, test_size=0.50, random_state=42, shuffle=True, stratify=y_strat_temp
)


imputer = SimpleImputer(strategy="mean")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

for name, arr in [("train", X_train_scaled), ("val", X_val_scaled), ("test", X_test_scaled)]:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf after impute+scale.")

pca = PCA(n_components=0.95, svd_solver="full")
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_cols = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols).reset_index(drop=True)
X_val_pca_df = pd.DataFrame(X_val_pca,   columns=pca_cols).reset_index(drop=True)
X_test_pca_df = pd.DataFrame(X_test_pca,  columns=pca_cols).reset_index(drop=True)

y_train_df = y_train.reset_index(drop=True)
y_val_df = y_val.reset_index(drop=True)
y_test_df = y_test.reset_index(drop=True)

train_out = pd.concat([X_train_pca_df, y_train_df], axis=1)
val_out = pd.concat([X_val_pca_df,   y_val_df],   axis=1)
test_out = pd.concat([X_test_pca_df,  y_test_df],  axis=1)

train_out.to_csv(OUT_TRAIN, index=False)
val_out.to_csv(OUT_VAL, index=False)
test_out.to_csv(OUT_TEST, index=False)

joblib.dump(imputer, IMP_PATH)
joblib.dump(scaler,  SCL_PATH)
joblib.dump(pca,     PCA_PATH)


