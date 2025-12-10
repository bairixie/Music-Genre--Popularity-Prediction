import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

SRC = "musicData_genreOneHot.csv"
OUT_TRAIN = "train_regression_PCA.csv"
OUT_VAL = "val_regression_PCA.csv"
OUT_TEST = "test_regression_PCA.csv"
IMP_PATH = "reg_imputer.pkl"
SCL_PATH = "reg_scaler.pkl"
PCA_PATH = "reg_pca.pkl"
COLS_PATH = "reg_feature_columns_before_pca.txt"

df = pd.read_csv(SRC)

if "popularity" not in df.columns:
    raise ValueError("Column 'popularity' not found in dataset.")
y = pd.to_numeric(df["popularity"], errors="coerce")

mask = y.notna()
df = df.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

X = df.drop(columns=["popularity"]).copy()

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

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
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
train_out = pd.DataFrame(X_train_pca, columns=pca_cols).assign(popularity=y_train.values)
val_out = pd.DataFrame(X_val_pca, columns=pca_cols).assign(popularity=y_val.values)
test_out = pd.DataFrame(X_test_pca, columns=pca_cols).assign(popularity=y_test.values)

train_out.to_csv(OUT_TRAIN, index=False)
val_out.to_csv(OUT_VAL, index=False)
test_out.to_csv(OUT_TEST, index=False)

joblib.dump(imputer, IMP_PATH)
joblib.dump(scaler, SCL_PATH)
joblib.dump(pca, PCA_PATH)

