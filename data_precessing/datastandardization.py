
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

SRC = "musicData_genreOneHot.csv"

CLS_TRAIN = "train_classification_std.csv"
CLS_VAL = "val_classification_std.csv"
CLS_TEST = "test_classification_std.csv"
CLS_SCALER = "scaler_classification.pkl"
REG_TRAIN = "train_regression_std.csv"
REG_VAL = "val_regression_std.csv"
REG_TEST = "test_regression_std.csv"
REG_SCALER = "scaler_regression.pkl"
SPLIT_PATH = "split_indices.npz"

print(" Loading:", SRC)
df = pd.read_csv(SRC)

genre_cols = [c for c in df.columns if c.startswith("genre_")]
if not genre_cols:
    raise ValueError("No genre_ one-hot columns found. Please ensure OHE has been done.")
if "popularity" not in df.columns:
    raise ValueError("Column 'popularity' not found.")

mask_has_genre = df[genre_cols].sum(axis=1) > 0
df = df.loc[mask_has_genre].reset_index(drop=True)

mask_pop = pd.to_numeric(df["popularity"], errors="coerce").notna()
df = df.loc[mask_pop].reset_index(drop=True)

for c in df.columns:
    if df[c].dtype == object and c not in genre_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

y_strat = df[genre_cols].idxmax(axis=1).str.replace("^genre_", "", regex=True)

idx_all = np.arange(len(df))
idx_train, idx_tmp, strat_train, strat_tmp = train_test_split(
    idx_all, y_strat, test_size=0.30, random_state=42, stratify=y_strat, shuffle=True
)
idx_val, idx_test, strat_val, strat_test = train_test_split(
    idx_tmp, strat_tmp, test_size=0.50, random_state=42, stratify=strat_tmp, shuffle=True
)

np.savez(SPLIT_PATH, train=idx_train, val=idx_val, test=idx_test)

X_cls = df.drop(columns=genre_cols + ["popularity"]).copy()
y_cls = df[genre_cols].copy()

imputer_cls = SimpleImputer(strategy="mean")
scaler_cls = StandardScaler()

X_cls_train = X_cls.iloc[idx_train].reset_index(drop=True)
X_cls_val = X_cls.iloc[idx_val].reset_index(drop=True)
X_cls_test = X_cls.iloc[idx_test].reset_index(drop=True)

# fit on train
X_cls_train_imp = imputer_cls.fit_transform(X_cls_train)
X_cls_val_imp = imputer_cls.transform(X_cls_val)
X_cls_test_imp = imputer_cls.transform(X_cls_test)

X_cls_train_std = scaler_cls.fit_transform(X_cls_train_imp)
X_cls_val_std = scaler_cls.transform(X_cls_val_imp)
X_cls_test_std = scaler_cls.transform(X_cls_test_imp)

X_cls_cols = X_cls.columns.tolist()
train_cls_out = pd.DataFrame(X_cls_train_std, columns=X_cls_cols).join(y_cls.iloc[idx_train].reset_index(drop=True))
val_cls_out = pd.DataFrame(X_cls_val_std, columns=X_cls_cols).join(y_cls.iloc[idx_val].reset_index(drop=True))
test_cls_out = pd.DataFrame(X_cls_test_std, columns=X_cls_cols).join(y_cls.iloc[idx_test].reset_index(drop=True))

train_cls_out.to_csv(CLS_TRAIN, index=False)
val_cls_out.to_csv(CLS_VAL, index=False)
test_cls_out.to_csv(CLS_TEST, index=False)
joblib.dump({"imputer": imputer_cls, "scaler": scaler_cls, "feature_cols": X_cls_cols}, CLS_SCALER)


X_reg = df.drop(columns=["popularity"]).copy()
y_reg = df["popularity"].astype(float).copy()

imputer_reg = SimpleImputer(strategy="mean")
scaler_reg = StandardScaler()

X_reg_train = X_reg.iloc[idx_train].reset_index(drop=True)
X_reg_val = X_reg.iloc[idx_val].reset_index(drop=True)
X_reg_test = X_reg.iloc[idx_test].reset_index(drop=True)

X_reg_train_imp = imputer_reg.fit_transform(X_reg_train)
X_reg_val_imp = imputer_reg.transform(X_reg_val)
X_reg_test_imp = imputer_reg.transform(X_reg_test)

X_reg_train_std = scaler_reg.fit_transform(X_reg_train_imp)
X_reg_val_std = scaler_reg.transform(X_reg_val_imp)
X_reg_test_std = scaler_reg.transform(X_reg_test_imp)

X_reg_cols = X_reg.columns.tolist()
train_reg_out = pd.DataFrame(X_reg_train_std, columns=X_reg_cols).assign(popularity=y_reg.iloc[idx_train].reset_index(drop=True))
val_reg_out = pd.DataFrame(X_reg_val_std, columns=X_reg_cols).assign(popularity=y_reg.iloc[idx_val].reset_index(drop=True))
test_reg_out = pd.DataFrame(X_reg_test_std, columns=X_reg_cols).assign(popularity=y_reg.iloc[idx_test].reset_index(drop=True))

train_reg_out.to_csv(REG_TRAIN, index=False)
val_reg_out.to_csv(REG_VAL, index=False)
test_reg_out.to_csv(REG_TEST, index=False)
joblib.dump({"imputer": imputer_reg, "scaler": scaler_reg, "feature_cols": X_reg_cols}, REG_SCALER)

