import pandas as pd

# --- File paths ---
paths = {
    "train_cls": "train_classification_std.csv",
    "train_reg": "train_regression_std.csv",
    "val_cls": "val_classification_std.csv",
    "val_reg": "val_regression_std.csv",
    "test_cls": "test_classification_std.csv",
    "test_reg": "test_regression_std.csv",
}

# --- Load CSV files ---
dfs = {name: pd.read_csv(path) for name, path in paths.items()}

# --- Detect genre one-hot columns automatically ---
def get_genre_columns(df):
    return [col for col in df.columns if col.startswith("genre")]

genre_cols = get_genre_columns(dfs["train_cls"])

# --- Function to merge classification + regression ---
def merge_multitask(cls_df, reg_df):
    # X: remove genre one-hot and popularity if present
    X = cls_df.drop(columns=genre_cols)
    if "popularity" in X.columns:
        X = X.drop(columns=["popularity"])

    # y_class: one-hot genre
    y_class = cls_df[genre_cols]

    # y_reg: popularity target
    y_reg = reg_df[["popularity"]]

    # Combine
    return pd.concat([X, y_class, y_reg], axis=1)

# --- Build final datasets ---
train_multitask = merge_multitask(dfs["train_cls"], dfs["train_reg"])
val_multitask   = merge_multitask(dfs["val_cls"], dfs["val_reg"])
test_multitask  = merge_multitask(dfs["test_cls"], dfs["test_reg"])

# --- Save ---
train_multitask.to_csv("train_multitask.csv", index=False)
val_multitask.to_csv("val_multitask.csv", index=False)
test_multitask.to_csv("test_multitask.csv", index=False)
