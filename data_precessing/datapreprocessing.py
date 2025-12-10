import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import numpy as np

df = pd.read_csv('musicData_clean.csv')


if 'music_genre' not in df.columns:
    raise ValueError("Column 'music_genre' not found in dataset. Please check your file.")

genre_ohe = pd.get_dummies(df['music_genre'], prefix='genre').astype(int)
df_ohe = pd.concat([df.drop(columns=['music_genre']), genre_ohe], axis=1)

genre_mapping = {col: col.replace('genre_', '') for col in genre_ohe.columns}
df_ohe.to_csv('musicData_genreOneHot.csv', index=False)
pd.Series(genre_mapping).to_csv('music_genre_mapping.csv')


for col, genre in genre_mapping.items():
    print(f"{col}  ➝  {genre}")


