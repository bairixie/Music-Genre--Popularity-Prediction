import pandas as pd
import numpy as np

file_path = 'musicData.csv'
df = pd.read_csv(file_path)

df = df.drop(columns=['artist_name', 'track_name'], errors='ignore')

df['duration_ms'] = df['duration_ms'].replace(-1, pd.NA)

genre_duration_mean = df.groupby('music_genre')['duration_ms'].mean()

for genre, mean_value in genre_duration_mean.items():
    df.loc[(df['music_genre'] == genre) & (df['duration_ms'].isna()), 'duration_ms'] = round(mean_value)

df['duration_ms'] = df['duration_ms'].fillna(round(df['duration_ms'].mean())).astype(int)

df['tempo'] = df['tempo'].replace('?', np.nan)

df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

genre_tempo_mean = df.groupby('music_genre')['tempo'].mean()

for genre, mean_value in genre_tempo_mean.items():
    df.loc[(df['music_genre'] == genre) & (df['tempo'].isna()), 'tempo'] = round(mean_value, 3)

df['tempo'] = df['tempo'].fillna(round(df['tempo'].mean(), 3))
df['tempo'] = df['tempo'].astype(float).round(3)

df = df.drop(columns=['obtained_date','instance_id'], errors='ignore')

key_map_sharp = {
    'C':0, 'C#':1,
    'D':2, 'D#':3,
    'E':4,
    'F':5, 'F#':6,
    'G':7, 'G#':8,
    'A':9, 'A#':10,
    'B':11
}

def encode_key(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    return key_map_sharp.get(s, np.nan)

if 'key' in df.columns:
    df['key_code'] = df['key'].apply(encode_key)

    if df['key_code'].isna().any():
        df['key_code'] = df['key_code'].fillna(df['key_code'].mode().iloc[0])

    df = df.drop(columns=['key'], errors='ignore')

def encode_mode(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == 'major':
        return 1
    if s == 'minor':
        return 0
    return np.nan

if 'mode' in df.columns:
    df['mode_code'] = df['mode'].apply(encode_mode)
    df['mode_code'] = df['mode_code'].fillna(df['mode_code'].mode().iloc[0])
    df = df.drop(columns=['mode'], errors='ignore')

num_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in num_cols:
    if df[col].isna().sum() > 0:
        df[col] = df.groupby('music_genre')[col].transform(
            lambda x: x.fillna(x.mean())
        )

df = df.fillna(df.mean(numeric_only=True))
df = df.dropna(subset=['music_genre']).reset_index(drop=True)
print(df.isna().sum())

df.to_csv('musicData_clean.csv', index=False)
