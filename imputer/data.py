# imputer/data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_numeric_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df_num = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
    return df_num

def make_missingness(X, rate, seed):
    rng = np.random.default_rng(seed)
    mask_obs = np.ones_like(X, dtype=bool)
    positions = rng.uniform(size=X.shape) < rate
    mask_obs[positions] = False
    X_masked = X.copy()
    X_masked[~mask_obs] = 0.0
    return X_masked, mask_obs

def scale_data(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler
