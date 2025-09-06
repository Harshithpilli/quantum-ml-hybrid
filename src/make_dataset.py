import numpy as np
import pandas as pd
import os

def generate_clean(n_samples=400, n_features=6):
    X_class0 = np.random.normal(0, 1, (n_samples // 2, n_features))
    X_class1 = np.random.normal(3, 1, (n_samples // 2, n_features))
    X = np.vstack((X_class0, X_class1))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y


def generate_noisy(n_samples=400, n_features=6):
    X, y = generate_clean(n_samples, n_features)
    noise = np.random.normal(0, 1.5, X.shape)   
    return X + noise, y


def generate_adversarial(n_samples=400, n_features=6, flip_ratio=0.1):
    X, y = generate_noisy(n_samples, n_features)
    n_flip = int(len(y) * flip_ratio)
    flip_idx = np.random.choice(len(y), size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]   # flip labels
    return X, y

def save_dataset(X, y, path):
    n_features = X.shape[1]  
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y
    df.to_csv(path, index=False)
    print(f"Saved: {path}, shape={df.shape}, columns={df.columns.tolist()}")

# ==================== Main Generator Script ====================
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    X, y = generate_clean()
    save_dataset(X, y, "data/sample_clean.csv")

    X, y = generate_noisy()
    save_dataset(X, y, "data/sample_noisy.csv")

    X, y = generate_adversarial()
    save_dataset(X, y, "data/sample_adversarial.csv")
