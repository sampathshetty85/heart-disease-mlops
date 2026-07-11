import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "heart.csv")


def download_dataset():
    print("Fetching Heart Disease UCI dataset (id=45)...")
    heart = fetch_ucirepo(id=45)

    features = heart.data.features
    targets = heart.data.targets

    df = pd.concat([features, targets], axis=1)

    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows x {len(df.columns)} columns to {OUTPUT_PATH}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df


if __name__ == "__main__":
    download_dataset()
