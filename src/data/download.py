import os
import sys
import pandas as pd
from ucimlrepo import fetch_ucirepo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from report_writer import write_report  # noqa: E402

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "heart.csv")


def download_dataset():
    print("=== download.py ===")
    print("Fetching Heart Disease UCI dataset (id=45)...")
    heart = fetch_ucirepo(id=45)

    features = heart.data.features
    targets = heart.data.targets
    df = pd.concat([features, targets], axis=1)

    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    print(f"Saved: {len(df)} rows x {len(df.columns)} columns → data/raw/heart.csv")
    print(f"Columns : {list(df.columns)}")
    print(f"Missing : {dict(missing_cols) if len(missing_cols) else 'none'}")
    print("download.py complete.")

    _write_validation_report(df, missing_cols)
    return df


def _write_validation_report(df, missing_cols):
    lines = [
        "STEP 1.1 — Dataset Download",
        "",
        "Source       : UCI Machine Learning Repository (id=45)",
        "Dataset      : Heart Disease UCI",
        "",
        f"Rows         : {len(df)}   (expected 303)",
        f"Columns      : {len(df.columns)}   (expected 14)",
        f"Column names : {list(df.columns)}",
        "",
        "Missing values:",
    ]
    if len(missing_cols):
        for col, count in missing_cols.items():
            lines.append(f"  {col:<8}: {count} missing")
    else:
        lines.append("  None")

    lines += [
        "",
        "Shape check  : " + ("PASS" if df.shape == (303, 14) else "FAIL"),
        "File saved   : data/raw/heart.csv",
        "",
        "Status       : COMPLETE",
    ]
    write_report("step_1_1_download", lines)


if __name__ == "__main__":
    download_dataset()
