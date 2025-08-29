#!/usr/bin/env python3
\"\"\"Preprocess dataset for churn prediction:
- Loads CSV
- Handles missing values
- Encodes categorical features (one-hot / label)
- Scales numeric features
- Balances classes using SMOTE
- Saves processed train/test splits and scaler/encoder objects
Usage:
    python src/preprocess.py --data data/churn.csv --target Churn --out_dir data/processed --test-size 0.2 --random-state 42
\"\"\"
from __future__ import annotations
import argparse, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

def load_data(path: Path):
    df = pd.read_csv(path)
    return df

def build_preprocessor(df: pd.DataFrame, numeric_strategy: str = "median", drop_cols=None):
    X = df.drop(columns=drop_cols) if drop_cols else df.copy()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_strategy)),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ], remainder='drop')
    return preprocessor, num_cols, cat_cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to raw CSV file")
    parser.add_argument("--target", type=str, default="Churn", help="Target column name (default: Churn)")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory for processed artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.data))
    if args.target not in df.columns:
        raise SystemExit(f\"Target column '{args.target}' not found in data. Columns: {list(df.columns)[:20]}\")

    # Optionally drop identifier columns (heuristic)
    drop_cols = [c for c in df.columns if c.lower() in ('customerid', 'id', 'custid', 'row')]
    if args.target in drop_cols:
        drop_cols.remove(args.target)

    y = df[args.target].copy()
    X = df.drop(columns=[args.target] + drop_cols) if drop_cols else df.drop(columns=[args.target])

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Fit preprocessor on full data for reproducibility (you may fit only on train in other workflows)
    X_processed = preprocessor.fit_transform(X)

    # Convert processed array back to DataFrame for readability (get feature names)
    # build feature names
    import numpy as np
    num_features = num_cols
    # get onehot names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    onehot_cols = []
    try:
        categories = ohe.categories_
        for i, cats in enumerate(categories):
            colname = cat_cols[i]
            onehot_cols.extend([f\"{colname}__{c}\" for c in cats])
    except Exception:
        onehot_cols = [f\"cat_{i}\" for i in range(X_processed.shape[1] - len(num_features))]

    feature_names = num_features + onehot_cols
    X_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    # Balance using SMOTE on training set only
    sm = SMOTE(random_state=args.random_state, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Save artifacts
    with open(outdir / 'X_train.pkl', 'wb') as f:
        pickle.dump(X_res, f)
    with open(outdir / 'y_train.pkl', 'wb') as f:
        pickle.dump(y_res, f)
    with open(outdir / 'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(outdir / 'y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open(outdir / 'preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print(f\"Preprocessing complete. Saved processed datasets to {outdir.resolve()}\")
    print(f\"Train set after SMOTE: {len(y_res)} samples. Test set: {len(y_test)} samples.\")

if __name__ == '__main__':
    main()
