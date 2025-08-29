# Customer-churn-prediction
# Churn Prediction — EDA + SMOTE + Random Forest

This repository is an end-to-end scaffold for churn prediction on customer datasets, focused on handling imbalance and producing explainable EDA visualizations.

## Project structure
```
churn-prediction/
├── src/
│   ├── preprocess.py   # Load, clean, encode, scale, and SMOTE-balance train set
│   ├── eda.py          # Produce pie chart, bar plots, correlation heatmap
│   ├── train.py        # Train Random Forest with GridSearchCV and save best model
│   └── evaluate.py     # Evaluate saved model and save metrics/ROC plot
├── data/               # Place your raw CSV here (e.g., data/churn.csv)
├── data/processed/     # Outputs: X_train.pkl, y_train.pkl, X_test.pkl, y_test.pkl, preprocessor.pkl
├── models/             # Saved model (best_model.pkl)
├── reports/            # EDA plots and evaluation reports
├── README.md
└── requirements.txt
```

## Quickstart
1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Place your dataset (CSV) at `data/churn.csv` and ensure it has a binary `Churn` column (values 0/1 or Yes/No). If `Yes/No` convert to 1/0 before running or the script will handle object targets by using them as-is in stratify.

3. Preprocess & balance with SMOTE:
```bash
python src/preprocess.py --data data/churn.csv --target Churn --out_dir data/processed --test-size 0.2
```

4. Run EDA to generate plots:
```bash
python src/eda.py --processed_dir data/processed --out_dir reports/eda_plots
```

5. Train Random Forest and tune hyperparameters:
```bash
python src/train.py --processed_dir data/processed --out_dir models
```

6. Evaluate model:
```bash
python src/evaluate.py --model models/best_model.pkl --processed_dir data/processed --out_dir reports
```

## Notes & Tips
- The preprocessing pipeline uses `OneHotEncoder` for categorical variables and `StandardScaler` for numerics.
- SMOTE is applied only on the training set to prevent leakage.
- To reproduce a specific accuracy (e.g., **91.46%**), make sure to use the same data split, same preprocessing steps, and similar hyperparameter grid. The scaffold prints and saves all artifacts for reproducibility.
- For large datasets, consider using sparse encodings or category reduction to save memory.

## Requirements
```
pandas
numpy
scikit-learn
imblearn
matplotlib
```

