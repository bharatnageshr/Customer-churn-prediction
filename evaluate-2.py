#!/usr/bin/env python3
\"\"\"Evaluate saved Random Forest model and display metrics & plots:
Usage:
    python src/evaluate.py --model models/best_model.pkl --processed_dir data/processed
\"\"\"
from __future__ import annotations
import argparse, pickle, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    model = pickle.load(open(args.model,'rb'))
    p = Path(args.processed_dir)
    X_test = pickle.load(open(p / 'X_test.pkl','rb'))
    y_test = pickle.load(open(p / 'y_test.pkl','rb'))

    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with open(outdir / 'evaluation.json', 'w', encoding='utf8') as f:
        json.dump({'accuracy': acc, 'confusion_matrix': cm.tolist(), 'classification_report': report}, f, indent=2)

    print(f\"Accuracy: {acc:.4f}\")
    print(\"Confusion matrix:\")
    print(cm)
    # plot ROC if probabilities exist
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / 'roc_curve.png')
        plt.close()
    print(f\"Saved evaluation JSON and ROC (if available) to {outdir.resolve()}\")

if __name__ == '__main__':
    main()
