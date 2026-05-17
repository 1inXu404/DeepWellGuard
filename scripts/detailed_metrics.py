#!/usr/bin/env python3
"""Generate detailed individual metrics for each model.

Reads the latest prediction files, calculates precision, recall, and f1 score per class,
generates a detailed classification report, and plots a count-based confusion matrix.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.train.evaluate import (
    compute_metrics, 
    generate_confusion_matrix_counts,
    generate_classification_report,
    generate_roc_curve
)

def _resolve_key(data, *candidates):
    for key in candidates:
        if key in data:
            return data[key]
    available = list(data.keys())
    raise KeyError(f"None of {candidates} found in .npz. Available keys: {available}")

def main():
    file_name_mapping = {
        "LSTM": "lstmmodel_predictions.npz",
        "CNN-LSTM-Attention": "cnnlstmattention_predictions.npz"
    }

    metrics_root = Path("results/metrics")
    figures_root = Path("results/figures")
    reports_root = Path("results/reports")
    
    reports_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    models_to_evaluate = ["LSTM", "CNN-LSTM-Attention"]
    # We will use indices 0 to 6 for the labels as requested by the user.
    class_names = [str(i) for i in range(7)]

    print("Generating detailed metrics, count-based confusion matrices, and ROC curves...\n")

    for model_name in models_to_evaluate:
        target_filename = file_name_mapping[model_name]
        
        found_files = list(metrics_root.rglob(target_filename))
        if not found_files:
            root_file = metrics_root / target_filename
            if root_file.exists():
                found_files = [root_file]
                
        if not found_files:
            print(f"WARN: Could not find prediction file for {model_name}. Skipping.")
            continue
            
        latest_file = sorted(found_files)[-1]
        print(f"[{model_name}] Loading predictions from: {latest_file}")
        
        data = np.load(str(latest_file))
        y_true = _resolve_key(data, "labels", "y_true")
        y_pred = _resolve_key(data, "preds", "y_pred")
        y_proba = _resolve_key(data, "probs", "y_proba")

        # 1. Generate text classification report
        report_str = generate_classification_report(y_true, y_pred)
        
        # Override class names in the string if we want, or just save it directly
        report_path = reports_root / f"{model_name.lower().replace('-', '_')}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Detailed Classification Report for {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_str)
            
        print(f"[{model_name}] Saved classification report -> {report_path}")

        # 2. Save detailed per-class metrics to CSV
        metrics = compute_metrics(y_true, y_pred)
        
        # Create a detailed dataframe for this specific model
        detailed_rows = []
        for i, class_name in enumerate(class_names):
            detailed_rows.append({
                "Class": class_name,
                "Precision": metrics["per_class_precision"][i],
                "Recall": metrics["per_class_recall"][i],
                "F1-Score": metrics["per_class_f1"][i],
            })
            
        df_detailed = pd.DataFrame(detailed_rows)
        csv_path = reports_root / f"{model_name.lower().replace('-', '_')}_per_class_metrics.csv"
        df_detailed.to_csv(csv_path, index=False)
        print(f"[{model_name}] Saved per-class metrics CSV -> {csv_path}")

        # 3. Generate count-based confusion matrix
        fig_name = figures_root / f"confusion_counts_{model_name.replace(' ', '_').replace('-', '_')}.png"
        generate_confusion_matrix_counts(y_true, y_pred, save_path=str(fig_name), class_names=class_names)
        print(f"[{model_name}] Saved count-based confusion matrix -> {fig_name}")
        
        # 4. Generate ROC-AUC curve
        roc_name = figures_root / f"roc_curve_{model_name.replace(' ', '_').replace('-', '_')}.png"
        generate_roc_curve(y_true, y_proba, save_path=str(roc_name), class_names=class_names)
        print(f"[{model_name}] Saved ROC-AUC curves -> {roc_name}")
        
        print("-" * 60)

    print("\nAll detailed metrics generated successfully!")

if __name__ == "__main__":
    main()
