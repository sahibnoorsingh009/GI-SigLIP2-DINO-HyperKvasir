import pandas as pd
from pathlib import Path

# IMPORTANT:
# This script can read prediction CSVs, but for training you should use validation mistakes,
# not test mistakes. Test mistakes are only for analysis/reporting.

pred_csv = "results/final_eval/test_predictions.csv"
out_csv = "metadata/hard_examples_from_test_predictions_ANALYSIS_ONLY.csv"

pred = pd.read_csv(pred_csv)

print("Prediction columns:", pred.columns.tolist())
print(pred.head())

true_col = "y_true"
pred_col = "y_pred"

hard = pred[pred[true_col] != pred[pred_col]].copy()
hard["hard_reason"] = "wrong_prediction"
hard["hard_weight"] = 3.0

hard.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Hard examples:", len(hard))

if len(hard) > 0:
    print("\nTop confusions:")
    print(
        hard.groupby([true_col, pred_col])
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_string()
    )

    print("\nHard examples preview:")
    print(hard[[true_col, pred_col, "confidence"]].head(30).to_string(index=False))
