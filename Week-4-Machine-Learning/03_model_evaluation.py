import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import joblib

# ============================================================
# WEEK 4 - MODEL EVALUATION
# ============================================================

print("=" * 60)
print("WEEK 4: MODEL EVALUATION")
print("=" * 60)

# Load test data
X_test = pd.read_csv("X_test.csv")

y_test = pd.read_csv("y_test.csv").squeeze()

# Load trained model
model = joblib.load(
    "logistic_regression_model.pkl"
)

print("\nTest Data Shape:")
print(X_test.shape)

# ------------------------------------------------------------
# 1. PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# 2. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nTesting Accuracy: {accuracy:.4f}"
)

# ------------------------------------------------------------
# 3. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ------------------------------------------------------------
# 4. CREATE VISUALIZATION FOLDER
# ------------------------------------------------------------

os.makedirs(
    "visualizations",
    exist_ok=True
)

# ------------------------------------------------------------
# 5. CONFUSION MATRIX
# ------------------------------------------------------------

labels = [
    "Low Performer",
    "Medium Performer",
    "High Performer"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Low",
        "Medium",
        "High"
    ],
    yticklabels=[
        "Low",
        "Medium",
        "High"
    ]
)

plt.title(
    "Confusion Matrix - Student Performance Classification"
)

plt.xlabel(
    "Predicted Performance"
)

plt.ylabel(
    "Actual Performance"
)

plt.tight_layout()

plt.savefig(
    "visualizations/01_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved: visualizations/01_confusion_matrix.png"
)

# ------------------------------------------------------------
# 6. ROC CURVE
# ------------------------------------------------------------

y_test_binary = label_binarize(
    y_test,
    classes=labels
)

y_probability = model.predict_proba(
    X_test
)

plt.figure(
    figsize=(9, 7)
)

for i, class_name in enumerate(labels):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        y_probability[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve - Student Performance Classification"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "visualizations/02_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: visualizations/02_roc_curve.png"
)

# ------------------------------------------------------------
# 7. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("WEEK 4 EVALUATION COMPLETED")
print("=" * 60)

print(
    f"\nTesting Accuracy: {accuracy:.4f}"
)

print("\nVisualizations:")
print("1. Confusion Matrix")
print("2. ROC Curve")

print("\nModel: Logistic Regression")
print("Target: Performance Category")

print("\n" + "=" * 60)