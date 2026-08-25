import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# ==================================================
# WEEK 3 - TEST 1
# Pearson Correlation
# Study Hours vs Exam Score
# ==================================================

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("student_performance.csv")

print("Original Dataset Shape:", df.shape)

# --------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates().copy()

print("Dataset After Cleaning:", df.shape)

# --------------------------------------------------
# 3. HYPOTHESES
# --------------------------------------------------

print("\n" + "=" * 60)
print("HYPOTHESIS TEST 1: PEARSON CORRELATION")
print("=" * 60)

print("\nResearch Question:")
print("Is there a significant linear relationship between Study Hours and Exam Score?")

print("\nH0:")
print("There is no significant linear relationship between Study Hours and Exam Score.")

print("\nH1:")
print("There is a significant linear relationship between Study Hours and Exam Score.")

# --------------------------------------------------
# 4. PEARSON CORRELATION
# --------------------------------------------------

r_value, p_value = pearsonr(
    df["StudyHours"],
    df["ExamScore"]
)

alpha = 0.05

print("\nPearson Correlation Results")
print("-" * 40)

print(f"Correlation coefficient (r): {r_value:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significance level: {alpha}")

# --------------------------------------------------
# 5. DECISION
# --------------------------------------------------

if p_value < alpha:
    print("\nDecision: Reject H0")
    print("Conclusion: There is a statistically significant linear relationship.")
else:
    print("\nDecision: Fail to reject H0")
    print("Conclusion: There is not enough evidence of a statistically significant linear relationship.")

# --------------------------------------------------
# 6. 95% CONFIDENCE INTERVAL
# --------------------------------------------------

n = len(df)

z = np.arctanh(r_value)
se = 1 / np.sqrt(n - 3)

z_critical = 1.96

z_lower = z - z_critical * se
z_upper = z + z_critical * se

r_lower = np.tanh(z_lower)
r_upper = np.tanh(z_upper)

print("\n95% Confidence Interval")
print("-" * 40)

print(f"Lower bound: {r_lower:.4f}")
print(f"Upper bound: {r_upper:.4f}")

# --------------------------------------------------
# 7. VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="StudyHours",
    y="ExamScore",
    scatter_kws={"alpha": 0.35},
    line_kws={"linewidth": 2}
)

plt.title(
    "Study Hours vs Exam Score: Pearson Correlation",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.text(
    0.02,
    0.95,
    f"r = {r_value:.4f}\n"
    f"p = {p_value:.4f}\n"
    f"95% CI = [{r_lower:.4f}, {r_upper:.4f}]",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)

plt.grid(alpha=0.2)
plt.tight_layout()

# Save directly in the current folder
plt.savefig(
    "01_study_hours_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()