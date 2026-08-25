import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ==================================================
# WEEK 3 - TEST 4
# Chi-Square Test of Independence
# Learning Style vs Performance Category
# ==================================================

# --------------------------------------------------
# 1. LOAD AND CLEAN DATA
# --------------------------------------------------

df = pd.read_csv("student_performance.csv")

print("Original Dataset Shape:", df.shape)

df = df.drop_duplicates().copy()

print("Dataset After Cleaning:", df.shape)

# --------------------------------------------------
# 2. CREATE PERFORMANCE CATEGORIES
# --------------------------------------------------

df["PerformanceCategory"] = pd.cut(
    df["ExamScore"],
    bins=[0, 59, 79, 100],
    labels=[
        "Low Performer",
        "Medium Performer",
        "High Performer"
    ]
)

# --------------------------------------------------
# 3. HYPOTHESES
# --------------------------------------------------

print("\n" + "=" * 60)
print("HYPOTHESIS TEST 4: CHI-SQUARE TEST")
print("=" * 60)

print("\nResearch Question:")
print(
    "Is Learning Style significantly associated "
    "with Performance Category?"
)

print("\nH0:")
print(
    "Learning Style and Performance Category "
    "are independent."
)

print("\nH1:")
print(
    "Learning Style and Performance Category "
    "are associated."
)

# --------------------------------------------------
# 4. CONTINGENCY TABLE
# --------------------------------------------------

contingency_table = pd.crosstab(
    df["LearningStyle"],
    df["PerformanceCategory"]
)

print("\nContingency Table:")
print(contingency_table)

# --------------------------------------------------
# 5. CHI-SQUARE TEST
# --------------------------------------------------

chi2, p_value, degrees_freedom, expected = chi2_contingency(
    contingency_table
)

alpha = 0.05

print("\nChi-Square Results")
print("-" * 40)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {degrees_freedom}")
print(f"Significance level: {alpha}")

# --------------------------------------------------
# 6. DECISION
# --------------------------------------------------

if p_value < alpha:

    print("\nDecision: Reject H0")

    print(
        "Conclusion: There is a statistically significant "
        "association between Learning Style and Performance Category."
    )

else:

    print("\nDecision: Fail to reject H0")

    print(
        "Conclusion: There is not enough evidence of an association "
        "between Learning Style and Performance Category."
    )

# --------------------------------------------------
# 7. VISUALIZATION
# --------------------------------------------------

percentage_table = pd.crosstab(
    df["LearningStyle"],
    df["PerformanceCategory"],
    normalize="index"
) * 100

percentage_table.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title(
    "Performance Category Distribution Across Learning Styles",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Learning Style")
plt.ylabel("Percentage of Students")

plt.legend(
    title="Performance Category"
)

plt.tight_layout()

plt.savefig(
    "04_chi_square_learning_style_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()