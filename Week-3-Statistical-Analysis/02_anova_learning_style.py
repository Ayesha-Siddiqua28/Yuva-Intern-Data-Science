import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# ==================================================
# WEEK 3 - TEST 2
# One-Way ANOVA
# Learning Style vs Exam Score
# ==================================================

# --------------------------------------------------
# 1. LOAD AND CLEAN DATA
# --------------------------------------------------

df = pd.read_csv("student_performance.csv")

print("Original Dataset Shape:", df.shape)

df = df.drop_duplicates().copy()

print("Dataset After Cleaning:", df.shape)

# --------------------------------------------------
# 2. HYPOTHESES
# --------------------------------------------------

print("\n" + "=" * 60)
print("HYPOTHESIS TEST 2: ONE-WAY ANOVA")
print("=" * 60)

print("\nResearch Question:")
print("Do exam scores differ significantly across Learning Style groups?")

print("\nH0:")
print("The mean exam score is equal across all Learning Style groups.")

print("\nH1:")
print("At least one Learning Style group has a different mean exam score.")

# --------------------------------------------------
# 3. CREATE GROUPS
# --------------------------------------------------

groups = []

for style in sorted(df["LearningStyle"].unique()):

    scores = df.loc[
        df["LearningStyle"] == style,
        "ExamScore"
    ]

    groups.append(scores)

    print(
        f"Learning Style {style}: "
        f"n = {len(scores)}, "
        f"mean = {scores.mean():.2f}"
    )

# --------------------------------------------------
# 4. ONE-WAY ANOVA
# --------------------------------------------------

f_statistic, p_value = f_oneway(*groups)

alpha = 0.05

print("\nANOVA Results")
print("-" * 40)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significance level: {alpha}")

# --------------------------------------------------
# 5. DECISION
# --------------------------------------------------

if p_value < alpha:

    print("\nDecision: Reject H0")

    print(
        "Conclusion: There is a statistically significant "
        "difference in exam scores among Learning Style groups."
    )

else:

    print("\nDecision: Fail to reject H0")

    print(
        "Conclusion: There is not enough evidence to conclude "
        "that mean exam scores differ among Learning Style groups."
    )

# --------------------------------------------------
# 6. VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="LearningStyle",
    y="ExamScore"
)

sns.stripplot(
    data=df,
    x="LearningStyle",
    y="ExamScore",
    alpha=0.15,
    size=2
)

plt.title(
    "Exam Score Distribution Across Learning Styles",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Learning Style")
plt.ylabel("Exam Score")

plt.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    "02_learning_style_anova.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
