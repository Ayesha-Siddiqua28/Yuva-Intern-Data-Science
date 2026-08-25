import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# WEEK 5 - COMPREHENSIVE DATA SCIENCE PROJECT
# ============================================================

print("=" * 65)
print("WEEK 5: COMPREHENSIVE DATA SCIENCE PROJECT")
print("=" * 65)

# ------------------------------------------------------------
# 1. FILE SETUP
# ------------------------------------------------------------

os.makedirs("visualizations", exist_ok=True)

# Load existing CSV
df = pd.read_csv("student_performance.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Records:")
print(duplicates)

df = df.drop_duplicates().copy()

print("\nDataset After Removing Duplicates:")
print(df.shape)

# ------------------------------------------------------------
# 3. CREATE PERFORMANCE CATEGORY
# ------------------------------------------------------------

def create_category(score):

    if score < 60:
        return "Low Performer"

    elif score < 80:
        return "Medium Performer"

    else:
        return "High Performer"


df["PerformanceCategory"] = df["ExamScore"].apply(
    create_category
)

print("\nPerformance Category Distribution:")
print(
    df["PerformanceCategory"].value_counts()
)

# ------------------------------------------------------------
# 4. FINAL VISUALIZATION 1
# PERFORMANCE CATEGORY DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="PerformanceCategory",
    order=[
        "Low Performer",
        "Medium Performer",
        "High Performer"
    ]
)

plt.title(
    "Student Performance Category Distribution"
)

plt.xlabel(
    "Performance Category"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    "visualizations/01_performance_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved: visualizations/01_performance_distribution.png"
)

# ------------------------------------------------------------
# 5. FINAL VISUALIZATION 2
# STRESS LEVEL VS EXAM SCORE
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="StressLevel",
    y="ExamScore"
)

plt.title(
    "Exam Score Distribution Across Stress Levels"
)

plt.xlabel(
    "Stress Level"
)

plt.ylabel(
    "Exam Score"
)

plt.tight_layout()

plt.savefig(
    "visualizations/02_stress_vs_exam_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: visualizations/02_stress_vs_exam_score.png"
)

# ------------------------------------------------------------
# 6. FINAL VISUALIZATION 3
# ASSIGNMENT COMPLETION BY PERFORMANCE
# ------------------------------------------------------------

performance_order = [
    "Low Performer",
    "Medium Performer",
    "High Performer"
]

assignment_summary = (
    df.groupby("PerformanceCategory")[
        "AssignmentCompletion"
    ]
    .mean()
    .reindex(performance_order)
)

plt.figure(figsize=(8, 6))

assignment_summary.plot(
    kind="bar"
)

plt.title(
    "Average Assignment Completion by Performance Category"
)

plt.xlabel(
    "Performance Category"
)

plt.ylabel(
    "Average Assignment Completion"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "visualizations/03_assignment_completion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: visualizations/03_assignment_completion.png"
)

# ------------------------------------------------------------
# 7. DESCRIPTIVE SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL DESCRIPTIVE SUMMARY")
print("=" * 65)

summary = (
    df.groupby("PerformanceCategory")[
        [
            "StudyHours",
            "Attendance",
            "Motivation",
            "AssignmentCompletion",
            "StressLevel"
        ]
    ]
    .mean()
    .reindex(performance_order)
    .round(2)
)

print("\nAverage Behavioral Profile:")
print(summary)

# ------------------------------------------------------------
# 8. WEEK 3 STATISTICAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("STATISTICAL FINDINGS FROM WEEK 3")
print("=" * 65)

print("\n1. Pearson Correlation")
print("Study Hours vs Exam Score")
print("Correlation coefficient (r): 0.0042")
print("P-value: 0.639007")
print("Decision: Fail to reject H0")
print(
    "Finding: No statistically significant linear "
    "relationship was detected."
)

print("\n2. Learning Style ANOVA")
print("F-statistic: 1.5729")
print("P-value: 0.193656")
print("Decision: Fail to reject H0")
print(
    "Finding: No sufficient evidence that mean exam "
    "scores differ across Learning Style groups."
)

print("\n3. Stress Level ANOVA")
print("F-statistic: 12.9840")
print("P-value: 0.000002")
print("Decision: Reject H0")
print(
    "Finding: Exam scores differ significantly "
    "among Stress Level groups."
)

print("\n4. Chi-Square Test")
print("Learning Style vs Performance Category")
print("Chi-square statistic: 18.1196")
print("Degrees of freedom: 6")
print("P-value: 0.005940")
print("Decision: Reject H0")
print(
    "Finding: There is a statistically significant "
    "association between Learning Style and "
    "Performance Category."
)

# ------------------------------------------------------------
# 9. WEEK 4 MACHINE LEARNING RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("MACHINE LEARNING FINDINGS FROM WEEK 4")
print("=" * 65)

print("\nModel: Logistic Regression")

print("Target: Performance Category")

print("Training Accuracy: 39.79%")

print("Testing Accuracy: 37.73%")

print("Macro F1-score: 0.37")

print("\nClass Performance:")

print("High Performer:")
print("Precision: 0.38")
print("Recall: 0.48")
print("F1-score: 0.42")

print("\nLow Performer:")
print("Precision: 0.40")
print("Recall: 0.32")
print("F1-score: 0.35")

print("\nMedium Performer:")
print("Precision: 0.36")
print("Recall: 0.33")
print("F1-score: 0.34")

print("\nROC / AUC:")
print("Low Performer AUC: 0.474")
print("Medium Performer AUC: 0.484")
print("High Performer AUC: 0.499")

# ------------------------------------------------------------
# 10. KEY PROJECT INSIGHTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("KEY PROJECT INSIGHTS")
print("=" * 65)

print("""
1. Study hours alone did not show a statistically significant
   linear relationship with exam scores.

2. Learning style did not produce statistically significant
   differences in mean exam scores.

3. Stress level showed a statistically significant difference
   in exam scores across groups.

4. Learning style was statistically associated with the
   performance category.

5. High performers had slightly higher average assignment
   completion and lower average stress than low performers.

6. Logistic Regression achieved 37.73% testing accuracy,
   indicating limited predictive performance.

7. ROC/AUC results were close to 0.50, suggesting that the
   baseline model had weak discriminatory ability.

8. Student performance appears to involve multiple factors
   rather than one simple predictor.
""")

# ------------------------------------------------------------
# 11. STRATEGIC RECOMMENDATIONS
# ------------------------------------------------------------

print("=" * 65)
print("STRATEGIC RECOMMENDATIONS")
print("=" * 65)

print("""
1. Use multiple student indicators instead of relying only
   on study hours.

2. Investigate stress-related factors more deeply because
   Stress Level showed a statistically significant relationship
   with exam scores.

3. Avoid treating learning style as a causal explanation of
   performance.

4. Improve the dataset by adding stronger academic-history
   and engagement features.

5. Compare Logistic Regression with nonlinear models such as
   Decision Trees and Random Forests.

6. Use cross-validation and multiple evaluation metrics when
   developing future predictive models.

7. Consider predicting the continuous ExamScore instead of
   only broad performance categories.
""")

# ------------------------------------------------------------
# 12. FINAL CONCLUSION
# ------------------------------------------------------------

print("=" * 65)
print("FINAL PROJECT CONCLUSION")
print("=" * 65)

print("""
The combined analysis shows that student performance is
multi-dimensional. The statistical tests identified meaningful
patterns involving stress and learning style, while study hours
alone did not show a significant linear relationship with exam
score.

The Logistic Regression model provided a useful baseline but
achieved limited predictive performance. Therefore, stronger
features, nonlinear modelling techniques and additional
validation would be required before using a predictive system
for practical decision-making.

The project demonstrates a complete data science workflow from
data cleaning and visualization to statistical testing,
machine learning evaluation and strategic recommendations.
""")

print("\n" + "=" * 65)
print("WEEK 5 ANALYSIS COMPLETED")
print("=" * 65)

print("\nFinal visualizations created:")

print(
    "1. visualizations/01_performance_distribution.png"
)

print(
    "2. visualizations/02_stress_vs_exam_score.png"
)

print(
    "3. visualizations/03_assignment_completion.png"
)

print("\n" + "=" * 65)