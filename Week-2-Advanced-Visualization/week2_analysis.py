import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Week 1 dataset
df = pd.read_csv("student_performance.csv")

# Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())
# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

# Remove duplicate records
df = df.drop_duplicates().copy()

print("\nDataset after removing duplicates:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())
# --------------------------------------------------
# VISUALIZATION 1 - CORRELATION HEATMAP
# --------------------------------------------------

# Select numerical variables
numeric_columns = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Age",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "ExamScore",
    "EduTech",
    "StressLevel",
    "FinalGrade"
]

# Calculate correlation matrix
correlation_matrix = df[numeric_columns].corr()

# Create the heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation"}
)

plt.title(
    "Correlation Between Student Learning Factors and Academic Performance",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Variables")
plt.ylabel("Variables")

plt.tight_layout()

# Display the heatmap
plt.show()

# Save the heatmap
plt.savefig(
    "01_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
# --------------------------------------------------
# VISUALIZATION 2
# Exam Score Distribution by Learning Style
# --------------------------------------------------

plt.figure(figsize=(10, 6))

# Violin plot
sns.violinplot(
    data=df,
    x="LearningStyle",
    y="ExamScore",
    inner="box"
)

plt.title(
    "Distribution of Exam Scores Across Learning Styles",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Learning Style", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)

plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

# Save the visualization
plt.savefig(
    "02_exam_score_by_learning_style.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# --------------------------------------------------
# VISUALIZATION 3
# Motivation × Stress Level × Exam Score
# --------------------------------------------------

# Calculate average exam score for each
# combination of Motivation and Stress Level
performance_matrix = df.pivot_table(
    values="ExamScore",
    index="Motivation",
    columns="StressLevel",
    aggfunc="mean"
)

# Create heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    performance_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Average Exam Score"}
)

plt.title(
    "Average Exam Score by Motivation and Stress Level",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Stress Level", fontsize=12)
plt.ylabel("Motivation Level", fontsize=12)

plt.tight_layout()

# Save the visualization
plt.savefig(
    "03_motivation_stress_exam_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# --------------------------------------------------
# VISUALIZATION 4
# Profile of Low, Medium and High Performing Students
# --------------------------------------------------

# Create performance groups based on ExamScore
df["PerformanceGroup"] = pd.cut(
    df["ExamScore"],
    bins=[0, 59, 79, 100],
    labels=["Low Performers", "Medium Performers", "High Performers"]
)

# Variables used to describe student behavior
profile_columns = [
    "StudyHours",
    "Attendance",
    "Motivation",
    "AssignmentCompletion",
    "StressLevel"
]

# Calculate average values for each performance group
group_profile = df.groupby(
    "PerformanceGroup",
    observed=False
)[profile_columns].mean()

# Standardize each variable so they can be compared
# despite having different numerical scales
group_profile_standardized = (
    group_profile - group_profile.mean()
) / group_profile.std()

# Create heatmap
plt.figure(figsize=(11, 6))

sns.heatmap(
    group_profile_standardized,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Relative Level (Standardized)"}
)

plt.title(
    "Behavioral Profile of Low, Medium and High Performing Students",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Student Behavior Factors", fontsize=12)
plt.ylabel("Performance Group", fontsize=12)

plt.tight_layout()

# Save visualization
plt.savefig(
    "04_performance_group_profile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Print the original group averages
print("\nAverage Behavioral Profile by Performance Group:")
print(group_profile.round(2))
# --------------------------------------------------
# VISUALIZATION 5
# Assignment Completion vs Exam Performance
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="AssignmentCompletion",
    y="ExamScore"
)

plt.title(
    "Exam Score Distribution by Assignment Completion",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Assignment Completion Level", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)

plt.grid(axis="y", alpha=0.2)
plt.tight_layout()

# Save visualization
plt.savefig(
    "05_assignment_completion_vs_exam_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# --------------------------------------------------
# VISUALIZATION 6
# High vs Low Performer Behavioral Comparison
# --------------------------------------------------

# Create two groups
df["PerformanceCategory"] = np.where(
    df["ExamScore"] >= 80,
    "High Performers",
    np.where(
        df["ExamScore"] < 60,
        "Low Performers",
        "Medium Performers"
    )
)

# Calculate average behavioral factors
behavior_columns = [
    "StudyHours",
    "Attendance",
    "Motivation",
    "AssignmentCompletion",
    "StressLevel"
]

behavior_comparison = df.groupby(
    "PerformanceCategory"
)[behavior_columns].mean()

# Reorder groups
order = [
    "Low Performers",
    "Medium Performers",
    "High Performers"
]

behavior_comparison = behavior_comparison.reindex(order)

# Standardize values for visual comparison
behavior_standardized = (
    behavior_comparison - behavior_comparison.mean()
) / behavior_comparison.std()

# Create heatmap
plt.figure(figsize=(12, 6))

sns.heatmap(
    behavior_standardized,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Relative Level"}
)

plt.title(
    "Behavioral Differences Across Student Performance Groups",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Behavioral Factors", fontsize=12)
plt.ylabel("Performance Category", fontsize=12)

plt.tight_layout()

# Save visualization
plt.savefig(
    "06_high_vs_low_performer_behavior.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nBehavioral Comparison:")
print(behavior_comparison.round(2))
# --------------------------------------------------
# VISUALIZATION 7
# Interactive Student Performance Analysis
# --------------------------------------------------

import plotly.express as px

fig = px.scatter(
    df,
    x="StudyHours",
    y="ExamScore",
    color="PerformanceCategory",
    size="AssignmentCompletion",
    hover_data=[
        "Attendance",
        "Motivation",
        "StressLevel",
        "LearningStyle"
    ],
    title="Interactive Analysis of Student Performance",
    labels={
        "StudyHours": "Study Hours",
        "ExamScore": "Exam Score",
        "PerformanceCategory": "Performance Group",
        "AssignmentCompletion": "Assignment Completion"
    }
)

fig.update_layout(
    template="plotly_white",
    title_x=0.5
)

fig.show()

# Save interactive visualization
fig.write_html("07_interactive_student_performance.html")