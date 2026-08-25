import pandas as pd

# Load the dataset
df = pd.read_csv("student_performance.csv")

# Display basic information
print("DATASET LOADED SUCCESSFULLY")
print("=" * 40)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Check data types
print("\nData types:")
print(df.dtypes)
# Remove duplicate rows
df_cleaned = df.drop_duplicates()

print("\nData cleaning completed.")
print("Original number of rows:", len(df))
print("Rows after removing duplicates:", len(df_cleaned))
print("Number of rows removed:", len(df) - len(df_cleaned))
# Summary statistics of cleaned data
print("\nSummary statistics:")
print(df_cleaned.describe())
# Visualization 1: Distribution of Exam Scores

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))

sns.histplot(df_cleaned["ExamScore"], bins=10, kde=True)

plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("exam_score_distribution.png")
plt.show()
# Visualization 2: Study Hours vs Exam Score

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df_cleaned,
    x="StudyHours",
    y="ExamScore",
    alpha=0.5
)

plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.savefig("study_hours_vs_exam_score.png")
plt.show()
# Visualization 3: Correlation Heatmap

plt.figure(figsize=(12, 8))

correlation = df_cleaned.corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Student Performance Dataset")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
# Visualization 4: Box Plot of Exam Scores

plt.figure(figsize=(8, 5))

sns.boxplot(x=df_cleaned["ExamScore"])

plt.title("Box Plot of Exam Scores")
plt.xlabel("Exam Score")

plt.tight_layout()
plt.savefig("exam_score_boxplot.png")
plt.show()
# Save the cleaned dataset
df_cleaned.to_csv("student_performance_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
# Key EDA insights

print("\n--- KEY EDA INSIGHTS ---")

print("Average Study Hours:", round(df_cleaned["StudyHours"].mean(), 2))
print("Average Attendance:", round(df_cleaned["Attendance"].mean(), 2))
print("Average Exam Score:", round(df_cleaned["ExamScore"].mean(), 2))

print("\nExam Score Range:")
print("Minimum:", df_cleaned["ExamScore"].min())
print("Maximum:", df_cleaned["ExamScore"].max())
print("Median:", df_cleaned["ExamScore"].median())

print("\nCorrelation with Exam Score:")
print(
    df_cleaned[
        ["StudyHours", "Attendance", "Motivation",
         "AssignmentCompletion", "ExamScore"]
    ].corr()["ExamScore"].sort_values(ascending=False)
)