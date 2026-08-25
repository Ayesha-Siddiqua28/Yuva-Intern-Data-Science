import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# WEEK 4 - DATA PREPARATION
# ============================================================

print("=" * 60)
print("WEEK 4: DATA PREPARATION")
print("=" * 60)

# Load the existing CSV
df = pd.read_csv("student_performance.csv")

print("\nOriginal Dataset Shape:")
print(df.shape)

# Remove duplicates
print("\nDuplicate Records:")
print(df.duplicated().sum())

df = df.drop_duplicates().copy()

print("\nDataset After Removing Duplicates:")
print(df.shape)

# Create performance categories
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
print(df["PerformanceCategory"].value_counts())

# Features
features = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "Age",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "EduTech",
    "StressLevel"
]

# Target
X = df[features]
y = df["PerformanceCategory"]

# ExamScore is NOT included to prevent data leakage

print("\nNumber of Features:")
print(len(features))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))

# Save prepared datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nPrepared datasets saved successfully.")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED")
print("=" * 60)