import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ============================================================
# WEEK 4 - MODEL TRAINING
# ============================================================

print("=" * 60)
print("WEEK 4: MODEL TRAINING")
print("=" * 60)

# Load prepared data
X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv("y_train.csv").squeeze()

print("\nTraining Data Shape:")
print(X_train.shape)

# Categorical features
categorical_features = [
    "Resources",
    "Extracurricular",
    "Internet",
    "Gender",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "EduTech",
    "StressLevel"
]

# Numerical features
numeric_features = [
    "StudyHours",
    "Attendance",
    "Motivation",
    "Age",
    "AssignmentCompletion"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Logistic Regression
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

# Train model
print("\nTraining Logistic Regression Model...")

model.fit(
    X_train,
    y_train
)

print("Model Training Completed.")

# Save trained model
joblib.dump(
    model,
    "logistic_regression_model.pkl"
)

print("\nModel saved as:")
print("logistic_regression_model.pkl")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)