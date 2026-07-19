import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LoanAssist EMI Default Prediction Training
# ==========================================

print("=" * 60)
print("LoanAssist EMI Default Prediction Model")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(
    "dataset/loan_dataset_20000.csv"
)

print("\nDataset Shape :", df.shape)

# ==========================================
# Handle Missing Values
# ==========================================

df.fillna(0, inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================

categorical_columns = [

    "employment_status",
    "grade_subgrade"

]

encoders = {}

for col in categorical_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = encoder

print("\nCategorical Encoding Completed")

# ==========================================
# Select Only Important Features
# ==========================================

selected_features = [

    "employment_status",
    "debt_to_income_ratio",
    "credit_score",
    "grade_subgrade",
    "interest_rate",
    "current_balance",
    "installment",
    "total_credit_limit",
    "loan_amount",
    "monthly_income",
    "num_of_open_accounts",
    "num_of_delinquencies"

]

# ==========================================
# Features and Target
# ==========================================

X = df[selected_features]

y = df["loan_paid_back"]

print(df["loan_paid_back"].value_counts())

print(df["loan_paid_back"].value_counts(normalize=True) * 100)

print("\nSelected Features")

for col in X.columns:
    print(col)

# ==========================================
# Save Feature Order
# ==========================================

features = X.columns.tolist()
# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed")

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTrain Test Split Completed")

print("Training Records :", len(X_train))
print("Testing Records  :", len(X_test))

# ==========================================
# Random Forest Model
# ==========================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=5,

    min_samples_leaf=8,

    class_weight={0:3,1:1},

    random_state=42,

    n_jobs=-1

)

# ==========================================
# Model Training
# ==========================================

print("\nTraining Started...")

model.fit(

    X_train,

    y_train

)

print("Training Completed Successfully")
# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(

    X_test

)

# ==========================================
# Model Evaluation
# ==========================================

accuracy = accuracy_score(

    y_test,

    y_pred

)

precision = precision_score(

    y_test,

    y_pred

)

recall = recall_score(

    y_test,

    y_pred

)

f1 = f1_score(

    y_test,

    y_pred

)

train_accuracy = model.score(

    X_train,

    y_train

)

test_accuracy = model.score(

    X_test,

    y_test

)


# ==========================================
# Print Results
# ==========================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Training Accuracy : {train_accuracy*100:.2f}%")
print(f"Testing Accuracy  : {test_accuracy*100:.2f}%")

print()

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred

    )

)

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        y_pred

    )

)

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})

importance.sort_values(

    by="Importance",

    ascending=False,

    inplace=True

)

print("\nTop Important Features")

print(importance)
# ==========================================
# Create Model Folder
# ==========================================

os.makedirs(

    "model",

    exist_ok=True

)

# ==========================================
# Save Model
# ==========================================

pickle.dump(

    model,

    open(

        "model/model.pkl",

        "wb"

    )

)

# ==========================================
# Save Scaler
# ==========================================

pickle.dump(

    scaler,

    open(

        "model/scaler.pkl",

        "wb"

    )

)

# ==========================================
# Save Encoders
# ==========================================

pickle.dump(

    encoders,

    open(

        "model/encoder.pkl",

        "wb"

    )

)

# ==========================================
# Save Feature Order
# ==========================================

pickle.dump(

    features,

    open(

        "model/features.pkl",

        "wb"

    )

)

# ==========================================
# Save Performance Metrics
# ==========================================

pickle.dump(

    round(accuracy * 100, 2),

    open(

        "model/accuracy.pkl",

        "wb"

    )

)

pickle.dump(

    round(precision * 100, 2),

    open(

        "model/precision.pkl",

        "wb"

    )

)

pickle.dump(

    round(recall * 100, 2),

    open(

        "model/recall.pkl",

        "wb"

    )

)

pickle.dump(

    round(f1 * 100, 2),

    open(

        "model/f1.pkl",

        "wb"

    )

)

# ==========================================
# Training Completed
# ==========================================

print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("""

Saved Files

 model.pkl
 scaler.pkl
 encoder.pkl
 features.pkl
 accuracy.pkl
 precision.pkl
 recall.pkl
 f1.pkl

Training Completed Successfully.

""")


print(df["loan_paid_back"].value_counts())

print(df["loan_paid_back"].value_counts(normalize=True)*100)