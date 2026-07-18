import os
import pickle
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Create Images Folder
# ===========================

os.makedirs("static/images", exist_ok=True)

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("dataset/loan_dataset_20000.csv")

# ===========================
# Load Model
# ===========================

model = pickle.load(open("model/model.pkl", "rb"))

features = pickle.load(open("model/features.pkl", "rb"))

# ===========================
# Chart Style
# ===========================

sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (8,5)

plt.rcParams["figure.dpi"] = 120

# ======================================
# 1 Loan Repayment Distribution
# ======================================

plt.figure()

df["loan_paid_back"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["#ef4444", "#22c55e"],
    startangle=90
)

plt.ylabel("")

plt.title("Loan Repayment Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/default_distribution.png"
)

plt.close()

# ======================================
# 2 Employment Status
# ======================================

plt.figure()

sns.countplot(
    data=df,
    x="employment_status",
    hue="employment_status",
    palette="viridis",
    legend=False
)

plt.xticks(rotation=20)

plt.title("Employment Status Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/employment.png"
)

plt.close()

# ======================================
# 3 Credit Score Distribution
# ======================================

plt.figure()

sns.histplot(
    df["credit_score"],
    bins=30,
    kde=True,
    color="royalblue"
)

plt.title("Credit Score Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/credit_score.png"
)

plt.close()

# ======================================
# 4 Loan Amount Distribution
# ======================================

plt.figure()

sns.histplot(
    df["loan_amount"],
    bins=30,
    kde=True,
    color="green"
)

plt.title("Loan Amount Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/loan_amount.png"
)

plt.close()

# ======================================
# 5 Monthly Income Distribution
# ======================================

plt.figure()

sns.histplot(
    df["monthly_income"],
    bins=30,
    kde=True,
    color="purple"
)

plt.title("Monthly Income Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/monthly_income.png"
)

plt.close()

# ======================================
# 6 Loan Purpose Analysis
# ======================================

plt.figure(figsize=(9,5))

sns.countplot(
    data=df,
    x="loan_purpose",
    hue="loan_purpose",
    palette="Set2",
    legend=False
)

plt.xticks(rotation=30)

plt.title("Loan Purpose Analysis")

plt.tight_layout()

plt.savefig(
    "static/images/loan_purpose.png"
)

plt.close()

# ======================================
# 7 Gender Distribution
# ======================================

plt.figure()

sns.countplot(
    data=df,
    x="gender",
    hue="gender",
    palette="pastel",
    legend=False
)

plt.title("Gender Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/gender.png"
)

plt.close()

# ======================================
# 8 Loan Grade Distribution
# ======================================

plt.figure(figsize=(10,5))

sns.countplot(
    data=df,
    x="grade_subgrade",
    order=sorted(df["grade_subgrade"].unique()),
    hue="grade_subgrade",
    palette="coolwarm",
    legend=False
)

plt.xticks(rotation=90)

plt.title("Loan Grade Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/grade.png"
)

plt.close()

# ======================================
# 9 Debt To Income Ratio
# ======================================

plt.figure()

sns.histplot(
    df["debt_to_income_ratio"],
    bins=30,
    kde=True,
    color="#2563eb"
)

plt.title("Debt To Income Ratio Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/debt_ratio.png"
)

plt.close()

# ======================================
# 10 Interest Rate Distribution
# ======================================

plt.figure()

sns.histplot(
    df["interest_rate"],
    bins=30,
    kde=True,
    color="#f59e0b"
)

plt.title("Interest Rate Distribution")

plt.tight_layout()

plt.savefig(
    "static/images/interest_rate.png"
)

plt.close()
# ======================================
# 11 Correlation Heatmap
# ======================================

plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=["int64", "float64"])

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "static/images/heatmap.png"
)

plt.close()


# ======================================
# 12 Feature Importance
# ======================================

try:

    importance = model.feature_importances_

    feature_df = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(

        data=feature_df.head(10),

        x="Importance",

        y="Feature",

        palette="viridis"

    )

    plt.title("Top 10 Feature Importance")

    plt.xlabel("Importance Score")

    plt.ylabel("Feature")

    plt.tight_layout()

    plt.savefig(
        "static/images/feature_importance.png"
    )

    plt.close()

except Exception as e:

    print("Feature Importance Error :", e)


# ======================================
# Finished
# ======================================

print("="*50)
print("LoanAssist Charts Generated Successfully")
print("="*50)
print("Charts Saved Inside : static/images/")
print("""
Generated Charts

1. default_distribution.png
2. employment.png
3. credit_score.png
4. loan_amount.png
5. monthly_income.png
6. loan_purpose.png
7. gender.png
8. grade.png
9. debt_ratio.png
10. interest_rate.png
11. heatmap.png
12. feature_importance.png
""")
print("="*50)