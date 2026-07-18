from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file
)

import pandas as pd
import pickle
import os

from fpdf import FPDF


app = Flask(__name__)

app.secret_key = "loanassist"

# =====================================
# Folder Configuration
# =====================================

REPORT_FOLDER = "static/reports"
IMAGE_FOLDER = "static/images"


os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)



# =====================================
# Load Dataset
# =====================================

def load_data():

    return pd.read_csv(
        "dataset/loan_dataset_20000.csv"
    )

# =====================================
# Prediction History CSV
# =====================================

PREDICTION_FILE = "static/reports/prediction_history.csv"

if not os.path.exists(PREDICTION_FILE):

    pd.DataFrame(columns=[

        "date",
        "prediction",
        "risk",
        "default_probability",
        "paid_probability"

    ]).to_csv(
        PREDICTION_FILE,
        index=False
    )

# =====================================
# Load ML Files
# =====================================

model = pickle.load(
    open("model/model.pkl", "rb")
)

scaler = pickle.load(
    open("model/scaler.pkl", "rb")
)

encoders = pickle.load(
    open("model/encoder.pkl", "rb")
)

features = pickle.load(
    open("model/features.pkl", "rb")
)

# =====================================
# Load Performance Metrics
# =====================================

model_accuracy = pickle.load(
    open("model/accuracy.pkl", "rb")
)

precision = pickle.load(
    open("model/precision.pkl", "rb")
)

recall = pickle.load(
    open("model/recall.pkl", "rb")
)

f1 = pickle.load(
    open("model/f1.pkl", "rb")
)

# =====================================
# Dashboard Variables
# =====================================

recent_activity = [
    "System Started Successfully"
]

# =====================================
# Login
# =====================================

@app.route("/", methods=["GET", "POST"])

def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        if username == "admin" and password == "Admin@123":

            session["user"] = username

            recent_activity.append(
                "Admin Logged In"
            )

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid Username or Password"
        )

    return render_template(
        "login.html"
    )

# =====================================
# Dashboard
# =====================================

@app.route("/dashboard")

def dashboard():

    if "user" not in session:

        return redirect(
            url_for("login")
        )

    df = load_data()

    total_customers = len(df)

    high_risk = len(
        df[df["loan_paid_back"] == 0]
    )

    average_risk = round(

        (high_risk / total_customers) * 100,

        2

    )

    customers = []

    for i, row in df.head(5).iterrows():

        score = row["credit_score"]

        if score < 600:

            risk = "High Risk"

        elif score < 750:

            risk = "Medium Risk"

        else:

            risk = "Low Risk"

        customers.append({

            "id": i + 1,

            "name": f"Customer {i+1}",

            "credit": score,

            "risk": risk

        })

    prediction_df = pd.read_csv(PREDICTION_FILE)

    total_predictions = len(prediction_df)

    if total_predictions > 0:

        last_prediction = prediction_df.iloc[-1]["date"]

    else:

        last_prediction = "No Prediction Yet"

    csv_notifications = len([
        x for x in recent_activity
        if "CSV" in x
    ])

    return render_template(

        "dashboard.html",

        total_customers=total_customers,

        high_risk=high_risk,

        average_risk=average_risk,

        customers=customers,

        total_predictions=total_predictions,

        last_prediction=last_prediction,

        recent_activity=recent_activity[-5:],

        csv_notifications=csv_notifications,

        model_accuracy=model_accuracy,

        precision=precision,

        recall=recall,

        f1=f1,

        current_date=pd.Timestamp.now().strftime(
            "%d-%m-%Y %H:%M"
        )

    )

# =====================================
# Customers
# =====================================

@app.route("/customers")
def customers():

    if "user" not in session:
        return redirect(url_for("login"))

    df = load_data()

    search = request.args.get("search", "").lower()

    customer_list = []

    for i, row in df.iterrows():

        score = row["credit_score"]

        if score < 600:
            risk = "High Risk"
        elif score < 750:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        customer = {

            "id": i + 1,
            "name": f"Customer {i+1}",
            "credit": score,
            "risk": risk

        }

        if search == "" or search in customer["name"].lower():
            customer_list.append(customer)

    return render_template(
        "customers.html",
        customers=customer_list
    )

# =====================================
# Prediction
# =====================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    global recent_activity

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        try:

            print("\n========== FORM DATA ==========")
            print(request.form)
            print("===============================\n")

            # ==========================
            # Get Form Data
            # ==========================

            monthly_income = float(request.form.get("monthly_income", 0))
            employment_status = request.form.get("employment_status", "")
            debt_to_income_ratio = float(request.form.get("debt_to_income_ratio", 0))
            credit_score = int(request.form.get("credit_score", 0))
            interest_rate = float(request.form.get("interest_rate", 0))
            current_balance = float(request.form.get("current_balance", 0))
            loan_amount = float(request.form.get("loan_amount", 0))
            grade_subgrade = request.form.get("grade_subgrade", "")
            total_credit_limit = float(request.form.get("total_credit_limit", 0))
            num_of_open_accounts = int(request.form.get("num_of_open_accounts", 0))
            num_of_delinquencies = int(request.form.get("num_of_delinquencies", 0))
            installment = float(request.form.get("installment", 0))

            # ==========================
            # Create DataFrame
            # ==========================

            input_df = pd.DataFrame([{

                "employment_status": employment_status,
                "debt_to_income_ratio": debt_to_income_ratio,
                "credit_score": credit_score,
                "grade_subgrade": grade_subgrade,
                "interest_rate": interest_rate,
                "current_balance": current_balance,
                "installment": installment,
                "total_credit_limit": total_credit_limit,
                "loan_amount": loan_amount,
                "monthly_income": monthly_income,
                "num_of_open_accounts": num_of_open_accounts,
                "num_of_delinquencies": num_of_delinquencies

            }])

            # ==========================
            # Encode
            # ==========================

            input_df["employment_status"] = encoders["employment_status"].transform(
                input_df["employment_status"]
            )

            input_df["grade_subgrade"] = encoders["grade_subgrade"].transform(
                input_df["grade_subgrade"]
            )

            # ==========================
            # Arrange Features
            # ==========================

            print("Features :", features)
            print("Input Columns :", input_df.columns.tolist())

            input_df = input_df[features]

            # ==========================
            # Scaling
            # ==========================

            input_scaled = scaler.transform(input_df)

            # ==========================
            # Prediction
            # ==========================

            prediction = model.predict(input_scaled)[0]

            proba = model.predict_proba(input_scaled)[0]

            classes = list(model.classes_)

            default_probability = proba[classes.index(0)] * 100
            paid_probability = proba[classes.index(1)] * 100

            # ==========================
            # Prediction Text
            # ==========================

            if prediction == 0:
                prediction_text = "Default Risk"
            else:
                prediction_text = "Loan Will Be Paid"

            # ==========================
            # Risk
            # ==========================

            if default_probability >= 70:
                risk = "High Risk"

            elif default_probability >= 40:
                risk = "Medium Risk"

            else:
                risk = "Low Risk"

            # ==========================
            # Save Prediction History
            # ==========================

            last_prediction = pd.Timestamp.now().strftime("%d-%m-%Y %H:%M")

            recent_activity.append(
                f"{last_prediction} - {prediction_text}"
            )

            history = pd.read_csv(PREDICTION_FILE)

            new_row = pd.DataFrame([{

                "date": last_prediction,

                "prediction": prediction_text,

                "risk": risk,

                "default_probability": f"{default_probability:.2f}%",

                "paid_probability": f"{paid_probability:.2f}%"

            }])

            history = pd.concat(
                [history, new_row],
                ignore_index=True
            )

            history.to_csv(
                PREDICTION_FILE,
                index=False
            )

            # ==========================
            # Return
            # ==========================

            return render_template(

                "predict.html",

                prediction=prediction_text,

                probability=f"{default_probability:.2f}%",

                paid_probability=f"{paid_probability:.2f}%",

                risk=risk,

                model_accuracy=model_accuracy

            )

        except Exception as e:

            import traceback

            print("\n========== ERROR ==========")
            traceback.print_exc()
            print("===========================\n")

            return render_template(

                "predict.html",

                prediction="Prediction Failed",

                probability="0%",

                paid_probability="0%",

                risk="",

                model_accuracy=model_accuracy,

                error=str(e)

            )

    return render_template(

        "predict.html",

        prediction=None,

        model_accuracy=model_accuracy

    )

# =====================================
# Export CSV
# =====================================

@app.route("/export_csv")
def export_csv():

    df = load_data()

    path = "static/reports/customers.csv"

    df.to_csv(

        path,

        index=False

    )

    return send_file(

        path,

        as_attachment=True

    )

# =====================================
# PDF Report
# =====================================

@app.route("/report")
def report():

    df = load_data()

    total_customers = len(df)

    high_risk = len(
        df[df["loan_paid_back"] == 0]
    )

    average_risk = round(
        (high_risk / total_customers) * 100,
        2
    )

    prediction_df = pd.read_csv(
        PREDICTION_FILE
    )

    total_predictions = len(
        prediction_df
    )

    if total_predictions > 0:

        last_prediction = prediction_df.iloc[-1]["date"]

        low_risk = len(
            prediction_df[
                prediction_df["risk"] == "Low Risk"
            ]
        )

        medium_risk = len(
            prediction_df[
                prediction_df["risk"] == "Medium Risk"
            ]
        )

        high_prediction = len(
            prediction_df[
                prediction_df["risk"] == "High Risk"
            ]
        )

    else:

        last_prediction = "No Prediction Yet"

        low_risk = 0
        medium_risk = 0
        high_prediction = 0

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        18
    )

    pdf.cell(
        190,
        12,
        "LoanAssist EMI Prediction Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    pdf.set_font(
        "Arial",
        size=12
    )

    pdf.cell(
        190,
        10,
        f"Total Customers : {total_customers}",
        ln=True
    )

    pdf.cell(
        190,
        10,
        f"High Risk Customers : {high_risk}",
        ln=True
    )

    pdf.cell(
        190,
        10,
        f"Average Risk : {average_risk}%",
        ln=True
    )

    pdf.cell(
        190,
        10,
        f"Predictions Generated : {total_predictions}",
        ln=True
    )

    pdf.cell(
        190,
        10,
        f"Last Prediction : {last_prediction}",
        ln=True
    )

    pdf.ln(10)

    pdf.set_font(
        "Arial",
        "B",
        14
    )

    pdf.cell(
        190,
        10,
        "Prediction Summary",
        ln=True
    )

    pdf.set_font(
        "Arial",
        size=12
    )

    pdf.cell(
        190,
        10,
        f"Low Risk : {low_risk}",
        ln=True
    )

    pdf.cell(
        190,
        10,
        f"Medium Risk : {medium_risk}",
        ln=True
    )

    pdf.cell(
        190,
        10,
        f"High Risk : {high_prediction}",
        ln=True
    )

    path = "static/reports/report.pdf"

    pdf.output(path)

    return send_file(
        path,
        as_attachment=True
    )
# =====================================
# Analytics
# =====================================

@app.route("/analytics")
def analytics():

    if "user" not in session:
        return redirect(url_for("login"))

    df = load_data()

    # ===============================
    # KPI Cards
    # ===============================


    total_customers = len(df)

    paid_count = len(df[df["loan_paid_back"] == 1])

    default_count = len(df[df["loan_paid_back"] == 0])

    default_rate = round(
        (default_count / total_customers) * 100,
        2
    )

    avg_credit = round(df["credit_score"].mean(), 2)

    avg_income = round(df["annual_income"].mean(), 2)

    avg_loan = round(df["loan_amount"].mean(), 2)

    avg_dti = round(df["debt_to_income_ratio"].mean(), 2)

    # ===============================
    # Feature Statistics
    # ===============================

    avg_age = round(df["age"].mean(), 2)
    min_age = df["age"].min()
    max_age = df["age"].max()

    avg_monthly_income = round(df["monthly_income"].mean(), 2)
    min_monthly_income = int(df["monthly_income"].min())
    max_monthly_income = int(df["monthly_income"].max())

    min_income = int(df["annual_income"].min())
    max_income = int(df["annual_income"].max())

    min_credit = df["credit_score"].min()
    max_credit = df["credit_score"].max()

    min_loan = int(df["loan_amount"].min())
    max_loan = int(df["loan_amount"].max())

    avg_interest = round(df["interest_rate"].mean(), 2)
    min_interest = round(df["interest_rate"].min(), 2)
    max_interest = round(df["interest_rate"].max(), 2)

    min_dti = round(df["debt_to_income_ratio"].min(), 2)
    max_dti = round(df["debt_to_income_ratio"].max(), 2)

    avg_accounts = round(df["num_of_open_accounts"].mean(), 2)
    min_accounts = df["num_of_open_accounts"].min()
    max_accounts = df["num_of_open_accounts"].max()

    return render_template(

        "analytics.html",

        total_customers=total_customers,

        paid_count=paid_count,

        default_count=default_count,

        default_rate=default_rate,

        avg_credit=avg_credit,

        avg_income=avg_income,

        avg_loan=avg_loan,

        avg_dti=avg_dti,

        avg_age=avg_age,
        min_age=min_age,
        max_age=max_age,

        avg_monthly_income=avg_monthly_income,
        min_monthly_income=min_monthly_income,
        max_monthly_income=max_monthly_income,

        min_income=min_income,
        max_income=max_income,

        min_credit=min_credit,
        max_credit=max_credit,

        min_loan=min_loan,
        max_loan=max_loan,

        avg_interest=avg_interest,
        min_interest=min_interest,
        max_interest=max_interest,

        min_dti=min_dti,
        max_dti=max_dti,

        avg_accounts=avg_accounts,
        min_accounts=min_accounts,
        max_accounts=max_accounts,

        model_accuracy=model_accuracy,
        precision=precision,
        recall=recall,
        f1=f1
        

    )
# =====================================
# Prediction Analytics
# =====================================

@app.route("/prediction_analytics")
def prediction_analytics():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction_df = pd.read_csv(PREDICTION_FILE)

    feature_count = len(features)

    total_predictions = len(prediction_df)

    if total_predictions > 0:

        low_risk = len(
            prediction_df[
                prediction_df["risk"] == "Low Risk"
            ]
        )

        medium_risk = len(
            prediction_df[
                prediction_df["risk"] == "Medium Risk"
            ]
        )

        high_prediction = len(
            prediction_df[
                prediction_df["risk"] == "High Risk"
            ]
        )

        safe_count = low_risk + medium_risk

        prediction_history = prediction_df["date"].tolist()

    else:

        low_risk = 0
        medium_risk = 0
        high_prediction = 0
        safe_count = 0
        prediction_history = []

    return render_template(

        "prediction_analytics.html",

        total_predictions=total_predictions,

        low_risk=low_risk,

        medium_risk=medium_risk,

        high_prediction=high_prediction,

        safe_count=safe_count,

        prediction_history=prediction_history,

        feature_count=feature_count,

        model_accuracy=model_accuracy,

        precision=precision,

        recall=recall,

        f1=f1

    )


# =====================================
# Logout
# =====================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =====================================
# Run Flask App
# =====================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

