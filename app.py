from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    result = None
    probability = None

    if request.method == "POST":

        gender = request.form["gender"]
        contract = request.form["contract"]
        internet = request.form["internet"]
        tenure = int(request.form["tenure"])
        monthly = float(request.form["monthly_charges"])

        # Encoding values
        gender = 1 if gender == "Male" else 0

        contract_dict = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }

        internet_dict = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }

        contract = contract_dict[contract]
        internet = internet_dict[internet]

        features = np.array([
            gender,
            contract,
            tenure,
            monthly,
            internet
        ]).reshape(1, -1)

        print(features)

        pred = model.predict(features)
        prob = model.predict_proba(features)

        print("Prediction:", pred)
        print("Probability:", prob)

        probability = round(prob[0][1] * 100, 2)

        if pred[0] == 1:
            result = "⚠ Customer is likely to Churn"
        else:
            result = "✅ Customer is likely to Stay"

    return render_template(
        "prediction.html",
        prediction=result,
        probability=probability
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)