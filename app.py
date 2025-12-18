from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

with open("data/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["CRIM"]),
        float(request.form["ZN"]),
        float(request.form["INDUS"]),
        float(request.form["CHAS"]),
        float(request.form["NOX"]),
        float(request.form["RM"]),
        float(request.form["AGE"]),
        float(request.form["DIS"]),
        float(request.form["RAD"]),
        float(request.form["TAX"]),
        float(request.form["PTRATIO"]),
        float(request.form["B"]),
        float(request.form["LSTAT"]),
    ]

    prediction = model.predict([features])[0]

    price = prediction * 1000  # Boston housing prices are in $1000s

    return render_template(
    "index.html",
    prediction_text=f"Estimated House Price: ${price:,.2f}"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
