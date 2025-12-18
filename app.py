from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open("data/model.pkl", "rb") as f:

    model = pickle.load(f)

@app.route("/")
def home():
    return "Boston Housing Price Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"predicted_price": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
