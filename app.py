from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("Model loaded successfully ✅")
except Exception as e:
    print("Error loading model:", e)
    model = None


# =========================
# HOME ROUTE (IMPORTANT)
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Cardiovascular Disease ML API is Live 🚀"
    })


# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        age_years = float(data["age"])
        gender    = float(data["gender"])
        height    = float(data["height"])
        weight    = float(data["weight"])
        ap_hi     = float(data["ap_hi"])
        ap_lo     = float(data["ap_lo"])
        smoke     = float(data.get("smoke", 0))
        alco      = float(data.get("alco", 0))

        # Derived features
        age_days       = age_years * 365
        bmi            = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo

        # Cholesterol and glucose logic
        if ap_hi <= 120 and ap_lo <= 80:
            cholesterol = 1.0
            gluc        = 1.0
            active      = 1.0
        else:
            cholesterol = 2.0
            gluc        = 2.0
            active      = 0.0

        features = np.array([[ 
            age_days,
            gender,
            height,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            bmi,
            pulse_pressure,
            age_years
        ]])

        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        result     = "High Risk" if prediction == 1 else "Low Risk"
        confidence = int(probability[1] * 100)

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# =========================
# REQUIRED FOR LOCAL ONLY
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)