from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Cardio ML Backend Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age_years = float(data["age"])
        gender    = float(data["gender"])
        height    = float(data["height"])
        weight    = float(data["weight"])
        ap_hi     = float(data["ap_hi"])
        ap_lo     = float(data["ap_lo"])

        # Derived features
        age_days       = age_years * 365
        bmi            = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo

        # Default values for lifestyle features
        cholesterol = 1.0
        gluc        = 1.0
        smoke       = 0.0
        alco        = 0.0
        active      = 1.0

        # Final 14 features — same order as model was trained
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

        prediction = model.predict(features)

        result = "High Risk" if prediction[0] == 1 else "Low Risk"

        return jsonify({
            "success": True,
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True, port=5000)