import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

# Test Case 1: Perfect healthy young person
test1 = np.array([[
    25 * 365,  # age_days (25 years)
    1,         # gender (male)
    175,       # height
    68,        # weight
    110,       # ap_hi
    70,        # ap_lo
    1,         # cholesterol (normal)
    1,         # gluc (normal)
    0,         # smoke (no)
    0,         # alco (no)
    1,         # active (yes)
    22.2,      # bmi (healthy)
    40,        # pulse_pressure
    25         # age_years
]])

# Test Case 2: High risk elderly
test2 = np.array([[
    60 * 365,  # age_days (60 years)
    1,         # gender (male)
    170,       # height
    95,        # weight
    180,       # ap_hi
    110,       # ap_lo
    3,         # cholesterol (high)
    3,         # gluc (high)
    1,         # smoke (yes)
    1,         # alco (yes)
    0,         # active (no)
    32.9,      # bmi (obese)
    70,        # pulse_pressure
    60         # age_years
]])

print("="*50)
print("TEST 1 - Healthy 25yr old:")
pred1 = model.predict(test1)[0]
prob1 = model.predict_proba(test1)[0]
print(f"  Prediction: {pred1} (0=Low Risk, 1=High Risk)")
print(f"  Probability: Low={prob1[0]:.2%}, High={prob1[1]:.2%}")

print("\n" + "="*50)
print("TEST 2 - Risky 60yr old:")
pred2 = model.predict(test2)[0]
prob2 = model.predict_proba(test2)[0]
print(f"  Prediction: {pred2} (0=Low Risk, 1=High Risk)")
print(f"  Probability: Low={prob2[0]:.2%}, High={prob2[1]:.2%}")

print("\n" + "="*50)
print(f"Model expects {model.n_features_in_} features")
print("="*50)