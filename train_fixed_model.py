import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("="*60)
print("CARDIO ML MODEL TRAINING - FIXED VERSION")
print("="*60)

# 1. Load data
print("\n[1/6] Loading dataset...")
df = pd.read_csv('cardio_train.csv', delimiter=';')
print(f"   Loaded {len(df)} records")

# 2. Data cleaning
print("\n[2/6] Cleaning data...")
# Remove outliers
df = df[
    (df['ap_hi'] >= 60) & (df['ap_hi'] <= 250) &
    (df['ap_lo'] >= 40) & (df['ap_lo'] <= 200) &
    (df['height'] >= 130) & (df['height'] <= 220) &
    (df['weight'] >= 30) & (df['weight'] <= 200)
]
print(f"   After cleaning: {len(df)} records")

# 3. Feature engineering
print("\n[3/6] Engineering features...")
df['age_years'] = df['age'] / 365  # Convert from days to years
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Select features (14 total)
feature_cols = [
    'age', 'gender', 'height', 'weight', 
    'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
    'smoke', 'alco', 'active',
    'bmi', 'pulse_pressure', 'age_years'
]

X = df[feature_cols]
y = df['cardio']

print(f"   Features: {X.shape[1]}")
print(f"   Feature names: {list(X.columns)}")

# 4. Split data
print("\n[4/6] Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
print(f"   Class balance: {y_train.value_counts().to_dict()}")

# 5. Train model with Pipeline (includes StandardScaler!)
print("\n[5/6] Training Logistic Regression with StandardScaler...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)
print("   ✓ Training complete")

# 6. Evaluate
print("\n[6/6] Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.4f}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Test on sample cases
print("\n" + "="*60)
print("TESTING ON SAMPLE CASES")
print("="*60)

# Test case 1: Young healthy person (should be LOW RISK)
test1 = pd.DataFrame([{
    'age': 25 * 365,
    'gender': 1,
    'height': 175,
    'weight': 68,
    'ap_hi': 110,
    'ap_lo': 70,
    'cholesterol': 1,
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 1,
    'bmi': 22.2,
    'pulse_pressure': 40,
    'age_years': 25
}])

pred1 = model.predict(test1)[0]
prob1 = model.predict_proba(test1)[0]
print(f"\nTest 1 - Healthy 25yr old:")
print(f"  Prediction: {'High Risk' if pred1 == 1 else 'Low Risk'}")
print(f"  Confidence: Low={prob1[0]:.1%}, High={prob1[1]:.1%}")

# Test case 2: Elderly high risk person (should be HIGH RISK)
test2 = pd.DataFrame([{
    'age': 60 * 365,
    'gender': 1,
    'height': 170,
    'weight': 95,
    'ap_hi': 180,
    'ap_lo': 110,
    'cholesterol': 3,
    'gluc': 3,
    'smoke': 1,
    'alco': 1,
    'active': 0,
    'bmi': 32.9,
    'pulse_pressure': 70,
    'age_years': 60
}])

pred2 = model.predict(test2)[0]
prob2 = model.predict_proba(test2)[0]
print(f"\nTest 2 - Risky 60yr old:")
print(f"  Prediction: {'High Risk' if pred2 == 1 else 'Low Risk'}")
print(f"  Confidence: Low={prob2[0]:.1%}, High={prob2[1]:.1%}")

# 8. Save model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
pickle.dump(model, open('model.pkl', 'wb'))
print("✓ Model saved as 'model.pkl'")
print(f"✓ Model includes StandardScaler + LogisticRegression")
print(f"✓ Expects {len(feature_cols)} features in this order:")
for i, col in enumerate(feature_cols, 1):
    print(f"   {i}. {col}")

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Copy 'model.pkl' to your project folder")
print("2. Restart Flask: python app.py")
print("3. Test the frontend!")