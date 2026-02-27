# 🫀 Cardiovascular Disease ML Prediction App

Full-stack Machine Learning web application for predicting cardiovascular disease risk using patient health data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![React](https://img.shields.io/badge/React-19-61DAFB)
![ML](https://img.shields.io/badge/ML-Logistic%20Regression-green)

---

## 🎯 Features

- ✅ Real-time cardiovascular risk prediction
- ✅ Dynamic confidence percentage (ML probability)
- ✅ 8 health parameter inputs
- ✅ Animated dark futuristic UI
- ✅ Glassmorphism design with Framer Motion
- ✅ RESTful Flask API backend
- ✅ Trained on 70,000+ patient records

---

## 🏗️ Tech Stack

### Backend
- **Flask** - Python web framework
- **Scikit-Learn** - Machine Learning (Logistic Regression)
- **Pandas & NumPy** - Data processing
- **Flask-CORS** - Cross-origin requests

### Frontend
- **React 19** - UI framework
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Axios** - HTTP client
- **Vite** - Build tool

---

## 📁 Project Structure
```
cardiovascular-disease-ml-app/
├── app.py                      # Flask API server
├── train_fixed_model.py        # Model training script
├── model.pkl                   # Trained ML model (gitignored)
├── cardio_train.csv            # Dataset (gitignored)
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
└── frontend/                   # React app
    ├── src/
    │   ├── components/         # Reusable UI components
    │   ├── pages/              # Main pages
    │   ├── hooks/              # Custom React hooks
    │   ├── api.js              # Axios API config
    │   └── App.jsx
    ├── package.json
    ├── tailwind.config.js
    └── vite.config.js
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm

### Backend Setup
```bash
# Clone repository
git clone https://github.com/YourUsername/cardiovascular-disease-ml-app.git
cd cardiovascular-disease-ml-app

# Install Python dependencies
pip install -r requirements.txt

# Train model (first time only)
python train_fixed_model.py

# Run Flask server
python app.py
```

Backend runs on: `http://127.0.0.1:5000`

---

### Frontend Setup
```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs on: `http://localhost:3000`

---

## 🧪 Usage

1. **Start Backend**: `python app.py` in project root
2. **Start Frontend**: `npm run dev` in `frontend/` folder
3. Open browser: `http://localhost:3000`
4. Enter patient data (8 parameters)
5. Click "Analyze Cardiovascular Risk"
6. View prediction with confidence percentage

---

## 📊 Input Parameters

| Parameter | Range | Description |
|---|---|---|
| Age | 1-120 years | Patient age |
| Gender | Male/Female | Biological sex |
| Height | 50-250 cm | Height in centimeters |
| Weight | 10-300 kg | Weight in kilograms |
| Systolic BP | 60-250 mmHg | Upper blood pressure |
| Diastolic BP | 40-200 mmHg | Lower blood pressure |
| Smoking | Yes/No | Smoking status |
| Alcohol | Yes/No | Alcohol consumption |

---

## 🤖 ML Model Details

| Metric | Value |
|---|---|
| Algorithm | Logistic Regression with StandardScaler |
| Training Data | 68,681 records (after cleaning) |
| Accuracy | ~72% |
| Features | 14 engineered features |
| Output | Binary (High Risk / Low Risk) + Confidence % |

### Feature Engineering
- BMI calculation
- Pulse pressure (Systolic - Diastolic)
- Age conversion (years → days)
- Cholesterol/Glucose inference from BP

---

## 📸 Screenshots

### Form View
![Form](https://via.placeholder.com/800x400?text=Add+Screenshot)

### High Risk Result
![High Risk](https://via.placeholder.com/800x400?text=Add+Screenshot)

### Low Risk Result
![Low Risk](https://via.placeholder.com/800x400?text=Add+Screenshot)

---

## 🎓 Academic Project

**University**: [Your University Name]  
**Course**: Machine Learning Lab (Semester 6)  
**Student**: Monil Kansagra  
**Year**: 2025

---

## 📝 API Documentation

### POST `/predict`

**Request Body:**
```json
{
  "age": 60,
  "gender": 1,
  "height": 170,
  "weight": 95,
  "ap_hi": 180,
  "ap_lo": 110,
  "smoke": 1,
  "alco": 1
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "High Risk",
  "confidence": 98
}
```

---

## 🔮 Future Enhancements

- [ ] Add cholesterol/glucose inputs
- [ ] Deploy on Heroku/Vercel
- [ ] Add user authentication
- [ ] Store prediction history
- [ ] Export reports as PDF
- [ ] Support multiple ML models

---

## 📄 License

This project is for educational purposes.

---

## 🤝 Contributing

This is an academic project. Feedback welcome via issues!

---

## 📧 Contact

**Monil Kansagra**  
GitHub: [@Monilkansagra](https://github.com/Monilkansagra)

---

⭐ Star this repo if you found it helpful!
```

---

### Step 3: Create `requirements.txt`

Create `requirements.txt` in project root:
```
flask==3.0.3
flask-cors==6.0.2
numpy==1.26.4
scikit-learn==1.5.0
pandas==2.2.2
