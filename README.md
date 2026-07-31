# Heart Disease Prediction – End-to-End ML Deployment

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.3-black?logo=flask)
![Scikit‑Learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikit-learn)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render)
![License](https://img.shields.io/badge/License-MIT-green)

> **AI‑ML Assignment 10** · VIT Bhopal University  
> **Student:** [Your Name] · **Reg. No.:** [Your Registration Number]  
> **Course:** Integrated M.Tech – Artificial Intelligence  

---

##  Live Deployment

**Render URL:** `https://heartdiseasedeployment-jcoy.onrender.com`

> Replace with your actual Render deployment URL after deployment.

---

##  Objective

Predict whether a patient has heart disease using clinical parameters via a Logistic Regression model deployed as a **Flask REST API** on **Render** cloud platform.

---

##  Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py              ← Flask REST API (main entry point)
├── train_model.py      ← Data preprocessing + model training
├── model.pkl           ← Serialised trained model (joblib)
├── heart.csv           ← Heart disease dataset (from Kaggle)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── templates/
│   └── index.html      ← Optional web UI for manual testing
└── static/             ← Static assets (CSS, JS, images)
```

---

##  Dataset

| Detail        | Value                                              |
|---------------|----------------------------------------------------|
| Source        | [Kaggle – Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| Rows          | 1,025                                              |
| Features      | 13 clinical parameters                             |
| Target        | `target` (0 = No Disease, 1 = Disease Present)     |

### Feature Descriptions

| Feature    | Description                                         |
|------------|-----------------------------------------------------|
| age        | Age in years                                        |
| sex        | Sex (1 = Male, 0 = Female)                          |
| cp         | Chest pain type (0–3)                               |
| trestbps   | Resting blood pressure (mmHg)                       |
| chol       | Serum cholesterol (mg/dL)                           |
| fbs        | Fasting blood sugar > 120 mg/dL (1 = True)         |
| restecg    | Resting ECG results (0–2)                           |
| thalach    | Maximum heart rate achieved                         |
| exang      | Exercise-induced angina (1 = Yes)                   |
| oldpeak    | ST depression induced by exercise relative to rest  |
| slope      | Slope of peak exercise ST segment (0–2)             |
| ca         | Number of major vessels (0–3) by fluoroscopy        |
| thal       | Thalassemia type (0–3)                              |

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HeartDiseaseDeployment.git
cd HeartDiseaseDeployment
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (generates model.pkl)

```bash
python train_model.py
```

### 5. Run the Flask app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 🔌 API Reference

### `GET /`

Returns the web UI (HTML page) or a welcome JSON message.

---

### `POST /predict`

Accepts patient clinical data and returns the prediction.

**Request Body (JSON):**

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response (JSON):**

```json
{
  "prediction": "Heart Disease Detected"
}
```

**cURL example:**

```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
    "fbs":1,"restecg":0,"thalach":150,"exang":0,
    "oldpeak":2.3,"slope":0,"ca":0,"thal":1
  }'
```

---

##  Model Performance

| Metric      | Value  |
|-------------|--------|
| Algorithm   | Logistic Regression |
| Accuracy    | ~85%   |
| Train Split | 80%    |
| Test Split  | 20%    |
| random_state| 42     |

> Exact metrics are printed when you run `train_model.py`.

---

##  Render Deployment Steps

> **Note:** A `render.yaml` is included in this repository — Render will auto-detect it and configure the service automatically.

1. Push this repository to GitHub.
2. Log in at [render.com](https://render.com) and click **New → Web Service**.
3. Connect your GitHub account and select this repository.
4. Configure the service (or let `render.yaml` auto-fill these):
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** Python 3
   - **Python Version:** 3.11.0
5. Click **Create Web Service** and wait ~3–5 minutes for deployment.
6. Access the live URL: `https://your-app-name.onrender.com`

> ⚠️ Do **not** use `python app.py` as the start command on Render — use `gunicorn app:app` instead.

---

## 📝 License

This project is for academic purposes – VIT Bhopal University, AI‑ML Assignment 10.
