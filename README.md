# 🏥 Predicting Early Readmission in Diabetic Patients

This project aims to predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge using clinical and behavioral features. Accurate identification of high-risk patients helps hospitals reduce readmission costs and improve patient outcomes.

---

## 📁 Project Structure

    ├── diabetic_data.csv # Dataset from UCI ML Repository

    ├── final_clean_code_with_MLP.ipynb # Jupyter notebook       (data prep, modeling, evaluation)

    ├── xgb_model.pkl # Trained XGBoost model (for deployment)

    ├── main.py # FastAPI app to serve the model

    ├── requirements.txt # Python dependencies

    ├── README.md # Project documentation

---

## 📄 File Descriptions

- **`diabetic_data.csv`** – Dataset from the UCI Machine Learning Repository.
- **`final_clean_code_with_MLP.ipynb`** – Jupyter notebook containing data preparation, feature engineering, modeling, and evaluation.
- **`xgb_model.pkl`** – Serialized XGBoost model trained to predict 30-day readmissions.
- **`main.py`** – FastAPI app that loads the model and exposes a prediction API.
- **`requirements.txt`** – Python packages needed to run the notebook and the API.
- **`README.md`** – Project documentation and usage instructions.

---

## 📊 Dataset Overview

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Duration**: 10 years (1999–2008) from 130 US hospitals
- **Features**: 50+ features (demographics, diagnoses, medications, labs, etc.)

**Filtering Criteria**:

- Inpatient encounters
- Confirmed diabetic diagnosis
- Hospital stay between 1–14 days
- Lab tests and medications administered

---

## 🧠 Modeling Summary

- **Challenge**: Class imbalance – `<30 days` readmissions are underrepresented
- **Feature Engineering**:

  - `polypharmacy_score`: Aggregated 23 medication features
  - `risk_cluster`: Patient risk categories using KMeans clustering
  - Scaled numerical and one-hot encoded categorical variables

- **Models Compared**:

  - Logistic Regression
  - Random Forest
  - XGBoost (with threshold tuning)
  - MLP Classifier

- **Best Models**:
  - **XGBoost (threshold-tuned)** and **MLP** achieved higher recall for early readmissions, making them suitable for clinical risk alerts.

---

## 🚀 Model Deployment with FastAPI

We use **FastAPI** to deploy the trained XGBoost model for prediction.

### 📂 Deployment Files

- `xgb_model.pkl` — Trained model file
- `main.py` — FastAPI app exposing a `/predict` endpoint

### ⚙️ Setup Instructions

1. **Install required packages**:

```bash
pip install fastapi uvicorn scikit-learn xgboost numpy joblib
```

2. **Run the API locally:**

```bash
uvicorn main:app --reload
```

3. **Access the Swagger UI:**

```bash
http://127.0.0.1:8000/docs
```

This interactive UI allows you to test predictions easily.

## 📤 Sample Input (Swagger)

```json
{
  "feature_vector": [0.85, 1.2, 0.0, 3.4, 2.1, ...]  // Replace with actual preprocessed feature values
}
```

## 📥 Sample Output

```json
{
  "prediction": 1,
  "label": "Readmission within 30 days (high risk)",
  "probability": {
    "No readmission (<30 days)": 0.12,
    "Readmission (<30 days)": 0.88
  }
}
```

📝 **NOTE:** You can refer to the test_features.csv file and copy a row (as a list of values) to use as input on the Swagger UI. Make sure the values are in the same order as used during model training.

## 🧪 How to Use Swagger UI for Prediction

1. Open your browser and go to:
   http://127.0.0.1:8000/docs

2. Find the POST /predict endpoint.

3. Click the dropdown arrow to expand it, then click "Try it out".

4. In the request body field, you'll see:

```json
{
  "feature_vector": [0]
}
```

5.  Replace the 0 with a full list of feature values.

    📝 NOTE: Open the test_features.csv file and copy a row as a comma-separated list to paste here.

    **Example:**

```json
{
  "feature_vector": [0.85, 1.67, 2.0, 0.0, 3.41, ...]
}
```

6. Click "Execute".

7. Check the Server response section for:

   - **prediction:** 0 or 1
   - **label:** Human-readable outcome (e.g., "Readmission within 30 days (high risk)")
   - **probability:** Confidence levels for both classes

🔍 **Interpretation:**

- "`1` = Readmission within 30 days (high risk)"

- "`0` = Readmission after 30 days (lower risk)"

## 🗂️ Project Kanban Board

We use a GitHub Project board to manage tasks and progress across different stages of the project lifecycle (To Do, In Progress, Done).

🔗 [View Kanban Board](https://github.com/users/clappy203/projects/3)

## 📌 Future Work

- Add SHAP explainability to deployed API

- Automate retraining using new EHR data

- Integrate model into clinical decision support tools

- Explore risk stratification across patient subgroups
