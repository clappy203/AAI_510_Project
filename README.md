# 🏥 Predicting Early Readmission in Diabetic Patients

This project aims to predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge using clinical and behavioral features. Accurate identification of high-risk patients helps hospitals reduce readmission costs and improve patient outcomes.

---

## 📁 Project Structure

    ├── diabetic_data.csv # Dataset from UCI ML Repository

    ├── final_clean_code+with_MLP.ipynb # Jupyter notebook
    containing data prep, modeling, and evaluation

    ├── requirements.txt # Python dependencies

    ├── README.md # Project documentation

---

### 📄 File Descriptions

- **`diabetic_data.csv`** – Dataset from the UCI Machine Learning Repository.
- **`final_clean_code_with_MLP.ipynb`** – Jupyter notebook containing data preparation, feature engineering, modeling, and evaluation.
- **`requirements.txt`** – Python dependencies required to run the notebook.
- **`README.md`** – Project documentation and usage instructions.

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Diabetes 130-US hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- **Filename**: `diabetic_data.csv`
- **License**: MIT License
- **Size**: 10 years of clinical data (1999–2008) from 130 US hospitals.
- **Features**: 50+ including demographics, diagnoses, lab results, hospital visits, and medications.

**Filtering Criteria:**

- Inpatient encounters only
- Diabetic diagnoses present
- Hospital stay: 1–14 days
- Lab tests and medications administered

---

## 🧠 Modeling Summary

- Addressed class imbalance (`<30` readmission) using clustering and threshold tuning.
- **Feature Engineering**:
  - Created `polypharmacy_score` by aggregating 23 medication features.
  - Applied one-hot encoding and feature scaling.
  - Used KMeans clustering to generate `risk_cluster` feature.
- **Models Used**:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost (with threshold tuning)
  - MLP Classifier (Neural Net)
- **Best Performers**: XGBoost (tuned) and MLP due to better recall on `<30` cases.

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash

git clone <https://github.com/clappy203/AAI_510_Project.git>
cd AAI_510_Project # if not in the repo yet

```

## 🗂️ Project Kanban Board

We use a GitHub Project board to manage tasks and progress across different stages of the project lifecycle (To Do, In Progress, Done).

🔗 [View Kanban Board](https://github.com/users/clappy203/projects/3)
