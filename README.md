# mlops-skct-727823TUAM013
Credit Card Default Prediction using MLflow and Azure ML pipeline with 12 experiments, model tracking, and evaluation.
# 💳 Credit Card Default Prediction using MLOps

## 👤 Student Details

* **Name:** GREESHMA YASHMI
* **Roll Number:** 727823TUAM013
* **Repository:** mlops-skct-727823TUAM013

---

## 📌 Project Overview

This project focuses on predicting whether a customer will default on their credit card payment in the next month using machine learning models.

The implementation follows an **MLOps pipeline** including:

* Exploratory Data Analysis (EDA)
* Model training with multiple experiments
* MLflow experiment tracking
* Azure ML pipeline deployment

---

## 📊 Dataset

The dataset used is the **Credit Card Default Dataset**, containing financial and demographic details of customers.

### 🎯 Target Variable

* `default_payment_next_month`

  * 1 → Default
  * 0 → No Default

### 📁 Features

* Credit limit (LIMIT_BAL)
* Payment history (PAY_0 to PAY_6)
* Bill amounts (BILL_AMT1 to BILL_AMT6)
* Payment amounts (PAY_AMT1 to PAY_AMT6)
* Demographic attributes (AGE, SEX, EDUCATION, MARRIAGE)

---

## 🔍 Exploratory Data Analysis

EDA was performed using Jupyter Notebook with the following insights:

* No missing values in dataset
* Target variable is imbalanced
* Strong correlation among billing features
* Presence of outliers in financial attributes

📓 Notebook: `notebooks/eda.ipynb`

---

## ⚙️ MLflow Experiment Tracking

### 📌 Experiment Name

```
SKCT_727823TUAM013_credit_card_default
```

### 🧪 Experiments Conducted

* Total Runs: **12**
* Algorithms used:

  * Logistic Regression
  * Random Forest
  * Support Vector Machine (SVM)
  * Gradient Boosting

### 📈 Metrics Logged

* F1-score
* Precision
* Recall
* ROC-AUC

### ⚡ Operational Metrics

* training_time_seconds
* model_size_mb
* n_features
* random_seed

### 🏆 Best Model

* **Random Forest Classifier**
* F1-score: ~0.468
* ROC-AUC: ~0.77

---

## ☁️ Azure ML Pipeline

A 3-stage pipeline was implemented:

1. **data_prep.py** → Data loading & preprocessing
2. **train_pipeline.py** → Model training
3. **evaluate.py** → Model evaluation

### 📄 Pipeline File

```
pipeline_727823TUAM013.yml
```

---

## 🛠️ Installation & Setup

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/<your-username>/mlops-skct-727823TUAM013.git
cd mlops-skct-727823TUAM013
```

### 🔹 Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Run EDA Notebook

```bash
jupyter notebook
```

Open:

```
notebooks/eda.ipynb
```

---

### 🔹 Run Training (MLflow)

```bash
mlflow ui
python training.py
```

Open MLflow UI:

```
http://127.0.0.1:5000
```

---

### 🔹 Run Azure Pipeline

```bash
az login
az ml job create --file pipeline_727823TUAM013.yml
```

---

## 📂 Project Structure

```
mlops-skct-727823TUAM013/

├── training.py
├── data_prep.py
├── train_pipeline.py
├── evaluate.py
├── pipeline_727823TUAM013.yml

├── notebooks/
│   └── eda.ipynb

├── requirements.txt
├── README.md
```

---

## ⚠️ Challenges Faced

```
Error:
AttributeError: 'SVC' object has no attribute 'predict_proba'

Fix:
Used decision_function() for SVM models instead of predict_proba().
```

---

## ✅ Conclusion

* Successfully conducted 12 ML experiments using MLflow
* Compared multiple models and selected the best
* Implemented Azure ML pipeline for automation
* Achieved reliable performance with Random Forest

---

