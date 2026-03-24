import mlflow
import mlflow.sklearn
import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 🔴 DETAILS
ROLL_NO = "727823TUAM013"
NAME = "GREESHMA_YASHMI"
DATASET = "CreditDefault"

# 🔥 FIX: SAME TRACKING PATH
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(f"SKCT_{ROLL_NO}_{DATASET}")

# =========================
# 📥 LOAD DATASET
# =========================
try:
    df = pd.read_csv("credit_data.csv")
except:
    df = pd.read_csv("credit_data.csv", header=1)

df.columns = df.columns.str.strip()

# Detect target column
target_col = [col for col in df.columns if "default" in col.lower()][0]
print("✅ Target Column:", target_col)

# =========================
# 📊 EDA (3 PLOTS)
# =========================
os.makedirs("artifacts", exist_ok=True)

plt.figure()
sns.countplot(x=df[target_col])
plt.title("Target Distribution")
plt.savefig("artifacts/target_distribution.png")
plt.close()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("artifacts/correlation.png")
plt.close()

plt.figure()
df.iloc[:,1].hist()
plt.title("Feature Distribution")
plt.savefig("artifacts/feature_dist.png")
plt.close()

# =========================
# 🧠 DATA PREP
# =========================
X = df.drop(target_col, axis=1)
y = df[target_col]

X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)

y = pd.to_numeric(y, errors='coerce')

# 🔥 Scaling (fix convergence issue)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 📦 MODEL SIZE
# =========================
def get_model_size(model):
    import joblib
    joblib.dump(model, "temp.pkl")
    size = os.path.getsize("temp.pkl") / (1024*1024)
    os.remove("temp.pkl")
    return size

# =========================
# 🔁 EXPERIMENT LOOP (12 RUNS)
# =========================
for i in range(12):

    model_type = "lr" if i % 2 == 0 else "rf"
    random_seed = np.random.randint(1, 1000)

    if model_type == "lr":
        model = LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            random_state=random_seed
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100 + i*20,
            class_weight='balanced',
            random_state=random_seed
        )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    # 🔥 FIXED METRICS
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    with mlflow.start_run():

        # PARAMETERS
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_seed", random_seed)

        # METRICS
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("training_time_seconds", end - start)
        mlflow.log_metric("model_size_mb", get_model_size(model))
        mlflow.log_metric("n_features", X.shape[1])

        # TAGS
        mlflow.set_tags({
            "student_name": NAME,
            "roll_number": ROLL_NO,
            "dataset": DATASET
        })

        # MODEL
        mlflow.sklearn.log_model(model, "model")

        # ARTIFACTS
        mlflow.log_artifact("artifacts/target_distribution.png")
        mlflow.log_artifact("artifacts/correlation.png")
        mlflow.log_artifact("artifacts/feature_dist.png")

print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
