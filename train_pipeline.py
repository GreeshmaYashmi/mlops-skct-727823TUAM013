# GREESHMA_YASHMI 727823TUAM013

from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Roll No: 727823TUAM013")
print("Timestamp:", datetime.now())

df = pd.read_csv("processed.csv")

X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
