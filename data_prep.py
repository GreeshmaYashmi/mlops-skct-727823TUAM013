# GREESHMA_YASHMI 727823TUAM013

from datetime import datetime
import pandas as pd

print("Roll No: 727823TUAM013")
print("Timestamp:", datetime.now())

df = pd.read_csv("credit_card_default.csv")
df.to_csv("processed.csv", index=False)
