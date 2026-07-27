import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

print("📊 Preparing dataset...")

data = pd.read_csv("data/dataset.csv")

# clean
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# lowercase columns
data.columns = [c.lower() for c in data.columns]

# ensure columns exist
required = ['ear','mar','perclos','pose','yawn_detected','label']
for col in required:
    if col not in data.columns:
        data[col] = 0

X = data[['ear','mar','perclos','pose','yawn_detected']]
y = data['label']

# normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.save")

# save processed data
df = pd.DataFrame(X_scaled, columns=X.columns)
df['label'] = y
df.to_csv("data/processed_data.csv", index=False)

print("✅ processed_data.csv created")