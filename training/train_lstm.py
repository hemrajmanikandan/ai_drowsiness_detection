import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

SEQ_LEN = 15

print("🚀 Training model...")

data = pd.read_csv("data/processed_data.csv")

features = ['ear','mar','perclos','pose','yawn_detected']

X = data[features].values
y = data['label'].values

# create sequences
X_seq, y_seq = [], []
for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i+SEQ_LEN])
    y_seq.append(y[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=True
)

# model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 5)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15)

os.makedirs("models", exist_ok=True)
model.save("models/drowsiness_model.h5")

print("✅ Model trained and saved")