# src/ai/training.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(normal_file="data/normal_events/data.npy", anomaly_file="data/anomaly_events/data.npy"):
    X_normal = np.load(normal_file)
    X_anomaly = np.load(anomaly_file)

    y_normal = np.zeros(len(X_normal))
    y_anomaly = np.ones(len(X_anomaly))

    X = np.vstack((X_normal, X_anomaly))
    y = np.hstack((y_normal, y_anomaly))

    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_shape):
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("🔁 Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("📐 Building model...")
    model = build_model(X_train.shape[1])

    print("🚀 Training model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    print("✅ Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    print("💾 Saving model...")
    model.save("src/ai/anomaly_model.h5")
    joblib.dump({"mean": X_train.mean(axis=0), "std": X_train.std(axis=0)}, "src/ai/normalization.pkl")

if __name__ == "__main__":
    main()
