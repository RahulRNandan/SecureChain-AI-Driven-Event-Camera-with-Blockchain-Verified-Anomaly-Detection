import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class AnomalyDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        # Normalize data
        X_train = self.scaler.fit_transform(X_train)
        
        # Define a simple neural network model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)

    def predict(self, X):
        X = self.scaler.transform(X)  # Normalize data
        return self.model.predict(X)
