# tests/ai_tests/test_inference.py

import numpy as np
import tensorflow as tf
import joblib
import unittest
import os

class TestAnomalyInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the trained model and normalization stats
        cls.model_path = "src/ai/anomaly_model.h5"
        cls.norm_path = "src/ai/normalization.pkl"

        assert os.path.exists(cls.model_path), "Model file not found!"
        assert os.path.exists(cls.norm_path), "Normalization file not found!"

        cls.model = tf.keras.models.load_model(cls.model_path)
        cls.norm = joblib.load(cls.norm_path)

    def preprocess(self, input_data):
        # Normalize input data
        mean, std = self.norm["mean"], self.norm["std"]
        return (input_data - mean) / std

    def test_normal_sample(self):
        # Simulated normal input: shape (1, 400)
        normal_sample = np.random.normal(loc=0.5, scale=0.1, size=(1, 400))
        input_data = self.preprocess(normal_sample)
        prediction = self.model.predict(input_data)[0][0]
        print(f"[Normal] Prediction score: {prediction:.4f}")
        self.assertLess(prediction, 0.5, "False positive detected in normal sample!")

    def test_anomaly_sample(self):
        # Simulated anomaly input: random spikes
        anomaly_sample = np.random.uniform(low=0.8, high=1.0, size=(1, 400))
        input_data = self.preprocess(anomaly_sample)
        prediction = self.model.predict(input_data)[0][0]
        print(f"[Anomaly] Prediction score: {prediction:.4f}")
        self.assertGreater(prediction, 0.5, "Failed to detect anomaly!")

if __name__ == "__main__":
    unittest.main()
