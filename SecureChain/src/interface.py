import numpy as np
from model import AnomalyDetectionModel

def load_model():
    model = AnomalyDetectionModel()
    model.model = tf.keras.models.load_model('anomaly_detection_model.h5')
    return model

def detect_anomaly(data):
    model = load_model()
    prediction = model.predict(np.array([data]))
    return prediction[0] > 0.5  # Assuming 1 is anomaly and 0 is normal

if __name__ == '__main__':
    data = np.load('data/test_event.npy')  # Example test data
    is_anomaly = detect_anomaly(data)
    print(f"Anomaly Detected: {is_anomaly}")
