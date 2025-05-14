import numpy as np
from model import AnomalyDetectionModel

def load_data():
    # Example: Load your dataset here
    normal_data = np.load('data/normal_events.npy')
    anomaly_data = np.load('data/anomaly_events.npy')
    
    X_train = np.vstack((normal_data, anomaly_data))
    y_train = np.array([0] * len(normal_data) + [1] * len(anomaly_data))
    
    return X_train, y_train

def train_model():
    X_train, y_train = load_data()
    model = AnomalyDetectionModel()
    model.train(X_train, y_train)
    model.model.save('anomaly_detection_model.h5')

if __name__ == '__main__':
    train_model()
