# model.py - Anomaly Detection Model (Stub)

import torch.nn as nn

class SimpleAnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(SimpleAnomalyDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)