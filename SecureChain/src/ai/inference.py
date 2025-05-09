# inference.py - Inference Engine (Stub)

from model import SimpleAnomalyDetector
import torch

def infer(input_tensor):
    model = SimpleAnomalyDetector(input_dim=10)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output