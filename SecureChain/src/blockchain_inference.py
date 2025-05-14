"""
SecureChain: AI-Driven Event Camera with Blockchain-Verified Anomaly Detection
Anomaly Detection Model Implementation

Author: Rahul R. Nandan
Date: May 9, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import time
import hashlib


class EventEncoder(nn.Module):
    """
    Encodes event-based camera data into latent representations
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, latent_dim: int = 32):
        super(EventEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim*2)
        self.conv3 = nn.Conv2d(hidden_dim*2, latent_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class TemporalAggregator(nn.Module):
    """
    Processes sequence of event frames to capture temporal dynamics
    """
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128):
        super(TemporalAggregator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time, channels, height, width]
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size, seq_len, channels * height * width)
        output, (h_n, _) = self.lstm(x)
        
        # Get final hidden state from both directions
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        concat_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return concat_hidden


class AnomalyDetector(nn.Module):
    """
    Complete anomaly detection model combining event encoding and temporal aggregation
    """
    def __init__(self, 
                 input_dim: int = 4,
                 hidden_dim: int = 64, 
                 latent_dim: int = 32, 
                 lstm_hidden: int = 128, 
                 num_classes: int = 1):
        super(AnomalyDetector, self).__init__()
        
        self.encoder = EventEncoder(input_dim, hidden_dim, latent_dim)
        self.temporal = TemporalAggregator(latent_dim, lstm_hidden)
        
        # Anomaly classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes),
            nn.Sigmoid()
        )
        
        # Anomaly type classification (multi-class)
        self.anomaly_type = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, 5)  # 5 different anomaly types
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch, time, channels, height, width]
        batch_size, seq_len, channels, height, width = x.size()
        
        # Process each frame in the sequence
        encoded_frames = []
        for t in range(seq_len):
            encoded = self.encoder(x[:, t])
            encoded_frames.append(encoded)
        
        # Stack encoded frames
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Process temporal dynamics
        temporal_features = self.temporal(encoded_sequence)
        
        # Anomaly detection (binary classification)
        anomaly_score = self.classifier(temporal_features)
        
        # Anomaly type classification (only used when anomaly is detected)
        anomaly_type_logits = self.anomaly_type(temporal_features)
        
        return anomaly_score, anomaly_type_logits


class EventPreprocessor:
    """
    Preprocesses raw event camera data into format suitable for the model
    """
    def __init__(self, time_window: float = 0.05, spatial_dim: Tuple[int, int] = (128, 128)):
        """
        Args:
            time_window: Time window in seconds for event accumulation
            spatial_dim: Spatial dimensions for event representation (height, width)
        """
        self.time_window = time_window
        self.height, self.width = spatial_dim
        
    def process_events(self, events: np.ndarray) -> np.ndarray:
        """
        Convert raw events to tensor representation
        
        Args:
            events: Numpy array of events with shape [N, 4]
                   Each row contains [x, y, timestamp, polarity]
        
        Returns:
            Event representation as tensor with shape [channels, height, width]
        """
        # Initialize event representation: [pos, neg, time_surface, count]
        representation = np.zeros((4, self.height, self.width), dtype=np.float32)
        
        if events.shape[0] == 0:
            return representation
        
        # Normalize timestamps
        ts = events[:, 2]
        ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
        
        # Process each event
        for i, (x, y, t, p) in enumerate(events):
            x, y = int(x), int(y)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                # Positive events
                if p > 0:
                    representation[0, y, x] += 1
                # Negative events
                else:
                    representation[1, y, x] += 1
                    
                # Time surface (latest event timestamp)
                representation[2, y, x] = ts_norm[i]
                
                # Count surface (event density)
                representation[3, y, x] += 1
        
        # Normalize count surface
        if representation[3].max() > 0:
            representation[3] /= representation[3].max()
        
        return representation


class SecureChainInference:
    """
    Handles real-time inference with blockchain integration
    """
    def __init__(self, model_path: str, device: str = "cuda", confidence_threshold: float = 0.75):
        """
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on (cuda or cpu)
            confidence_threshold: Threshold for anomaly detection
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = AnomalyDetector()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = EventPreprocessor()
        self.confidence_threshold = confidence_threshold
        self.event_buffer = []
        self.sequence_length = 16  # Number of frames in a sequence
        
        # Mapping of anomaly types
        self.anomaly_types = {
            0: "unauthorized_access",
            1: "suspicious_movement",
            2: "object_removal",
            3: "loitering",
            4: "abandoned_object"
        }
        
    def add_events(self, events: np.ndarray) -> None:
        """
        Add new events to the buffer
        
        Args:
            events: Numpy array of events with shape [N, 4]
        """
        self.event_buffer.append(events)
        
        # Keep only the latest sequence_length event frames
        if len(self.event_buffer) > self.sequence_length:
            self.event_buffer.pop(0)
            
    def detect_anomaly(self) -> Dict:
        """
        Run anomaly detection on current event buffer
        
        Returns:
            Dictionary with detection results
        """
        if len(self.event_buffer) < self.sequence_length:
            return {
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_type": None,
                "timestamp": time.time(),
                "data_hash": None
            }
        
        # Preprocess events
        processed_frames = []
        for events in self.event_buffer:
            frame = self.preprocessor.process_events(events)
            processed_frames.append(frame)
            
        # Convert to tensor
        sequence = torch.tensor(np.array(processed_frames)).float().unsqueeze(0)  # [1, seq_len, channels, H, W]
        sequence = sequence.to(self.device)
        
        # Run inference
        with torch.no_grad():
            anomaly_score, anomaly_type_logits = self.model(sequence)
            confidence = anomaly_score.item()
            is_anomaly = confidence > self.confidence_threshold
            
            if is_anomaly:
                anomaly_type_idx = torch.argmax(anomaly_type_logits, dim=1).item()
                anomaly_type = self.anomaly_types[anomaly_type_idx]
            else:
                anomaly_type = None
                
        # Generate data hash for blockchain verification
        data_string = f"{time.time()}_{is_anomaly}_{confidence}_{anomaly_type}"
        data_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": confidence * 100,  # Scale to percentage
            "anomaly_type": anomaly_type,
            "timestamp": time.time(),
            "data_hash": data_hash
        }
        
    def get_confidence_score(self) -> int:
        """
        Get confidence score as integer for blockchain (0-100)
        
        Returns:
            Integer confidence score
        """
        if len(self.event_buffer) < self.sequence_length:
            return 0
            
        # Preprocess events
        processed_frames = []
        for events in self.event_buffer:
            frame = self.preprocessor.process_events(events)
            processed_frames.append(frame)
            
        # Convert to tensor
        sequence = torch.tensor(np.array(processed_frames)).float().unsqueeze(0)
        sequence = sequence.to(self.device)
        
        # Run inference (confidence score only)
        with torch.no_grad():
            anomaly_score, _ = self.model(sequence)
            confidence = int(anomaly_score.item() * 100)  # Convert to integer 0-100
            
        return confidence


if __name__ == "__main__":
    # Example usage
    model = AnomalyDetector()
    print(f"Model architecture:\n{model}")
    
    # Example input
    batch_size = 2
    seq_len = 16
    channels = 4
    height = 128
    width = 128
    
    x = torch.randn(batch_size, seq_len, channels, height, width)
    anomaly_score, anomaly_type = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape - anomaly score: {anomaly_score.shape}")
    print(f"Output shape - anomaly type: {anomaly_type.shape}")
