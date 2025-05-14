# src/camera/processor.py

import numpy as np
import cv2

class EventStreamProcessor:
    def __init__(self):
        # Initialize parameters
        self.buffer = []

    def process_event(self, event):
        """
        Process a single event from the event-based camera.
        Assume 'event' is a dict with {x, y, timestamp, polarity}
        """
        x, y, timestamp, polarity = event["x"], event["y"], event["timestamp"], event["polarity"]

        # Normalize coordinates (example)
        x = x / 240.0  # assuming width=240
        y = y / 180.0  # assuming height=180

        # Convert event into a feature vector
        event_vector = [x, y, timestamp % 1e6, int(polarity)]
        self.buffer.append(event_vector)

        # Keep buffer limited
        if len(self.buffer) > 1000:
            self.buffer.pop(0)

        return event_vector

    def get_current_frame(self):
        """
        Returns a numpy array for inference (e.g., 2D histogram or feature matrix)
        """
        if not self.buffer:
            return np.zeros((100, 4))

        frame = np.array(self.buffer[-100:])  # last 100 events
        return frame
