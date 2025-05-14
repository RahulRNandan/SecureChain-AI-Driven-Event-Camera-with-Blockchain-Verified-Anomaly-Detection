"""
SecureChain: AI-Driven Event Camera with Blockchain-Verified Anomaly Detection
Event Camera Interface

Author: Rahul R. Nandan
Date: May 9, 2025
"""

import time
import os
import numpy as np
import threading
import queue
from typing import Dict, List, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SecureChain.Camera")

class EventCameraInterface(ABC):
    """
    Abstract base class for event camera interfaces
    """
    @abstractmethod
    def start(self) -> None:
        """Start event acquisition"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop event acquisition"""
        pass
    
    @abstractmethod
    def get_events(self) -> np.ndarray:
        """Get latest events batch"""
        pass
    
    @abstractmethod
    def get_visualization(self) -> np.ndarray:
        """Get visualized event frame for display"""
        pass


class PropheseeCameraInterface(EventCameraInterface):
    """
    Interface for Prophesee event cameras
    
    Requires Metavision SDK installation:
    https://docs.prophesee.ai/stable/installation/index.html
    """
    def __init__(self, 
                 serial_number: Optional[str] = None,
                 biases_file: Optional[str] = None, 
                 time_window: float = 0.05):
        """
        Initialize Prophesee camera interface
        
        Args:
            serial_number: Serial number of the camera to connect to
            biases_file: Path to camera biases file for configuration
            time_window: Time window in seconds for event accumulation
        """
        try:
            import metavision_hal
            import metavision_sdk_core
            import metavision_sdk_ui
            self.mv_hal = metavision_hal
            self.mv_core = metavision_sdk_core
            self.mv_ui = metavision_sdk_ui
        except ImportError:
            logger.error("Failed to import Metavision SDK. Make sure it's properly installed.")
            raise ImportError("Metavision SDK not found")
            
        self.serial_number = serial_number
        self.biases_file = biases_file
        self.time_window = time_window  # in seconds
        self.time_window_us = int(time_window * 1_000_000)  # in microseconds
        
        # Initialize camera
        self.device = None
        self.camera = None
        self.cd_producer = None
        self.visualization = None
        
        # Event buffer
        self.events_buffer = []
        self.buffer_lock = threading.Lock()
        self.is_running = False
        
        # Event queue
        self.events_queue = queue.Queue(maxsize=100)
        
    def start(self) -> None:
        """Start camera and event acquisition"""
        # Create camera
        if self.serial_number:
            self.device = self.mv_hal.DeviceDiscovery.open_raw_file(self.serial_number)
        else:
            # If no serial number provided, use the first available camera
            devices = self.mv_hal.DeviceDiscovery.get_available_sources()
            if not devices:
                raise RuntimeError("No camera found")
            self.device = self.mv_hal.DeviceDiscovery.open(devices[0])
            
        if not self.device:
            raise RuntimeError("Failed to open camera device")
            
        # Configure camera
        self.camera = self.mv_hal.get_i_camera(self.device)
        self.cd_producer = self.mv_core.get_i_event_cd_decoder(self.device)
        
        if not self.cd_producer:
            raise RuntimeError("Failed to get CD event producer")
            
        # Apply biases if provided
        if self.biases_file and os.path.exists(self.biases_file):
            biases = self.mv_hal.get_i_ll_biases(self.device)
            if biases:
                biases.load_from_file(self.biases_file)
                logger.info(f"Applied biases from {self.biases_file}")
                
        # Create event buffer callback
        def on_events_callback(events_buffer):
            try:
                # Create a copy of the events buffer
                events_array = np.array([[e.x, e.y, e.t, 1 if e.p else -1] for e in events_buffer])
                
                # Add to queue
                if not self.events_queue.full():
                    self.events_queue.put(events_array, block=False)
                    
                # Update visualization data
                with self.buffer_lock:
                    self.events_buffer = events_array
            except Exception as e:
                logger.error(f"Error in events callback: {e}")
        
        # Register callback
        self.cd_producer.add_callback(on_events_callback)
        
        # Start camera
        self.camera.start()
        logger.info("Camera started")
        
        # Set running flag
        self.is_running = True
        
    def stop(self) -> None:
        """Stop camera and event acquisition"""
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
            
        self.is_running = False
        
        # Clear buffers
        with self.buffer_lock:
            self.events_buffer = []
            
        # Clear queue
        while not self.events_queue.empty():
            try:
                self.events_queue.get_nowait()
            except queue.Empty:
                break
                
    def get_events(self) -> np.ndarray:
        """
        Get latest batch of events
        
        Returns:
            Numpy array of events with shape [N, 4]
            Each row contains [x, y, timestamp, polarity]
        """
        try:
            return self.events_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            # Return empty array if no events
            return np.zeros((0, 4))
            
    def get_visualization(self) -> np.ndarray:
        """
        Get visualization of recent events
        
        Returns:
            RGB image for visualization
        """
        with self.buffer_lock:
            if len(self.events_buffer) == 0:
                # Return black image if no events
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
            events = self.events_buffer
            
        # Get camera dimensions
        width = self.camera.geometry().width()
        height = self.camera.geometry().height()
        
        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process events
        if len(events) > 0:
            # Separate positive and negative events
            pos_events = events[events[:, 3] > 0]
            neg_events = events[events[:, 3] < 0]
            
            # Plot positive events in green
            if len(pos_events) > 0:
                x_pos = pos_events[:, 0].astype(int)
                y_pos = pos_events[:, 1].astype(int)
                valid_pos = (x_pos >= 0) & (x_pos < width) & (y_pos >= 0) & (y_pos < height)
                image[y_pos[valid_pos], x_pos[valid_pos], 1] = 255
                
            # Plot negative events in red
            if len(neg_events) > 0:
                x_neg = neg_events[:, 0].astype(int)
                y_neg = neg_events[:, 1].astype(int)
                valid_neg = (x_neg >= 0) & (x_neg < width) & (y_neg >= 0) & (y_neg < height)
                image[y_neg[valid_neg], x_neg[valid_neg], 0] = 255
                
        # Resize for display
        image = cv2.resize(image, (640, 480))
        
        return image


class SimulatedEventCamera(EventCameraInterface):
    """
    Simulated event camera for testing and development
    """
    def __init__(self, 
                 width: int = 128, 
                 height: int = 128,
                 time_window: float = 0.05,
                 simulation_type: str = "random"):
        """
        Initialize simulated event camera
        
        Args:
            width: Width of the simulated sensor
            height: Height of the simulated sensor
            time_window: Time window in seconds for event accumulation
            simulation_type: Type of simulation ('random', 'moving_object', or 'anomaly')
        """
        self.width = width
        self.height = height
        self.time_window = time_window
        self.simulation_type = simulation_type
        
        # State variables
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.last_events = np.zeros((0, 4))
        self.events_lock = threading.Lock()
        
        # Simulation parameters
        self.last_timestamp = 0
        self.anomaly_probability = 0.05  # 5% chance of anomaly
        self.object_position = [width // 2, height // 2]
        self.object_velocity = [2, 1]
        
        # Event buffer
        self.events_queue = queue.Queue(maxsize=100)
        
    def start(self) -> None:
        """Start simulated event generation"""
        if self.is_running:
            return
            
        self.stop_event.clear()
        self.is_running = True
        self.thread = threading.Thread(target=self._generate_events)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started simulated event camera")
        
    def stop(self) -> None:
        """Stop simulated event generation"""
        if not self.is_running:
            return
            
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
            
        self.is_running = False
        logger.info("Stopped simulated event camera")
        
    def get_events(self) -> np.ndarray:
        """
        Get latest batch of events
        
        Returns:
            Numpy array of events with shape [N, 4]
            Each row contains [x, y, timestamp, polarity]
        """
        try:
            return self.events_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            # Return empty array if no events
            return np.zeros((0, 4))
            
    def get_visualization(self) -> np.ndarray:
        """
        Get visualization of recent events
        
        Returns:
            RGB image for visualization
        """
        with self.events_lock:
            events = self.last_events.copy()
            
        # Create image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Process events
        if len(events) > 0:
            # Separate positive and negative events
            pos_events = events[events[:, 3] > 0]
            neg_events = events[events[:, 3] < 0]
            
            # Plot positive events in green
            if len(pos_events) > 0:
                x_pos = pos_events[:, 0].astype(int)
                y_pos = pos_events[:, 1].astype(int)
                valid_pos = (x_pos >= 0) & (x_pos < self.width) & (y_pos >= 0) & (y_pos < self.height)
                image[y_pos[valid_pos], x_pos[valid_pos], 1] = 255
                
            # Plot negative events in red
            if len(neg_events) > 0:
                x_neg = neg_events[:, 0].astype(int)
                y_neg = neg_events[:, 1].astype(int)
                valid_neg = (x_neg >= 0) & (x_neg < self.width) & (y_neg >= 0) & (y_neg < self.height)
                image[y_neg[valid_neg], x_neg[valid_neg], 0] = 255
                
        # Resize for display
        image = cv2.resize(image, (640, 480))
        
        return image
        
    def _generate_events(self) -> None:
        """Event generation thread function"""
        while not self.stop_event.is_set():
            # Generate different types of events based on simulation type
            if self.simulation_type == "random":
                events = self._generate_random_events()
            elif self.simulation_type == "moving_object":
                events = self._generate_moving_object_events()
            elif self.simulation_type == "anomaly":
                # Randomly switch between normal and anomaly events
