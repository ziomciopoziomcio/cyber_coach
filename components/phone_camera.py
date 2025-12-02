
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
from typing import Optional, Callable
import logging
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IPWebcamClient:

    def __init__(self, ip_webcam_url: str = "http://192.168.1.100:8080"):
        """
        Inicjalizacja klienta IP Webcam.

        Args:
            ip_webcam_url: URL do IP Webcam (np. http://192.168.1.100:8080)
        """
        self.base_url = ip_webcam_url.rstrip('/')
        self.video_url = f"{self.base_url}/video"
        self.shot_url = f"{self.base_url}/shot.jpg"

        self.current_frame: Optional[np.ndarray] = None
        self.frame_callback: Optional[Callable] = None
        self.is_running = False
        self._stream_thread = None

