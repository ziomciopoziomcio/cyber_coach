
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
    """
    Klient do odbierania strumienia z aplikacji IP Webcam.

    IP Webcam udostępnia kilka endpointów:
    - http://<phone-ip>:8080/video - MJPEG stream
    - http://<phone-ip>:8080/shot.jpg - pojedyncze zdjęcie
    - http://<phone-ip>:8080/videofeed - alternatywny feed
    """

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

    def test_connection(self) -> bool:
        """
        Testuje połączenie z IP Webcam.

        Returns:
            True jeśli połączenie działa, False w przeciwnym razie
        """
        try:
            response = requests.get(self.shot_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_single_frame(self) -> Optional[np.ndarray]:
        """
        Pobiera pojedynczą klatkę z IP Webcam.

        Returns:
            numpy array z obrazem lub None w przypadku błędu
        """
        try:
            response = requests.get(self.shot_url, timeout=5)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
        return None

    def _stream_loop(self):
        """Główna pętla odbierająca strumień MJPEG."""
        logger.info(f"Starting IP Webcam stream from {self.video_url}")

        try:
            response = requests.get(self.video_url, stream=True, timeout=10)

            if response.status_code != 200:
                logger.error(f"Failed to connect: HTTP {response.status_code}")
                self.is_running = False
                return

            # Bufor do odczytu MJPEG
            bytes_data = bytes()

            for chunk in response.iter_content(chunk_size=1024):
                if not self.is_running:
                    break

                bytes_data += chunk

                # MJPEG format: każda klatka zaczyna się od FFD8 i kończy FFD9
                a = bytes_data.find(b'\xff\xd8')  # Start JPEG
                b = bytes_data.find(b'\xff\xd9')  # End JPEG

                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]

                    # Dekoduj obraz
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is not None:
                        self.current_frame = frame
                        if self.frame_callback:
                            self.frame_callback(frame)

        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            self.is_running = False
            logger.info("Stream stopped")

    def start_stream(self):
        """Rozpoczyna odbieranie strumienia w osobnym wątku."""
        if self.is_running:
            logger.warning("Stream already running")
            return

        self.is_running = True
        self._stream_thread = threading.Thread(target=self._stream_loop)
        self._stream_thread.daemon = True
        self._stream_thread.start()
        logger.info("IP Webcam stream started")

    def stop_stream(self):
        """Zatrzymuje odbieranie strumienia."""
        self.is_running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2)
        logger.info("IP Webcam stream stopped")

