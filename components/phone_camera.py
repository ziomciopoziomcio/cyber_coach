"""
Phone Camera Module - Obsługa przekazywania obrazu z kamery telefonu po sieci

Ten moduł obsługuje odbieranie obrazów z kamery telefonu przez sieć
używając Flask-SocketIO dla komunikacji w czasie rzeczywistym.
Wspiera również IP Webcam (MJPEG stream).
"""

import logging
import threading
import time
from typing import Optional, Callable

import cv2
import numpy as np
import requests

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

    def __init__(self, ip_webcam_url: str = "http://192.168.1.100:8080", *,
                 backoff_base: float = 0.5, max_backoff: float = 5.0,
                 max_consecutive_errors: int = 10, max_buffer_size_bytes: int = 10 * 1024 * 1024,
                 log_decode_exceptions: bool = False):
        """
        Inicjalizacja klienta IP Webcam.

        Args:
            ip_webcam_url: URL do IP Webcam (np. http://192.168.1.100:8080)
            backoff_base: podstawowy czas backoffu przy reconnectach (exponential backoff)
            max_backoff: maksymalny czas w sekundach między próbami reconnectu
            max_consecutive_errors: po ilu kolejnych błędach przerywamy próbę łączenia
            max_buffer_size_bytes: maksymalny rozmiar wewnętrznego bufora MJPEG; po przekroczeniu czyścimy
            log_decode_exceptions: jeśli True, logujemy szczegóły wyjątków dekodowania (może być głośne)
        """
        self.base_url = ip_webcam_url.rstrip('/')
        self.video_url = f"{self.base_url}/video"
        self.shot_url = f"{self.base_url}/shot.jpg"

        self.current_frame: Optional[np.ndarray] = None
        self.frame_callback: Optional[Callable] = None
        self.is_running = False
        self._stream_thread = None
        self._frame_lock = threading.Lock()

        # reconnect / robustness settings
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self.max_consecutive_errors = max_consecutive_errors
        self.max_buffer_size_bytes = max_buffer_size_bytes
        self.log_decode_exceptions = log_decode_exceptions

        # Stats
        self.frames_received = 0
        self.frames_dropped = 0

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
        """Główna pętla odbierająca strumień MJPEG.

        Zmieniono aby była odporna na chwilowe przerwy połączenia/przesyłu:
        - ignoruje puste/chybione kawałki danych zamiast przerywać działanie
        - próbuje ponownie po krótkim backoffie w przypadku błędów sieciowych
        - ogranicza częstotliwość ponownych prób, by nie tworzyć busy-loop
        """
        logger.info(f"Starting IP Webcam stream from {self.video_url}")

        consecutive_errors = 0

        # Dopóki klient jest uruchomiony, próbuj (ponownie) połączyć/odczytywać
        while self.is_running:
            try:
                response = requests.get(self.video_url, stream=True, timeout=10)

                if response.status_code != 200:
                    logger.error(f"Failed to connect: HTTP {response.status_code}")
                    consecutive_errors += 1
                    sleep_time = min(self.backoff_base * (2 ** consecutive_errors),
                                     self.max_backoff)
                    time.sleep(sleep_time)
                    continue

                # Reset błędów po poprawnym połączeniu
                consecutive_errors = 0

                # Bufor do odczytu MJPEG
                bytes_data = bytes()
                last_frame_time = time.time()

                for chunk in response.iter_content(chunk_size=8192):  # Większe chunki
                    if not self.is_running:
                        break

                    # Chunk może być pusty gdy łącze chwilowo nie wysyła danych
                    if not chunk:
                        # krótki sleep żeby nie spinować CPU
                        time.sleep(0.001)
                        continue

                    bytes_data += chunk

                    # Regularnie czyść stary bufor jeśli jest za duży (zapobiega memory leaks)
                    if len(bytes_data) > self.max_buffer_size_bytes:
                        logger.warning(f"Buffer overflow ({len(bytes_data)} bytes), resetting")
                        # Znajdź ostatni kompletny start markera i zachowaj tylko od niego
                        last_start = bytes_data.rfind(b'\xff\xd8')
                        if last_start > 0:
                            bytes_data = bytes_data[last_start:]
                        else:
                            bytes_data = bytes()
                        continue

                    # MJPEG format: każda klatka zaczyna się od FFD8 i kończy FFD9
                    while True:  # Przetwórz wszystkie kompletne ramki w buforze
                        a = bytes_data.find(b'\xff\xd8')  # Start JPEG
                        b = bytes_data.find(b'\xff\xd9')  # End JPEG

                        if a == -1 or b == -1 or b <= a:
                            # Brak kompletnej ramki, usuń śmieci przed startem
                            if a > 0:
                                bytes_data = bytes_data[a:]
                            elif a == -1 and len(
                                    bytes_data) > 50000:  # Jeśli brak początku i dużo danych, wyczyść
                                bytes_data = bytes()
                            break

                        jpg = bytes_data[a:b + 2]
                        bytes_data = bytes_data[b + 2:]

                        # Dekoduj obraz - imdecode może zwrócić None dla uszkodzonych danych
                        try:
                            arr = np.frombuffer(jpg, dtype=np.uint8)
                            if arr.size == 0:
                                # pusta ramka - pomiń
                                logger.debug("Received empty JPEG buffer, skipping")
                                self.frames_dropped += 1
                                continue

                            try:
                                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            except Exception as cv_err:
                                # Cv2 may raise cv2.error; avoid printing full stack by default
                                if self.log_decode_exceptions:
                                    logger.exception(f"cv2.imdecode error: {cv_err}")
                                else:
                                    logger.debug(f"cv2.imdecode error (skipping frame): {cv_err}")
                                self.frames_dropped += 1
                                continue

                            if frame is None:
                                logger.debug(
                                    "cv2.imdecode returned None (corrupted JPEG?), skipping frame")
                                self.frames_dropped += 1
                                continue

                            # Ustaw ramkę i wywołaj callback (thread-safe)
                            with self._frame_lock:
                                self.current_frame = frame
                                self.frames_received += 1

                            current_time = time.time()
                            fps = 1.0 / (
                                        current_time - last_frame_time) if current_time > last_frame_time else 0
                            last_frame_time = current_time

                            if self.frames_received % 100 == 0:
                                logger.info(
                                    f"Frames: {self.frames_received}, Dropped: {self.frames_dropped}, FPS: {fps:.1f}")

                            if self.frame_callback:
                                try:
                                    self.frame_callback(frame)
                                except Exception as cb_e:
                                    logger.exception(f"Frame callback error: {cb_e}")

                        except Exception as decode_e:
                            # Nie przerywamy całego streamu z powodu pojedynczego złego kawałka
                            if self.log_decode_exceptions:
                                logger.debug(f"Frame decode error (skipping): {decode_e}")
                            else:
                                logger.debug("Frame decode error (skipping)")
                            self.frames_dropped += 1
                            continue

                # Jeśli wyszliśmy z pętli iter_content bez self.is_running, prawdopodobnie zakończono
                if not self.is_running:
                    break

                # Jeżeli połączenie zostało przerwane ze strony serwera, spróbujmy ponownie
                logger.info("Stream connection closed by server, will attempt reconnect")
                consecutive_errors += 1
                sleep_time = min(self.backoff_base * (2 ** consecutive_errors), self.max_backoff)
                time.sleep(sleep_time)

            except requests.RequestException as req_e:
                logger.warning(f"Stream network error: {req_e}")
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    logger.error("Maximum consecutive connection errors reached, stopping stream")
                    break
                sleep_time = min(self.backoff_base * (2 ** consecutive_errors), self.max_backoff)
                time.sleep(sleep_time)
                continue
            except Exception as e:
                # Nieprzewidziane błędy - loguj bez długiego stosu (zachowaj prosty komunikat)
                logger.error(f"Stream unexpected error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= self.max_consecutive_errors:
                    logger.error("Maximum consecutive errors reached, stopping stream")
                    break
                sleep_time = min(self.backoff_base * (2 ** consecutive_errors), self.max_backoff)
                time.sleep(sleep_time)
                continue

        # Zakończenie pracy pętli
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

    def set_frame_callback(self, callback: Callable):
        """
        Ustawia funkcję callback wywoływaną po otrzymaniu nowego obrazu.

        Args:
            callback: Funkcja przyjmująca jeden argument (numpy array z obrazem)
        """
        self.frame_callback = callback

    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Zwraca ostatni otrzymany obraz (thread-safe copy).

        Returns:
            numpy array z obrazem lub None jeśli nie ma obrazu
        """
        with self._frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None


# Przykład użycia
if __name__ == '__main__':
    from queue import Queue, Empty

    # używamy kolejki aby przenieść wyświetlanie obrazu do głównego wątku
    frame_q: Queue = Queue(maxsize=2)


    def process_frame(frame):
        """Callback wywoływany z wątku odbioru streamu.

        Nie wolno wywoływać cv2.imshow / cv2.waitKey z wątku innym niż główny na Windows.
        Tutaj tylko wpychamy klatkę do kolejki, a wyświetlanie robi pętla główna.
        """
        try:
            # put_nowait, aby nie blokować wątku strumienia; nadpisywanie jest OK
            if frame_q.full():
                # odrzuć najstarszą klatkę, żeby zrobić miejsce (podejście "drop oldest")
                try:
                    _ = frame_q.get_nowait()
                except Exception:
                    pass
            frame_q.put_nowait(frame)
        except Exception as e:
            logger.debug(f"Failed to enqueue frame: {e}")


    ip = input("Podaj IP telefonu (np. 192.168.1.100): ").strip() or "192.168.1.100"
    port = input("Podaj port IP Webcam (domyślnie 8080): ").strip() or "8080"

    url = f"http://{ip}:{port}"
    client = IPWebcamClient(url)

    print(f"\nTestowanie połączenia z {url}...")
    if client.test_connection():
        print("✓ Połączenie OK!")
        client.set_frame_callback(process_frame)
        client.start_stream()

        print("\nStreamowanie... Naciśnij 'q' w oknie obrazu aby zakończyć")
        try:
            while client.is_running:
                try:
                    # czekaj krótką chwilę na nową klatkę
                    frame = frame_q.get(timeout=0.5)
                    # wyświetlanie w głównym wątku — bezpieczne na Windows
                    cv2.imshow('Phone Camera', frame)
                except Empty:
                    # brak klatki w kolejce w tym cyklu
                    pass

                # obsługa klawisza 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nZatrzymywanie...")
        finally:
            client.stop_stream()
            cv2.destroyAllWindows()
    else:
        print("✗ Nie można połączyć z IP Webcam")
        print(f"\nSprawdź czy:")
        print(f"1. IP Webcam działa na telefonie")
        print(f"2. Używasz poprawnego adresu: {url}")
        print(f"3. Telefon i komputer są w tej samej sieci")
