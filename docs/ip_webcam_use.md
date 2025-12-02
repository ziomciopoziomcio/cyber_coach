## 🚀 Jak to działa?

### **Krok 1: Na telefonie**
1. Zainstaluj **IP Webcam** z Google Play
2. Uruchom aplikację
3. Naciśnij **"Start server"** na dole ekranu
4. Zapisz adres IP (np. `192.168.1.105:8080`)

### **Krok 2: W Pythonie**
```python
from components.phone_camera import IPWebcamClient
import cv2

# Wpisz IP z telefonu
client = IPWebcamClient("http://192.168.1.105:8080")

# Funkcja wywoływana dla każdej klatki
def process_frame(frame):
    # frame to numpy array (BGR)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.stop_stream()

# Start!
client.set_frame_callback(process_frame)
client.start_stream()

while client.is_running:
    pass

cv2.destroyAllWindows()
```

---

## 💡 Dodatkowe funkcje IP Webcam:

```python
import requests

PHONE = "http://192.168.1.105:8080"

# Włącz latarkę
requests.get(f"{PHONE}/enabletorch")

# Przełącz na tylną kamerę
requests.get(f"{PHONE}/settings/ffc?set=off")

# Pobierz pojedyncze zdjęcie
response = requests.get(f"{PHONE}/shot.jpg")
```

---

## ⚠️ Wymagania:

- Telefon i komputer w **tej samej sieci WiFi**
- Biblioteki Python (już w requirements.txt):
  ```
  pip install opencv-python numpy requests flask flask-socketio
  ```

