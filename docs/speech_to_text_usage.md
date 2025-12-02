# Speech-to-Text Library Usage

The `components/speech_to_text.py` module provides offline speech recognition using VOSK.

## Setup

1. **Install dependencies:**
```powershell
pip install vosk sounddevice numpy
```

2. **Set up VOSK model** (one-time setup):

The model is already installed at `C:\models\vosk-model-small-en-us-0.15`

**For each new terminal session**, set the environment variable:
```powershell
$env:VOSK_MODEL_PATH = "C:\models\vosk-model-small-en-us-0.15"
```

**Or set it permanently** (already done on your system):
```powershell
setx VOSK_MODEL_PATH "C:\models\vosk-model-small-en-us-0.15"
```

## Usage

### Quick Import

```python
from components.speech_to_text import start_listening, stop_listening, transcribe_once
```

### Example 1: Continuous Background Listening

```python
from components.speech_to_text import start_listening, stop_listening
import time

def on_transcription(text: str, is_final: bool):
    if is_final:
        print(f"Final: {text}")
    else:
        print(f"Partial: {text}")

# Start listening
start_listening(callback=on_transcription)

# Do other work while listening in background
time.sleep(30)

# Stop when done
stop_listening()
```

### Example 2: One-Shot Transcription

```python
from components.speech_to_text import transcribe_once

# Record and transcribe 5 seconds
result = transcribe_once(duration=5.0)
print(f"You said: {result}")
```

### Example 3: Integration with Application

```python
from components.speech_to_text import start_listening, stop_listening

class MyApp:
    def __init__(self):
        self.running = False
    
    def on_speech(self, text: str, is_final: bool):
        if is_final and text.strip():
            # Process the command
            self.handle_command(text)
    
    def handle_command(self, text: str):
        # Your application logic here
        print(f"Processing: {text}")
    
    def start(self):
        start_listening(callback=self.on_speech)
        self.running = True
    
    def stop(self):
        stop_listening()
        self.running = False

# Use it
app = MyApp()
app.start()
# ... app runs ...
app.stop()
```

## API Reference

### `start_listening(callback, backend='vosk', model_path=None, sample_rate=16000)`

Start continuous background speech recognition.

**Parameters:**
- `callback`: Function(text: str, is_final: bool) - Called with transcription results
- `backend`: 'vosk' (default, more backends coming soon)
- `model_path`: Optional path to VOSK model (uses VOSK_MODEL_PATH env var if None)
- `sample_rate`: Audio sample rate (default: 16000 Hz)

**Example:**
```python
def my_callback(text, is_final):
    if is_final:
        print(f"Final result: {text}")

start_listening(callback=my_callback)
```

### `stop_listening()`

Stop the background listening session and flush any remaining audio.

### `transcribe_once(duration=5.0, backend='vosk', model_path=None, sample_rate=16000)`

Blocking one-shot transcription.

**Parameters:**
- `duration`: How many seconds to record (default: 5.0)
- `backend`: 'vosk' (default)
- `model_path`: Optional path to VOSK model
- `sample_rate`: Audio sample rate (default: 16000 Hz)

**Returns:** String with transcription result

**Example:**
```python
text = transcribe_once(duration=3.0)
print(text)
```

## Example Scripts

See the `examples/` directory:
- `examples/stt_example.py` - Basic usage examples
- `examples/coach_with_stt.py` - Integration with a coaching application

## Running Examples

**Simple example:**
```powershell
$env:VOSK_MODEL_PATH = "C:\models\vosk-model-small-en-us-0.15"
python examples/stt_example.py
```

**Cyber coach example:**
```powershell
$env:VOSK_MODEL_PATH = "C:\models\vosk-model-small-en-us-0.15"
python examples/coach_with_stt.py
```

## Troubleshooting

### "VOSK model not found" error
Make sure `VOSK_MODEL_PATH` is set in your current terminal session:
```powershell
$env:VOSK_MODEL_PATH = "C:\models\vosk-model-small-en-us-0.15"
```

### "No sound" or empty transcriptions
- Check your microphone is connected and working
- Verify default audio input device in Windows settings
- Speak clearly and close to the microphone
- The English model works best with English speech

### Unicode path errors
The model must be in a path with only ASCII characters (no accented letters like Ł, ó). 
This is why we use `C:\models\...` instead of the OneDrive path.

## Notes

- **Language**: Current model is English (vosk-model-small-en-us-0.15)
- **Offline**: Everything runs locally, no internet required
- **Thread-safe**: Background listening runs in a separate thread
- **Performance**: Small model is fast but less accurate; larger models available at https://alphacephei.com/vosk/models/

