"""
Simple microphone -> text module.

API:
- start_listening(callback, backend='vosk', model_path=None, sample_rate=16000)
    Starts a background audio capture and calls callback(text, is_final) for partial/final results.

- transcribe_once(duration=5.0, backend='vosk', model_path=None, sample_rate=16000)
    Blocking one-shot transcription of `duration` seconds.

- stop_listening()
    Stops background listening and emits final result via callback.

This module defaults to VOSK (offline). If VOSK or sounddevice are not installed or model not found,
functions will raise informative RuntimeError explaining how to install or configure dependencies.
"""
from __future__ import annotations

import os
import threading
import queue
import time
from typing import Callable, Optional, Tuple

# Module-level state for background listening
_capture = None
_worker_thread: Optional[threading.Thread] = None
_running = False
_callback: Optional[Callable[[str, bool], None]] = None
_backend = None
_lock = threading.Lock()

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MODEL_ENV = "VOSK_MODEL_PATH"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Language to model directory mapping
LANGUAGE_MODELS = {
    "pl": "vosk-model-small-pl-0.22",
    "en": "vosk-model-small-en-us-0.15",
}
DEFAULT_LANGUAGE = "pl"


class _MissingDependencyError(RuntimeError):
    pass


class VoskBackend:
    """Backend wrapper around vosk.KaldiRecognizer.

    feed(pcm_bytes) -> (text, is_final)
    finish() -> final_text
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        try:
            from vosk import Model, KaldiRecognizer
        except Exception as e:  # pragma: no cover - dependency check
            raise _MissingDependencyError(
                "VOSK is not installed. Install with: pip install vosk"
            ) from e

        if model_path is None:
            # try env var
            model_path = os.environ.get(DEFAULT_MODEL_ENV)
        if model_path is None:
            # try to find model based on language
            if language in LANGUAGE_MODELS:
                candidate = os.path.join(DEFAULT_MODEL_DIR, LANGUAGE_MODELS[language])
                if os.path.isdir(candidate):
                    model_path = candidate
        if model_path is None:
            # fallback: choose the first subdir in components/models if exists
            if os.path.isdir(DEFAULT_MODEL_DIR):
                try:
                    first = next(os.scandir(DEFAULT_MODEL_DIR))
                    model_path = first.path
                except StopIteration:
                    model_path = None

        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"VOSK model not found. Set environment {DEFAULT_MODEL_ENV} or place a model under {DEFAULT_MODEL_DIR}."
            )

        self.sample_rate = sample_rate
        # Some native VOSK builds (C++ layer) on Windows cannot handle non-ASCII
        # characters in filesystem paths. Detect this early and give an actionable
        # message so the user can move the model to an ASCII-only path.
        try:
            if any(ord(ch) > 127 for ch in model_path):
                raise RuntimeError(
                    "VOSK model path contains non-ASCII characters (e.g. accented letters).\n"
                    "The native VOSK backend on Windows often fails to open paths with Unicode characters.\n"
                    "Quick fix: move the model to a path with only ASCII characters, for example: C:\\models\\vosk-model-small-en-us-0.15\n"
                    "Then either set the environment variable VOSK_MODEL_PATH to that path or pass it as model_path when constructing the backend.\n"
                )
        except TypeError:
            # model_path may be None or not a str; ignore here (will fail later)
            pass

        # Try to create the VOSK model; if it fails provide diagnostics and actionable tips
        try:
            self.model = Model(model_path)
        except Exception as e:
            # Collect some quick diagnostics about the path to help the user
            try:
                entries = os.listdir(model_path) if os.path.isdir(model_path) else []
            except Exception:
                entries = []
            diag = (
                f"Failed to create VOSK model from path: {model_path}\n"
                f"Directory exists: {os.path.isdir(model_path)}\n"
                f"Top-level entries (up to 10): {entries[:10]}\n\n"
                "Common causes:\n"
                " - The model archive wasn't unzipped (pass the unzipped folder, not the .zip).\n"
                " - The downloaded model is corrupted or incomplete. Try re-downloading.\n"
                " - VOSK / wheel mismatch for your platform (use a matching vosk package).\n\n"
                "Quick PowerShell checks (run in project root):\n"
                f"  Get-ChildItem -Path '{model_path}' -Force\n"
                "Quick download (small English model):\n"
                "  mkdir -Force .\\components\\models; Invoke-WebRequest \"https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\" -OutFile .\\components\\models\\model.zip; Expand-Archive -Path .\\components\\models\\model.zip -DestinationPath .\\components\\models\\; Remove-Item .\\components\\models\\model.zip\n\n"
                "If this doesn't help, re-download a matching model from: https://alphacephei.com/vosk/models/"
            )
            raise RuntimeError(diag) from e

        self.rec = KaldiRecognizer(self.model, float(sample_rate))
        # optional: allow word-level timestamps by passing JSON options

    def feed(self, pcm_bytes: bytes) -> Tuple[Optional[str], bool]:
        """Feed PCM16LE bytes. Returns (text, is_final).
        If partial result available, returns (text, False).
        If final result, returns (text, True).
        If no new text, returns (None, False).
        """
        # KaldiRecognizer.AcceptWaveform expects bytes
        accepted = self.rec.AcceptWaveform(pcm_bytes)
        if accepted:
            res = self.rec.Result()
            # Result is JSON like {"text":"..."}
            try:
                import json

                j = json.loads(res)
                text = j.get("text", "")
            except Exception:
                text = res
            return (text.strip(), True)
        else:
            try:
                partial = self.rec.PartialResult()
                import json

                j = json.loads(partial)
                p = j.get("partial", "")
            except Exception:
                p = ""
            if p:
                return (p.strip(), False)
        return (None, False)

    def finish(self) -> Optional[str]:
        try:
            res = self.rec.FinalResult()
            import json

            j = json.loads(res)
            return j.get("text", "").strip()
        except Exception:
            return None


class AudioCapture:
    """Capture audio from default microphone using sounddevice and push to a queue as raw PCM16 bytes."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, dtype: str = "int16"):
        try:
            import sounddevice as sd
        except Exception as e:  # pragma: no cover - dependency check
            raise _MissingDependencyError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from e

        self.sd = sd
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.q: "queue.Queue[bytes]" = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        # indata is a numpy array shaped (frames, channels)
        try:
            # Ensure we copy the buffer because sounddevice reuses memory
            b = indata.copy().tobytes()
            self.q.put(b, block=False)
        except queue.Full:
            # drop frame
            pass

    def start(self):
        if self.stream is not None:
            return
        self.stream = self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=self.dtype,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.stream = None

    def read(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


# Public API

def start_listening(
    callback: Callable[[str, bool], None],
    backend: str = "vosk",
    model_path: Optional[str] = None,
    language: str = DEFAULT_LANGUAGE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Start background listening. callback(text, is_final)

    Args:
        callback: Function to call with (text, is_final) results
        backend: Recognition backend (default: 'vosk')
        model_path: Optional explicit path to model (overrides language selection)
        language: Language code ('pl' or 'en', default: 'pl')
        sample_rate: Audio sample rate in Hz (default: 16000)

    Raises RuntimeError if required dependencies or models are missing.
    """
    global _capture, _worker_thread, _running, _callback, _backend

    with _lock:
        if _running:
            raise RuntimeError("Already listening")
        _callback = callback
        # initialize backend
        if backend == "vosk":
            _backend = VoskBackend(model_path=model_path, language=language, sample_rate=sample_rate)
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

        _capture = AudioCapture(sample_rate=sample_rate)
        _capture.start()
        _running = True

        def worker():
            try:
                buffer = bytearray()
                # recommended chunk size; sounddevice gives frames of some small block
                while True:
                    with _lock:
                        running = _running
                    if not running:
                        break
                    chunk = _capture.read(timeout=0.5)
                    if chunk:
                        # feed directly to recognizer
                        text, is_final = _backend.feed(chunk)
                        if text is not None and _callback:
                            try:
                                _callback(text, is_final)
                            except Exception:
                                pass
                    else:
                        # no audio; continue
                        time.sleep(0.01)
                # when stopped, flush final
                final = _backend.finish()
                if final and _callback:
                    try:
                        _callback(final, True)
                    except Exception:
                        pass
            finally:
                # cleanup
                try:
                    _capture.stop()
                except Exception:
                    pass

        _worker_thread = threading.Thread(target=worker, daemon=True)
        _worker_thread.start()


def stop_listening() -> None:
    """Stop background listening (if any). Emits final result via callback."""
    global _running, _worker_thread
    with _lock:
        if not _running:
            return
        _running = False
    # wait briefly for worker to flush
    if _worker_thread is not None:
        _worker_thread.join(timeout=5.0)


def transcribe_once(
    duration: float = 5.0,
    backend: str = "vosk",
    model_path: Optional[str] = None,
    language: str = DEFAULT_LANGUAGE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Blocking one-shot transcription using the selected backend.

    Args:
        duration: Recording duration in seconds (default: 5.0)
        backend: Recognition backend (default: 'vosk')
        model_path: Optional explicit path to model (overrides language selection)
        language: Language code ('pl' or 'en', default: 'pl')
        sample_rate: Audio sample rate in Hz (default: 16000)

    Returns:
        Transcribed text as string

    Records `duration` seconds from default microphone and returns the final transcript.
    """
    # Use sounddevice's blocking record for simplicity
    try:
        import sounddevice as sd
        import numpy as np
    except Exception as e:  # pragma: no cover - dependency check
        raise _MissingDependencyError(
            "sounddevice (and numpy) are required for transcribe_once. Install with: pip install sounddevice numpy"
        ) from e

    if backend != "vosk":
        raise RuntimeError(f"Unsupported backend: {backend}")

    backend_obj = VoskBackend(model_path=model_path, language=language, sample_rate=sample_rate)

    frames = int(duration * sample_rate)
    try:
        rec = sd.rec(frames=frames, samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
    except Exception as e:
        raise RuntimeError(f"Error recording audio: {e}") from e

    # rec is numpy array shape (frames, 1)
    if hasattr(rec, "tobytes"):
        pcm = rec.tobytes()
    else:
        pcm = bytes(rec)

    # VOSK recognizes in chunks; feed entire buffer
    text, is_final = backend_obj.feed(pcm)
    final = backend_obj.finish()
    if final:
        return final
    if text:
        return text
    return ""


__all__ = ["start_listening", "stop_listening", "transcribe_once"]

