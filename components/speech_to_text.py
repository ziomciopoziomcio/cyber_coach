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
