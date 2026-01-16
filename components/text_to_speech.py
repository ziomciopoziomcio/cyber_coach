"""
Library for text-to-speech conversion using gTTS (Google Text-to-Speech).
"""
from gtts import gTTS
import os
import tempfile
import playsound
def text_to_speech(text, lang='en'):
    """
    Convert text to speech and play the audio.

    Parameters:
    text (str): The text to be converted to speech.
    lang (str): The language for the TTS conversion (default is 'en' for English).
    """
    try:
        # Create a gTTS object
        tts = gTTS(text=text, lang=lang)

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)

        # Play the audio file
        playsound.playsound(temp_file)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

