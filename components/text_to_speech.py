"""
Library for text-to-speech conversion using gTTS (Google Text-to-Speech).
"""
from gtts import gTTS
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

