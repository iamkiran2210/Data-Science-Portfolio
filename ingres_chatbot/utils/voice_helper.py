import speech_recognition as sr
from gtts import gTTS
import io

def recognize_speech_from_mic():
    """
    Listens for speech via the microphone and transcribes it to text.

    Returns:
        A dictionary with 'success' (bool) and 'error' or 'text' (str).
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            return {"success": False, "error": "Timeout: No speech detected."}

    try:
        text = r.recognize_google(audio)
        return {"success": True, "text": text}
    except sr.RequestError:
        return {"success": False, "error": "API unavailable. Please check your internet connection."}
    except sr.UnknownValueError:
        return {"success": False, "error": "Unable to recognize speech."}

def convert_text_to_audio(text: str, lang: str = 'en'):
    """
    Converts text to speech in a specified language and returns it as an in-memory audio file.
    """
    try:
        audio_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0) # Rewind the file pointer to the beginning
        return audio_fp
    except Exception as e:
        print(f"Error in TTS conversion: {e}")
        return None
