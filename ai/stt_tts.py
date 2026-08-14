import speech_recognition as sr
import requests
import tempfile
import pygame
import os

from dotenv import load_dotenv

load_dotenv()


r = sr.Recognizer()

def record_audio():
    """record audio and convert it to text"""

    # connect user mic and listen to audio
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio = r.listen(source)

        # converting user audio to text using google
        try:
            text = r.recognize_google(audio)
            return text
        except Exception as e:
            print("Error: ", e)
            return None


"""
source: https://www.youtube.com/watch?v=3BMy5KPa_kQ 
"""
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

def get_api_key():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY environment variable is not set. "
            "Copy .env.example to .env and add your key."
        )
    return api_key

def play_audio(file_path):
    """plays audio through pytgame"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
    except Exception as e:
        print(f"Error playing audio: {str(e)}")

def output_audio(text):
    voice_id = DEFAULT_VOICE_ID

    # API endpoint for text-to-speech conversion
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": get_api_key()
    }
    
    # Request payload
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_filename = temp_file.name
            temp_file.close()
            
            # Save the audio to the temporary file
            with open(temp_filename, "wb") as f:
                f.write(response.content)
            
            
            # Play the audio
            play_audio(temp_filename)
            
            # Delete the file after playing
            try:
                os.unlink(temp_filename)
            except Exception as e:
                print(f"Error deleting temporary file: {str(e)}")
            
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False
