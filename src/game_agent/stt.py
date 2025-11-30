"""Speech-to-Text using Google Speech Recognition."""

import io
import wave
from pathlib import Path

import speech_recognition as sr

from .audio_record.record_audio import AudioRecorder


class SpeechToText:
    """Speech-to-Text handler using AudioRecorder + Google Speech Recognition."""

    def __init__(self, device_index=None):
        """Initialize STT.

        Args:
            device_index: Audio device index. If None, will prompt user to choose.
        """
        print("Inicializando Speech-to-Text (Google Speech Recognition)...")

        # Initialize audio recorder
        self.recorder = AudioRecorder(device_index=device_index)

        # Expose recorder properties for compatibility
        self.device_index = self.recorder.device_index
        self.sample_rate = self.recorder.sample_rate

        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        self.audio_dir = Path("audio_input")
        self.audio_dir.mkdir(exist_ok=True)

        print("Speech-to-Text pronto!")

    def listen_and_transcribe(self, key_to_press='m', language: str = "pt-BR") -> str:
        """Record audio with push-to-talk and transcribe.

        Args:
            key_to_press: Key to use for push-to-talk (default: 'm')
            language: Language code (default: pt-BR for Portuguese Brazil)

        Returns:
            Transcribed text
        """
        # Record audio using push-to-talk
        audio_data = self.recorder.record_with_ptt(key_to_press=key_to_press)

        if audio_data is None:
            return ""

        # Convert to WAV format in memory
        wav_io = self.recorder.audio_to_wav_bytes(audio_data)

        # Use SpeechRecognition to transcribe
        print("🔄 Transcrevendo áudio...")
        try:
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=language)
                print(f"✓ Transcrição: {text}")
                return text
        except sr.UnknownValueError:
            print("⚠️  Não consegui entender o áudio")
            return ""
        except sr.RequestError as e:
            print(f"⚠️  Erro ao acessar o serviço do Google: {e}")
            return ""
        except Exception as e:
            print(f"⚠️  Erro durante transcrição: {e}")
            return ""
