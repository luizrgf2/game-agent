"""Speech-to-Text using Google Speech Recognition."""

import io
import os
import wave
from pathlib import Path

import speech_recognition as sr

from .audio_record.record_audio import AudioRecorder
from .audio_record.record_audio_xbox import AudioRecorderXbox


class SpeechToText:
    """Speech-to-Text handler using AudioRecorder + Google Speech Recognition."""

    def __init__(self, device_index=None, ptt_mode=None, ptt_key=None, xbox_button=None):
        """Initialize STT.

        Args:
            device_index: Audio device index. If None, will prompt user to choose.
            ptt_mode: Push-to-talk mode ('keyboard' or 'xbox'). If None, reads from env.
            ptt_key: Key for keyboard mode. If None, reads from env.
            xbox_button: Button for xbox mode. If None, reads from env.
        """
        print("Inicializando Speech-to-Text (Google Speech Recognition)...")

        # Determine PTT mode from env if not specified
        if ptt_mode is None:
            ptt_mode = os.getenv("PTT_MODE", "keyboard").lower()

        self.ptt_mode = ptt_mode

        # Initialize appropriate recorder based on mode
        if self.ptt_mode == "xbox":
            print("Modo: Controle Xbox")
            self.recorder = AudioRecorderXbox(device_index=device_index)
            self.xbox_button = xbox_button or os.getenv("XBOX_BUTTON")
        else:
            print("Modo: Teclado")
            self.recorder = AudioRecorder(device_index=device_index)
            self.ptt_key = ptt_key or os.getenv("PTT_KEY", "m").lower()

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

    def listen_and_transcribe(self, language: str = "pt-BR") -> str:
        """Record audio with push-to-talk and transcribe.

        Args:
            language: Language code (default: pt-BR for Portuguese Brazil)

        Returns:
            Transcribed text
        """
        # Record audio using appropriate push-to-talk method
        if self.ptt_mode == "xbox":
            audio_data = self.recorder.record_with_xbox_ptt(button=self.xbox_button)
        else:
            audio_data = self.recorder.record_with_ptt(key_to_press=self.ptt_key)

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
