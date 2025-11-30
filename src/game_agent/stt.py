"""Speech-to-Text using Google Speech Recognition."""

import io
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import speech_recognition as sr


class SpeechToText:
    """Speech-to-Text handler using sounddevice + Google Speech Recognition."""

    def __init__(self):
        """Initialize STT."""
        print("Inicializando Speech-to-Text (Google Speech Recognition)...")

        # Suppress ALSA warnings
        import os
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

        # Find first device with input channels
        self.device_index = None
        self.sample_rate = 44100

        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                self.device_index = idx
                self.sample_rate = int(device['default_samplerate'])
                print(f"Usando dispositivo {idx}: {device['name']}")
                print(f"Taxa de amostragem: {self.sample_rate} Hz")
                break

        if self.device_index is None:
            print("⚠️  Nenhum dispositivo de entrada encontrado! Usando padrão.")
            self.device_index = 1  # Force device 1 as fallback

        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        self.audio_dir = Path("audio_input")
        self.audio_dir.mkdir(exist_ok=True)

        print("Speech-to-Text pronto!")

    def listen_and_transcribe(self, duration: int = 15, language: str = "pt-BR") -> str:
        """Record audio and transcribe in one step.

        Args:
            duration: Duration in seconds to record
            language: Language code (default: pt-BR for Portuguese Brazil)

        Returns:
            Transcribed text
        """
        print(f"🎤 Gravando por {duration} segundos...")
        print("Fale agora! (FALE BEM ALTO E CLARO)")

        # Record audio with sounddevice using default device
        try:
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                device=self.device_index,
                blocking=True,
            )
        except Exception as e:
            print(f"⚠️  Erro ao gravar: {e}")
            return ""

        print("✓ Gravação concluída!")

        # Check audio level
        max_val = np.max(np.abs(audio_data))
        if max_val < 100:
            print(f"⚠️  Áudio muito baixo (pico: {max_val}) - tente falar mais alto")
            return ""
        else:
            print(f"✓ Nível de áudio OK (pico: {max_val})")

        # Convert to WAV format in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        wav_io.seek(0)

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
