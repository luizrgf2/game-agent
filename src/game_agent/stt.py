"""Speech-to-Text using OpenAI Whisper."""

import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper


class SpeechToText:
    """Speech-to-Text handler using OpenAI Whisper."""

    def __init__(self, model_size: str = "base"):
        """Initialize STT with Whisper model.

        Args:
            model_size: Whisper model size. Options:
                       - tiny: Fastest, least accurate
                       - base: Good balance (default)
                       - small: Better accuracy
                       - medium: Even better
                       - large: Best accuracy, slowest
        """
        print(f"Carregando modelo Whisper '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("Modelo carregado!")

        # Use the default sample rate from the audio device
        default_device = sd.query_devices(kind='input')
        self.sample_rate = int(default_device['default_samplerate'])
        print(f"Taxa de amostragem do dispositivo: {self.sample_rate} Hz")

        self.audio_dir = Path("audio_input")
        self.audio_dir.mkdir(exist_ok=True)

    def record_audio(self, duration: int = 10) -> str:
        """Record audio from microphone.

        Args:
            duration: Duration in seconds to record

        Returns:
            Path to the recorded audio file
        """
        print(f"🎤 Gravando por {duration} segundos...")
        print("Fale agora!")

        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()  # Wait for recording to finish

        print("✓ Gravação concluída!")

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", dir=self.audio_dir
        )
        output_file = temp_file.name
        temp_file.close()

        # Write WAV file
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        return output_file

    def transcribe_audio(self, audio_file: str, language: str = "pt") -> str:
        """Transcribe audio file to text.

        Args:
            audio_file: Path to audio file
            language: Language code (default: pt for Portuguese)

        Returns:
            Transcribed text
        """
        print("🔄 Transcrevendo áudio...")

        # Transcribe
        result = self.model.transcribe(audio_file, language=language)

        text = result["text"].strip()
        print(f"✓ Transcrição: {text}")

        return text

    def listen_and_transcribe(self, duration: int = 15, language: str = "pt") -> str:
        """Record audio and transcribe in one step.

        Args:
            duration: Duration in seconds to record
            language: Language code

        Returns:
            Transcribed text
        """
        audio_file = self.record_audio(duration)
        text = self.transcribe_audio(audio_file, language)
        return text

    @staticmethod
    def list_audio_devices():
        """List all available audio input devices."""
        print("\nDispositivos de áudio disponíveis:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                print(f"{i}: {device['name']}")
                print(f"   Canais de entrada: {device['max_input_channels']}")
                print(f"   Taxa de amostragem: {device['default_samplerate']} Hz")
                print()
