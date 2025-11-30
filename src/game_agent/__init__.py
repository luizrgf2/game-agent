"""Game Agent - AI-powered game analysis using screenshots and vision."""

import os
import sys
import queue
import threading
from pathlib import Path

from dotenv import load_dotenv
from pynput import keyboard

from .agent import GameAgent
from .tts import TextToSpeech

# Import STT only when needed to avoid PortAudio dependency error
SpeechToText = None


def main() -> None:
    """Main entry point for the game agent CLI."""
    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        sys.exit(1)

    # Check if features should be enabled
    enable_tts = os.getenv("ENABLE_TTS", "true").lower() == "true"
    enable_stt = os.getenv("ENABLE_STT", "false").lower() == "true"
    ptt_key = os.getenv("PTT_KEY", "m").lower()

    # Create the agent
    agent = GameAgent(api_key=api_key)

    # Create TTS if enabled
    tts = TextToSpeech() if enable_tts else None

    # Create STT if enabled
    stt = None
    if enable_stt:
        try:
            # Import STT only when enabled to avoid dependency errors
            from .stt import SpeechToText as STT
            stt = STT()
        except OSError as e:
            print(f"⚠️  Erro ao inicializar STT: {e}")
            print("Instale PortAudio para usar Speech-to-Text:")
            print("  Linux: sudo apt install portaudio19-dev")
            print("  Mac: brew install portaudio")
            print("  Windows: geralmente já incluído")
            enable_stt = False

    # Push-to-talk state
    is_ptt_pressed = {"value": False}
    audio_chunks = []

    def on_press(key):
        """Handle key press for push-to-talk."""
        try:
            if hasattr(key, 'char') and key.char == ptt_key and not is_ptt_pressed["value"]:
                is_ptt_pressed["value"] = True
                audio_chunks.clear()
                print(f"\n🎤 Gravando... (Segure '{ptt_key.upper()}' pressionado)", flush=True)
        except AttributeError:
            pass

    def on_release(key):
        """Handle key release for push-to-talk."""
        try:
            if hasattr(key, 'char') and key.char == ptt_key and is_ptt_pressed["value"]:
                is_ptt_pressed["value"] = False
                print("🔴 Parou de gravar! Processando...", flush=True)
        except AttributeError:
            pass

    # Setup keyboard listener for push-to-talk
    listener = None
    if enable_stt:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    print("Game Agent initialized!")
    print("=" * 60)
    print("Welcome to Game Agent - AI-powered game analysis")
    print("🔊 Text-to-Speech: ENABLED")
    print("🎤 Speech-to-Text: ENABLED (Push-to-Talk)")
    print(f"⌨️  Pressione e SEGURE '{ptt_key.upper()}' para gravar sua voz")
    print(f"    Solte '{ptt_key.upper()}' quando terminar de falar")
    print("=" * 60)
    print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")

    # Import for audio recording
    import time
    import sounddevice as sd
    import numpy as np

    # Interactive loop - push-to-talk
    while True:
        try:
            # Wait for key press
            while not is_ptt_pressed["value"]:
                time.sleep(0.05)

            # Record while key is pressed
            audio_chunks.clear()
            chunk_duration = 0.1  # 100ms chunks
            samples_per_chunk = int(chunk_duration * stt.sample_rate)

            while is_ptt_pressed["value"]:
                try:
                    chunk = sd.rec(
                        samples_per_chunk,
                        samplerate=stt.sample_rate,
                        channels=1,
                        dtype=np.int16,
                        device=stt.device_index,
                        blocking=True,
                    )
                    audio_chunks.append(chunk)
                except Exception as e:
                    print(f"⚠️  Erro ao gravar chunk: {e}")
                    break

            # Key was released, process audio
            if not audio_chunks:
                print("⚠️  Nenhum áudio gravado")
                print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")
                continue

            # Combine chunks and transcribe
            try:
                import io
                import wave
                import speech_recognition as sr

                audio_data = np.concatenate(audio_chunks, axis=0)

                # Check audio level
                max_val = np.max(np.abs(audio_data))
                if max_val < 100:
                    print(f"⚠️  Áudio muito baixo (pico: {max_val})")
                    print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")
                    continue

                print(f"✓ Nível de áudio OK (pico: {max_val})")

                # Convert to WAV
                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(stt.sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
                wav_io.seek(0)

                # Transcribe
                print("🔄 Transcrevendo áudio...")
                with sr.AudioFile(wav_io) as source:
                    audio = stt.recognizer.record(source)
                    user_input = stt.recognizer.recognize_google(audio, language="pt-BR")
                    print(f"📝 Você disse: {user_input}\n")

            except sr.UnknownValueError:
                print("⚠️  Não consegui entender o áudio")
                print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")
                continue
            except Exception as e:
                print(f"⚠️  Erro ao processar áudio: {e}")
                print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")
                continue

            if not user_input or not user_input.strip():
                print("Nenhum texto detectado")
                print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")
                continue

            # Process with agent
            try:
                print("\nProcessando seu pedido...")
                result = agent.run(user_input)

                # Display response
                print("\n" + "=" * 60)
                messages = result.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    response_text = last_message.content
                    print(f"\nAgent: {response_text}")

                    # Speak response
                    if tts and isinstance(response_text, str):
                        print("\n🔊 Reproduzindo áudio...")
                        try:
                            tts.speak(response_text, play=True)
                        except Exception as e:
                            print(f"⚠️  Erro ao reproduzir: {e}")
                print("=" * 60)
                print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")

            except Exception as e:
                print(f"⚠️  Erro ao processar: {e}")
                print("Verifique sua chave de API no .env")
                print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            if listener:
                listener.stop()
            break
        except Exception as e:
            print(f"\nErro: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nAguardando tecla '{ptt_key.upper()}'...\n")


if __name__ == "__main__":
    main()
