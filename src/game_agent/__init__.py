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

    # Queue for voice commands triggered by hotkey
    voice_queue = queue.Queue()
    is_recording = threading.Lock()

    def on_hotkey_pressed():
        """Callback when hotkey is pressed."""
        if stt and not is_recording.locked():
            # Clear any pending triggers
            while not voice_queue.empty():
                try:
                    voice_queue.get_nowait()
                except queue.Empty:
                    break
            voice_queue.put("VOICE_TRIGGERED")
            print("\n🎤 Hotkey detectada! Gravando...", flush=True)

    # Setup hotkey listener (Ctrl+Shift+V)
    hotkey_listener = None
    if enable_stt:
        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse('<ctrl>+<shift>+m'),
            on_hotkey_pressed
        )

        def for_canonical(f):
            return lambda k: f(hotkey_listener.canonical(k))

        hotkey_listener = keyboard.Listener(
            on_press=for_canonical(hotkey.press),
            on_release=for_canonical(hotkey.release)
        )
        hotkey_listener.start()

    print("Game Agent initialized!")
    print("=" * 60)
    print("Welcome to Game Agent - AI-powered game analysis")
    print("🔊 Text-to-Speech: ENABLED")
    print("🎤 Speech-to-Text: ENABLED")
    print("⌨️  Hotkey: Pressione Ctrl+Shift+M para gravar sua voz (15 segundos)")
    print("=" * 60)
    print("\nAguardando hotkey...\n")

    # Interactive loop - only process voice from hotkey
    while True:
        try:
            # Wait for hotkey trigger
            trigger = voice_queue.get()  # Blocking - waits for hotkey

            if trigger == "VOICE_TRIGGERED" and stt:
                with is_recording:
                    # Record audio
                    try:
                        user_input = stt.listen_and_transcribe(duration=15)
                        print(f"📝 Você disse: {user_input}\n")
                    except Exception as e:
                        print(f"⚠️  Erro ao gravar/transcrever áudio: {e}")
                        print("\nAguardando próxima hotkey...\n")
                        continue

                    if not user_input or not user_input.strip():
                        print("Nenhum áudio detectado. Aguardando próxima hotkey...\n")
                        continue

                    # Process with agent
                    try:
                        print("\nProcessing your request...")
                        result = agent.run(user_input)

                        # Display the response
                        print("\n" + "=" * 60)
                        messages = result.get("messages", [])
                        if messages:
                            last_message = messages[-1]
                            response_text = last_message.content
                            print(f"\nAgent: {response_text}")

                            # Speak the response if TTS is enabled
                            if tts and isinstance(response_text, str):
                                print("\n🔊 Playing audio...")
                                try:
                                    tts.speak(response_text, play=True)
                                except Exception as e:
                                    print(f"⚠️  Error playing audio: {e}")
                        print("=" * 60)
                        print("\nAguardando próxima hotkey...\n")

                    except Exception as e:
                        print(f"⚠️  Erro ao processar com o agente: {e}")
                        print("Verifique se sua chave de API do OpenRouter está válida no arquivo .env")
                        print("\nAguardando próxima hotkey...\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            if hotkey_listener:
                hotkey_listener.stop()
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print("\nAguardando próxima hotkey...\n")


if __name__ == "__main__":
    main()
