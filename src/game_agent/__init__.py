"""Game Agent - AI-powered game analysis using screenshots and vision."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import GameAgent
from .tts import TextToSpeech


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
    ptt_mode = os.getenv("PTT_MODE", "keyboard").lower()

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
        except Exception as e:
            print(f"⚠️  Erro ao inicializar STT: {e}")
            import traceback
            traceback.print_exc()
            enable_stt = False

    if not enable_stt or stt is None:
        print("STT não está habilitado. Encerrando...")
        sys.exit(1)

    print("\nGame Agent initialized!")
    print("=" * 60)
    print("Welcome to Game Agent - AI-powered game analysis")
    print("🔊 Text-to-Speech: ENABLED" if enable_tts else "🔇 Text-to-Speech: DISABLED")
    print("🎤 Speech-to-Text: ENABLED (Push-to-Talk)")

    if ptt_mode == "xbox":
        print("🎮 Modo: Controle Xbox")
        xbox_button = os.getenv("XBOX_BUTTON", "RB")
        print(f"    Pressione e SEGURE o botão '{xbox_button}' para gravar")
        print(f"    Solte o botão quando terminar de falar")
    else:
        print("⌨️  Modo: Teclado")
        ptt_key = os.getenv("PTT_KEY", "m").upper()
        print(f"    Pressione e SEGURE '{ptt_key}' para gravar sua voz")
        print(f"    Solte '{ptt_key}' quando terminar de falar")

    print("=" * 60)
    print(f"\nAguardando comando...\n")

    # Interactive loop - push-to-talk
    while True:
        try:
            # Use STT to listen and transcribe
            user_input = stt.listen_and_transcribe()

            if not user_input or not user_input.strip():
                print("\nNenhum texto detectado")
                print(f"\nAguardando comando...\n")
                continue

            print(f"📝 Você disse: {user_input}\n")

            # Process with agent
            try:
                print("Processando seu pedido...")
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
                print(f"\nAguardando comando...\n")

            except Exception as e:
                print(f"⚠️  Erro ao processar: {e}")
                print("Verifique sua chave de API no .env")
                print(f"\nAguardando comando...\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            # Cleanup recorder resources
            if hasattr(stt.recorder, 'cleanup'):
                stt.recorder.cleanup()
            break
        except Exception as e:
            print(f"\nErro: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nAguardando comando...\n")


if __name__ == "__main__":
    main()
