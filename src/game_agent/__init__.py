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

    # Check if TTS should be enabled
    enable_tts = os.getenv("ENABLE_TTS", "true").lower() == "true"

    # Create the agent
    agent = GameAgent(api_key=api_key)

    # Create TTS if enabled
    tts = TextToSpeech() if enable_tts else None

    print("Game Agent initialized!")
    print("=" * 60)
    print("Welcome to Game Agent - AI-powered game analysis")
    if enable_tts:
        print("🔊 Text-to-Speech: ENABLED")
    print("=" * 60)
    print("\nCommands:")
    print("  - 'analyze this game screen' - Takes screenshot and analyzes it")
    print("  - 'read and interpret this text' - Takes screenshot and reads text")
    print("  - 'tts on/off' - Enable/disable text-to-speech")
    print("  - 'quit' or 'exit' - Exit the program")
    print("\n" + "=" * 60 + "\n")

    # Interactive loop
    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            # Handle TTS commands
            if user_input.lower() == "tts on":
                tts = TextToSpeech()
                print("🔊 Text-to-Speech ENABLED")
                continue
            elif user_input.lower() == "tts off":
                tts = None
                print("🔇 Text-to-Speech DISABLED")
                continue

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

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
