"""List available audio devices."""

from src.game_agent.stt import SpeechToText

if __name__ == "__main__":
    SpeechToText.list_audio_devices()
