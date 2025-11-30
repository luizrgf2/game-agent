"""List available Edge TTS voices."""

import asyncio
from src.game_agent.tts import TextToSpeech


async def main():
    """List all available Brazilian Portuguese voices."""
    await TextToSpeech.list_voices()


if __name__ == "__main__":
    asyncio.run(main())
