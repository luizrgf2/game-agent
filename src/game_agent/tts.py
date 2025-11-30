"""Text-to-Speech using Edge TTS."""

import asyncio
import os
import tempfile
from pathlib import Path

import edge_tts


class TextToSpeech:
    """Text-to-Speech handler using Edge TTS."""

    def __init__(self, voice: str = "pt-BR-FranciscaNeural"):
        """Initialize TTS with a voice.

        Args:
            voice: Voice to use. Default is Brazilian Portuguese female voice.
                   Other options:
                   - pt-BR-FranciscaNeural (Female, natural)
                   - pt-BR-AntonioNeural (Male)
                   - pt-BR-BrendaNeural (Female)
        """
        self.voice = voice
        self.audio_dir = Path("audio_output")
        self.audio_dir.mkdir(exist_ok=True)

    async def speak_async(self, text: str, play: bool = True) -> str:
        """Convert text to speech asynchronously.

        Args:
            text: Text to convert to speech
            play: Whether to play the audio immediately

        Returns:
            Path to the generated audio file
        """
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", dir=self.audio_dir
        )
        output_file = temp_file.name
        temp_file.close()

        # Generate speech
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

        # Play audio if requested
        if play:
            await self._play_audio(output_file)

        return output_file

    def speak(self, text: str, play: bool = True) -> str:
        """Convert text to speech synchronously.

        Args:
            text: Text to convert to speech
            play: Whether to play the audio immediately

        Returns:
            Path to the generated audio file
        """
        return asyncio.run(self.speak_async(text, play))

    async def _play_audio(self, file_path: str):
        """Play audio file using system player.

        Args:
            file_path: Path to audio file
        """
        # Try different players based on OS
        if os.name == "posix":  # Linux/Mac
            # Try different players with arguments to avoid opening a window
            players = {
                "mpg123": ["-q"],
                "mpv": ["--no-video", "--no-audio-display"],
                "ffplay": ["-autoexit", "-nodisp"],
            }
            for player, args in players.items():
                try:
                    process = await asyncio.create_subprocess_exec(
                        player,
                        *args,
                        file_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await process.wait()
                    return
                except FileNotFoundError:
                    continue

            print(
                f"Áudio gerado em: {file_path}\n"
                "Instale mpg123, mpv ou ffplay para reprodução automática."
            )
        else:  # Windows
            try:
                process = await asyncio.create_subprocess_exec(
                    "powershell",
                    "-c",
                    f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await process.wait()
            except Exception:
                print(f"Áudio gerado em: {file_path}")

    @staticmethod
    async def list_voices():
        """List all available voices."""
        voices = await edge_tts.list_voices()
        # Filter Brazilian Portuguese voices
        pt_br_voices = [v for v in voices if v["Locale"].startswith("pt-BR")]

        print("\nVozes disponíveis em Português Brasileiro:")
        print("=" * 60)
        for voice in pt_br_voices:
            print(f"- {voice['ShortName']}")
            print(f"  Gênero: {voice['Gender']}")
            print(f"  Nome: {voice['FriendlyName']}")
            print()

        return pt_br_voices
