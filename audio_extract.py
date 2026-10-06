"""Convert a video or audio file to a temporary 16 kHz mono MP3."""

import os
import subprocess
import tempfile
from pathlib import Path


def extract_mp3(source: Path, offset_seconds: int = 0) -> Path:
    """Extract speech audio to a temporary MP3. Caller deletes the file."""
    if not source.exists():
        raise FileNotFoundError(source)

    fd, raw_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    destination = Path(raw_path)
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if offset_seconds > 0:
        command.extend(["-ss", str(offset_seconds)])
    command.extend(
        [
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(destination),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return destination
