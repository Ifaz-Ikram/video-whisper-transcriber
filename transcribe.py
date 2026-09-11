import argparse
import subprocess
import tempfile
from pathlib import Path

import whisper


def extract_audio_segment(video_path: Path, offset_seconds: int) -> Path:
    """Extract audio from video starting at offset_seconds into a temp WAV file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(offset_seconds),
        "-i", str(video_path),
        "-vn",                  # drop video
        "-acodec", "pcm_s16le",
        "-ar", "16000",         # Whisper expects 16 kHz
        "-ac", "1",             # mono
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return Path(tmp.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a video locally using Whisper."
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to the video file",
    )

    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model to use (default: small)",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. 'en'. Strongly recommended to avoid hallucinations.",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Skip this many seconds from the start before transcribing (default: 0). "
             "Use this to skip a silent intro.",
    )

    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Probability threshold above which a segment is treated as silence and "
             "skipped (default: 0.6). Raise towards 1.0 to be more aggressive about "
             "dropping silent chunks.",
    )

    args = parser.parse_args()

    video_path = args.video

    if not video_path.exists() and len(video_path.parts) == 1:
        input_video_path = Path("input") / video_path
        if input_video_path.exists():
            video_path = input_video_path

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {args.video}. "
            "Use a full path or place the file in input/."
        )

    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)

    # Extract audio segment if an offset was requested.
    tmp_audio: Path | None = None
    audio_source: str
    if args.offset > 0:
        print(f"Extracting audio from {args.offset}s offset …")
        tmp_audio = extract_audio_segment(video_path, args.offset)
        audio_source = str(tmp_audio)
    else:
        audio_source = str(video_path)

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    print(f"Transcribing: {video_path}" + (f" (offset {args.offset}s)" if args.offset else ""))

    transcribe_options: dict = {
        # Disabling conditioning on previous text prevents a hallucinated phrase
        # in one chunk from being "suggested" to the next chunk.
        "condition_on_previous_text": False,
        "no_speech_threshold": args.no_speech_threshold,
    }

    if args.language:
        transcribe_options["language"] = args.language

    result = model.transcribe(audio_source, **transcribe_options)

    # Clean up temp file if we created one.
    if tmp_audio is not None:
        tmp_audio.unlink(missing_ok=True)

    text = result["text"].strip()

    transcript_path = output_directory / f"{video_path.stem}.txt"
    transcript_path.write_text(text, encoding="utf-8")

    print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    main()
