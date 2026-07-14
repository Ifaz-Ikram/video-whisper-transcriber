import argparse
from pathlib import Path

import whisper


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
        help="Whisper model to use",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code such as en",
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

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    print(f"Transcribing: {video_path}")

    options = {}

    if args.language:
        options["language"] = args.language

    result = model.transcribe(
        str(video_path),
        **options,
    )

    transcript_path = output_directory / f"{video_path.stem}.txt"

    transcript_path.write_text(
        result["text"].strip(),
        encoding="utf-8",
    )

    print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    main()
