import argparse
from pathlib import Path

import whisper

from audio_extract import extract_mp3


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

    # Phrases Whisper commonly hallucinates during silence.
    hallucinated_phrases = {
        "thank you",
        "thanks for watching",
        "thank you for watching",
        "thank you very much",
        "thanks",
        "bye",
        "bye bye",
        "you",
        ".",
        ",",
        "...",
    }

    transcribe_options: dict = {
        # Disabling conditioning on previous text prevents a hallucinated phrase
        # in one chunk from being "suggested" to the next chunk.
        "condition_on_previous_text": False,
        "no_speech_threshold": args.no_speech_threshold,
    }

    if args.language:
        transcribe_options["language"] = args.language

    print(f"Converting to MP3: {video_path}")
    tmp_audio = extract_mp3(video_path, args.offset)
    try:
        print(f"MP3 ready: {tmp_audio.stat().st_size / 1_048_576:.1f} MB")
        print(f"Loading Whisper model: {args.model}")
        model = whisper.load_model(args.model)
        print(
            f"Transcribing: {video_path}"
            + (f" (offset {args.offset}s)" if args.offset else "")
        )
        result = model.transcribe(str(tmp_audio), **transcribe_options)
    finally:
        tmp_audio.unlink(missing_ok=True)

    # Filter segments individually: skip high no-speech-probability segments
    # and known hallucinated filler phrases that Whisper emits during silence.
    kept_segments = []
    for seg in result.get("segments", []):
        if seg.get("no_speech_prob", 0.0) >= args.no_speech_threshold:
            continue
        seg_text = seg["text"].strip()
        if seg_text.lower().rstrip(".,!? ") in hallucinated_phrases:
            continue
        kept_segments.append(seg_text)

    text = " ".join(kept_segments).strip()

    transcript_path = output_directory / f"{video_path.stem}.txt"
    transcript_path.write_text(text, encoding="utf-8")

    print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    main()
