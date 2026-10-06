"""Transcribe a video on a free CUDA GPU (Colab T4, Kaggle T4/P100).

Converts the file to a 16 kHz mono MP3 first, then runs faster-whisper large-v3.
"""

import argparse
import glob
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

from audio_extract import extract_mp3

# Same filler phrases the local transcriber drops during silence.
HALLUCINATED_PHRASES = {
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

VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4a", ".mp3", ".wav", ".flac"}


def require_gpu() -> str:
    if shutil.which("nvidia-smi") is None:
        raise SystemExit(
            "No GPU detected. Colab: Runtime → Change runtime type → T4 GPU. "
            "Kaggle: Settings → Accelerator → GPU, and enable Internet."
        )
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not name:
        raise SystemExit("nvidia-smi did not report a GPU.")
    return name.splitlines()[0].strip()


def ensure_cuda_libs() -> None:
    """Make libcublas loadable. Colab and Kaggle usually already have it."""
    import ctypes

    try:
        ctypes.CDLL("libcublas.so.12")
        return
    except OSError:
        pass

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
        ]
    )
    roots = []
    try:
        roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)
    lib_dirs = sorted(
        {
            os.path.dirname(path)
            for root in roots
            for path in glob.glob(f"{root}/nvidia/**/*.so*", recursive=True)
        }
    )
    if not lib_dirs:
        raise RuntimeError("Installed NVIDIA wheels, but libcublas was not found.")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, current] if current else lib_dirs)


def gpu_settings(gpu_name: str) -> tuple[str, int]:
    """Pick a compute type and batch size that fit a free GPU."""
    name = gpu_name.lower()
    if "p100" in name:
        return "int8", 4
    return "float16", 8


def default_output_path(source: Path) -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working") / f"{source.stem}.txt"
    output_directory = Path("output")
    if output_directory.exists() or not Path("/content").is_dir():
        output_directory.mkdir(exist_ok=True)
        return output_directory / f"{source.stem}.txt"
    return Path("/content") / f"{source.stem}.txt"


def transcribe(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    model_size: str = "large-v3",
    language: str | None = None,
    offset_seconds: int = 0,
    no_speech_threshold: float = 0.6,
) -> Path:
    import time

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(source)
    destination = Path(output_path) if output_path else default_output_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if model_size == "large":
        model_size = "large-v3"

    gpu_name = require_gpu()
    compute_type, batch_size = gpu_settings(gpu_name)
    print(f"GPU: {gpu_name} ({compute_type}, batch {batch_size})", flush=True)
    ensure_cuda_libs()

    print(f"Converting to MP3: {source}", flush=True)
    mp3_path = extract_mp3(source, offset_seconds)
    try:
        mp3_mb = mp3_path.stat().st_size / 1_048_576
        print(f"MP3 ready: {mp3_mb:.1f} MB", flush=True)

        started = time.perf_counter()
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model=model)
        kept = _transcribe_batches(
            pipeline,
            mp3_path,
            batch_size,
            language,
            no_speech_threshold,
        )
    finally:
        mp3_path.unlink(missing_ok=True)

    text = " ".join(kept).strip()
    destination.write_text(text + "\n", encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"Transcript saved to: {destination} ({elapsed:.0f}s)", flush=True)
    return destination


def _transcribe_batches(pipeline, mp3_path: Path, batch_size: int, language, no_speech_threshold: float) -> list[str]:
    last_error: RuntimeError | None = None
    sizes: list[int] = []
    for size in (batch_size, 4, 1):
        if size not in sizes:
            sizes.append(size)
    for size in sizes:
        try:
            segments, info = pipeline.transcribe(
                str(mp3_path),
                batch_size=size,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=no_speech_threshold,
                language=language,
            )
            print(
                f"Language: {info.language} ({info.language_probability:.2f})",
                flush=True,
            )
            kept: list[str] = []
            for index, segment in enumerate(segments):
                if index % 25 == 0:
                    print(f"segment {index} at {segment.start:.0f}s", flush=True)
                if segment.no_speech_prob >= no_speech_threshold:
                    continue
                text = segment.text.strip()
                if text.lower().rstrip(".,!? ") in HALLUCINATED_PHRASES:
                    continue
                kept.append(text)
            print(f"Transcribed {info.duration:.0f}s of audio", flush=True)
            return kept
        except RuntimeError as exc:
            if "memory" not in str(exc).lower():
                raise
            last_error = exc
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print(f"Batch size {size} ran out of GPU memory, retrying smaller.", flush=True)
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a video on a CUDA GPU.")
    parser.add_argument("video", type=Path, help="Video or audio file")
    parser.add_argument("--output", type=Path, default=None, help="Transcript path")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model (default: large-v3)")
    parser.add_argument("--language", default=None, help="Language code, e.g. en")
    parser.add_argument("--offset", type=int, default=0, help="Seconds to skip at the start")
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    args = parser.parse_args()
    transcribe(
        args.video,
        args.output,
        model_size=args.model,
        language=args.language,
        offset_seconds=args.offset,
        no_speech_threshold=args.no_speech_threshold,
    )


if __name__ == "__main__":
    main()
