"""Transcribe audio on a Modal GPU with faster-whisper large-v3.

The local CPU path falls back to FP32 and is far too slow for a long recording.
This sends compressed audio to whichever of L4, A10, or T4 is free first.
"""

from pathlib import Path

import modal

app = modal.App("whisper-transcribe")

MODEL_DIR = "/models/large-v3"
MODEL_REPO = "Systran/faster-whisper-large-v3"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper",
        "huggingface_hub",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    )
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_REPO}', local_dir='{MODEL_DIR}')\""
    )
)

# Pip CUDA wheels keep libcublas outside the default loader path.
CUDA_LIB_PATH = ":".join(
    [
        "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib",
        "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
        "/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib",
    ]
)

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


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    timeout=60 * 60,
    startup_timeout=60 * 10,
    env={"LD_LIBRARY_PATH": CUDA_LIB_PATH},
)
def transcribe(audio: bytes, no_speech_threshold: float = 0.6) -> dict:
    import os
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    print(f"GPU: {gpu}", flush=True)

    fd, raw_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    audio_path = Path(raw_path)
    audio_path.write_bytes(audio)

    started = time.perf_counter()
    model = WhisperModel(
        MODEL_DIR,
        device="cuda",
        compute_type="float16",
        local_files_only=True,
    )
    pipeline = BatchedInferencePipeline(model=model)
    segments, info = pipeline.transcribe(
        str(audio_path),
        batch_size=16,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        no_speech_threshold=no_speech_threshold,
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

    elapsed = time.perf_counter() - started
    print(f"Transcribed {info.duration:.0f}s of audio in {elapsed:.1f}s", flush=True)
    audio_path.unlink(missing_ok=True)
    return {
        "text": " ".join(kept).strip(),
        "language": info.language,
        "duration": info.duration,
        "gpu": gpu,
        "elapsed": elapsed,
    }


@app.local_entrypoint()
def main(video_path: str, output_path: str = "", no_speech_threshold: float = 0.6) -> None:
    from audio_extract import extract_mp3

    source = Path(video_path)
    if not output_path:
        output_path = str(Path("output") / f"{source.stem}.txt")
    print(f"Converting to MP3: {source}")
    mp3_path = extract_mp3(source)
    try:
        audio = mp3_path.read_bytes()
    finally:
        mp3_path.unlink(missing_ok=True)
    print(f"Uploading {len(audio) / 1_048_576:.1f} MB MP3 to Modal")
    result = transcribe.remote(audio, no_speech_threshold)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result["text"] + "\n", encoding="utf-8")
    print(
        f"Saved {destination} "
        f"({result['language']}, {result['duration']:.0f}s audio, "
        f"{result['elapsed']:.0f}s on {result['gpu']})"
    )
