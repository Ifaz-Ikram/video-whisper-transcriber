# Video Whisper Transcriber

A local video transcription tool that converts the audio from video files into text using OpenAI's open-source Whisper model.

The transcription runs locally on your computer. It does not require an OpenAI API key.

## How It Works

```text
Video file
    ↓
FFmpeg reads and extracts the audio
    ↓
Whisper processes the audio locally
    ↓
Text transcript is saved in output/
```

## Features

- Transcribes MP4, MKV, MOV, AVI and other FFmpeg-supported files
- Runs locally without an OpenAI API key
- Automatically detects the spoken language
- Supports different Whisper model sizes
- Saves transcripts as plain-text files
- Keeps input videos and generated transcripts out of Git

## Project Structure

```text
video-whisper-transcriber/
├── .venv/
├── input/
│   ├── .gitkeep
│   └── your-video.mp4
├── output/
│   └── .gitkeep
├── .gitignore
├── README.md
├── requirements.txt
└── transcribe.py
```

## Requirements

You need:

- Python
- pip
- FFmpeg
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/video-whisper-transcriber.git
cd video-whisper-transcriber
```

When working with the existing local project, simply open the project directory:

```bash
cd video-whisper-transcriber
```

## 2. Install System Dependencies

### Arch Linux

```bash
sudo pacman -Syu
sudo pacman -S --needed python python-pip ffmpeg git
```

Check that FFmpeg is installed:

```bash
ffmpeg -version
```

Check the Python version:

```bash
python --version
```

## 3. Remove Unnecessary API Files

This project runs Whisper locally, so `.env` files and an OpenAI API key are not needed.

```bash
rm -f .env .env.example
```

## 4. Create the Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, the terminal should show something similar to:

```text
(.venv)
```

To deactivate it later:

```bash
deactivate
```

## 5. Create the Requirements File

```bash
cat > requirements.txt <<'REQUIREMENTS'
openai-whisper
REQUIREMENTS
```

## 6. Install Whisper

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Verify that Whisper was installed:

```bash
python -c "import whisper; print('Whisper installed successfully')"
```

You can also check the Whisper command:

```bash
whisper --help
```

## 7. Create the Project Folders

```bash
mkdir -p input output
touch input/.gitkeep output/.gitkeep
```

## 8. Configure `.gitignore`

```bash
cat > .gitignore <<'GITIGNORE'
.venv/
__pycache__/
*.pyc

input/*
output/*

!input/.gitkeep
!output/.gitkeep
GITIGNORE
```

This prevents videos and generated transcripts from being uploaded to GitHub.

## 9. Add a Video

Copy or move a video into the `input` folder.

Example:

```bash
cp "/path/to/your/video.mp4" input/
```

For example:

```bash
cp "$HOME/Videos/Business Lecture 2026-07-13.mp4" input/
```

List the available videos:

```bash
ls -lh input/
```

## 10. Run the Transcriber

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the program using the default `small` model:

```bash
python transcribe.py "input/Business Lecture 2026-07-13.mp4"
```

The quotation marks are important when the filename contains spaces.

The transcript will be saved as:

```text
output/Business Lecture 2026-07-13.txt
```

## Whisper Model Options

### Tiny Model

Fastest, but provides lower accuracy:

```bash
python transcribe.py "input/your-video.mp4" --model tiny
```

### Base Model

Good for quick testing:

```bash
python transcribe.py "input/your-video.mp4" --model base
```

### Small Model

Recommended starting point:

```bash
python transcribe.py "input/your-video.mp4" --model small
```

### Medium Model

Better accuracy but slower:

```bash
python transcribe.py "input/your-video.mp4" --model medium
```

### Large Model

Higher accuracy but requires more memory and processing time:

```bash
python transcribe.py "input/your-video.mp4" --model large
```

### Turbo Model

Fast transcription with strong accuracy:

```bash
python transcribe.py "input/your-video.mp4" --model turbo
```

## Language Selection

Whisper automatically detects the spoken language when no language is provided:

```bash
python transcribe.py "input/your-video.mp4" --model small
```

For English:

```bash
python transcribe.py "input/your-video.mp4" --model small --language en
```

For Sinhala:

```bash
python transcribe.py "input/your-video.mp4" --model small --language si
```

For Tamil:

```bash
python transcribe.py "input/your-video.mp4" --model small --language ta
```

For videos containing more than one language, leave out the `--language` option:

```bash
python transcribe.py "input/your-video.mp4" --model medium
```

## Transcribe the Current Videos

### Business Lecture

```bash
python transcribe.py "input/Business Lecture 2026-07-13.mp4" --model small
```

### Calculus Recording

First, check the exact filename:

```bash
ls input/
```

Then run:

```bash
python transcribe.py "input/Calculus from 8.33 a.m. 2026-07-10.mp4" --model small
```

### Operational Research Recording

```bash
python transcribe.py "input/Operational Research 2026-07-07.mp4" --model small
```

Use the exact filenames shown by:

```bash
find input -maxdepth 1 -type f
```

`transcribe.py` and `modal_transcribe.py` convert the video to a temporary 16 kHz mono MP3 before transcription, then delete that MP3.

## Free GPU: Colab and Kaggle

Long recordings are slow on CPU. These notebooks run faster-whisper `large-v3` on the free T4 (or a Kaggle P100).

### Google Colab

1. Upload `notebooks/colab_transcribe.ipynb` to [Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → T4 GPU.
3. For a long video, put it in Google Drive and set `VIDEO` to that path. Leave `VIDEO` empty to upload a smaller file.
4. Run all cells. The transcript downloads when the run finishes.

### Kaggle

1. Upload `notebooks/kaggle_transcribe.ipynb` as a notebook.
2. Settings → Accelerator → GPU.
3. Settings → Internet → On.
4. Add the video as a dataset input. Set `VIDEO` if more than one media file is attached.
5. Run all cells. The transcript is written to `/kaggle/working`.

## Optional: Extract Audio Manually

The transcriber already converts to a temporary MP3. To make that file yourself:

```bash
ffmpeg \
  -i "input/your-video.mp4" \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a libmp3lame \
  -b:a 64k \
  "output/your-video.mp3"
```

Then transcribe the MP3:

```bash
python transcribe.py "output/your-video.mp3" --model small
```

## Run Whisper Directly

You can also use the Whisper command without `transcribe.py`:

```bash
whisper "input/your-video.mp4" \
  --model small \
  --output_dir output \
  --output_format txt
```

For English:

```bash
whisper "input/your-video.mp4" \
  --model small \
  --language en \
  --output_dir output \
  --output_format txt
```

Generate TXT, SRT, VTT, TSV and JSON files:

```bash
whisper "input/your-video.mp4" \
  --model small \
  --output_dir output \
  --output_format all
```

## Subtitle Generation

Generate an SRT subtitle file:

```bash
whisper "input/your-video.mp4" \
  --model small \
  --output_dir output \
  --output_format srt
```

The generated file will appear as:

```text
output/your-video.srt
```

Generate a WebVTT subtitle file:

```bash
whisper "input/your-video.mp4" \
  --model small \
  --output_dir output \
  --output_format vtt
```

## Check Generated Files

```bash
ls -lh output/
```

Read a transcript in the terminal:

```bash
cat "output/your-video.txt"
```

Open the output folder in VS Code:

```bash
code output/
```

## First Model Download

The first time a model is used, Whisper downloads its model file.

For example:

```bash
python transcribe.py "input/your-video.mp4" --model small
```

Later runs using the same model use the downloaded local copy.

## Troubleshooting

### Virtual Environment Is Not Active

Activate it:

```bash
source .venv/bin/activate
```

### `ModuleNotFoundError: No module named 'whisper'`

Install the requirements inside the virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `ffmpeg: command not found`

Install FFmpeg:

```bash
sudo pacman -S ffmpeg
```

Then verify it:

```bash
ffmpeg -version
```

### Video File Not Found

Check the filename:

```bash
ls -lh input/
```

Always use quotation marks for filenames containing spaces:

```bash
python transcribe.py "input/video file.mp4"
```

### Transcription Is Too Slow

Use a smaller model:

```bash
python transcribe.py "input/your-video.mp4" --model base
```

For an even faster test:

```bash
python transcribe.py "input/your-video.mp4" --model tiny
```

### Transcription Accuracy Is Low

Try a larger model:

```bash
python transcribe.py "input/your-video.mp4" --model medium
```

You can also provide the language:

```bash
python transcribe.py "input/your-video.mp4" \
  --model medium \
  --language en
```

### Check Available Disk Space

```bash
df -h
```

### Check Git Status

```bash
git status
```

Videos inside `input/` and transcripts inside `output/` should not appear as files waiting to be committed.

## Git Commands

Check the repository:

```bash
git status
```

Add the project files:

```bash
git add .
```

Create the first commit:

```bash
git commit -m "Create local video transcription pipeline"
```

Rename the branch to `main`:

```bash
git branch -M main
```

Connect the GitHub repository:

```bash
git remote add origin https://github.com/YOUR-USERNAME/video-whisper-transcriber.git
```

Push the project:

```bash
git push -u origin main
```

For later changes:

```bash
git add .
git commit -m "Update video transcription project"
git push
```

## Complete Setup Commands

For a new local setup on Arch Linux:

```bash
sudo pacman -S --needed python python-pip ffmpeg git

git clone https://github.com/YOUR-USERNAME/video-whisper-transcriber.git
cd video-whisper-transcriber

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p input output
touch input/.gitkeep output/.gitkeep

python transcribe.py "input/your-video.mp4" --model small
```

## Privacy and Cost

The video and audio are processed locally.

- No OpenAI API key is needed
- No OpenAI API request is made
- Videos are not uploaded by this project
- Processing speed depends on your computer
- Model files require internet access when downloaded for the first time

## Official Whisper Repository

https://github.com/openai/whisper
