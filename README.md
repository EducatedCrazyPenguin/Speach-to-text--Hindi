# Private Conversation Transcriber

A small Windows app for turning a conversation recording into searchable text. It records from the microphone or opens an existing audio/video file, runs speech recognition locally, and writes:

- a readable timestamped `.txt` transcript;
- `.srt` subtitles;
- structured `.json` data.
- a readable Markdown `.md` transcript.

The linked **Indic-TTS** repository is not used because TTS converts text *to* speech. The default engine is `faster-whisper`, which handles long recordings, silence detection, timestamps, automatic language recognition, and Hindi/English code-switching. A second engine uses Windows-compatible ONNX conversions of **AI4Bharat IndicConformer** for recordings that stay in one Indian language.

## Install on this PC

Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Then double-click **Run Voice to Text.bat**. To install and use the optional NVIDIA runtime automatically, double-click **Run Voice to Text (GPU).bat** instead; its first run includes a large download. Both launchers open a local browser page at `127.0.0.1`; that address is this PC, not an internet service. Use **Stop local app** when finished.

On the first transcription, the selected model downloads once. Later transcription is local. Approximate model choices are:

- **Small**: quickest download/test, lower Indic-language accuracy.
- **Medium**: recommended starting point.
- **Large v3**: best accuracy, largest download and slowest without GPU support.

System FFmpeg is not required; the installed audio decoder bundles the needed libraries.

## Use the app

1. Get the consent of everyone being recorded.
2. Click **Start microphone**, talk, then **Stop recording**; or choose an existing WAV, MP3, M4A, MP4, or similar file.
3. Use **AI4Bharat IndicConformer** for normal Hindi conversations; it is the app default on this PC. Switch to **Whisper** only for conversations with substantial Hindi/English code-switching or when language auto-detection is needed.
4. Put names and unusual words in the optional Whisper prompt.
5. Click **Transcribe**.

Results go to the `transcripts` folder by default. The app deletes its temporary copy of the recording when the job finishes unless **Keep the app's local copy** is enabled; an original file selected from elsewhere is never changed. For Hindi mixed with English, try Whisper with both Auto-detect and Hindi and keep the better transcript. IndicConformer does not support English or language auto-detection.

## Optional: label two speakers

The app can run the open-source pyannote `community-1` diarization model locally and constrain it to exactly two voices.

1. Run `.\setup-diarization.ps1 -Gpu` for an NVIDIA GPU (or omit `-Gpu` for CPU). This is a multi-gigabyte optional PyTorch download.
2. Sign in at [Hugging Face](https://huggingface.co/), accept the conditions for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), and create a read token.
3. In the app, enable **Identify exactly two voices**, enter the two names, and paste the token. Leave **Remember securely for this Windows account** enabled. After the first successful speaker-identification run, the token is stored in Windows Credential Manager and the field can remain empty on later runs. Use **Forget saved token** to remove it.

Speaker 1 means the first distinct voice detected, not necessarily the person who started the recording. If the labels are reversed, swap the two names and transcribe again. With IndicConformer, the app separates the two voices first and transcribes each exclusive speaker turn independently; this avoids mixing both voices in one ASR chunk. Diarization is imperfect when both people talk over each other or sit far from the microphone.

## Command line

```powershell
.\.venv\Scripts\python.exe .\transcribe.py "C:\path\conversation.m4a" --language hi --model medium
```

For the AI4Bharat backend, add `--engine indicconformer` and select a language. For two speakers, first set `$env:HF_TOKEN` and add `--diarize --speaker-names "Person 1,Person 2"`. Omit `--language hi` only when using Whisper auto-detection. Run `python transcribe.py --help` for every option.

## GPU and privacy notes

The app tries an NVIDIA GPU when the CUDA 12/cuDNN 9 runtime libraries are available and safely falls back to CPU in automatic mode. The NVIDIA driver alone does not install those runtime libraries. To add the large optional GPU runtime, close the app, run `.\setup-gpu.ps1`, and restart it with Device left on **auto**. The status line records which device was actually used.

Audio is not uploaded by the app. The initial model download comes from Hugging Face; Google TTS and other cloud APIs from VerbalLinguists are intentionally excluded.

## GitHub components evaluated

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) supplies efficient Whisper inference, bundled audio decoding and voice-activity detection.
- [onnx-asr](https://github.com/istupakov/onnx-asr) and the [OpenVoiceOS ONNX model registry](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr) make AI4Bharat IndicConformer practical on Python 3.12/Windows without NeMo.
- [WhisperX](https://github.com/m-bain/whisperX), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [aTrain](https://github.com/jstone09/atrain), and [Transcript Studio](https://github.com/Comput3rUs3r/AudioTranscript-Studio) informed the optional two-speaker workflow and speaker-aware exports.
- [lumiaspic/transcription](https://github.com/lumiaspic/transcription) demonstrates useful future additions: crash recovery, a persistent job queue and separate microphone/system tracks.

## Why not VerbalLinguists as-is?

The repository is an unfinished 2023 hackathon prototype rather than an installable application: the server sends hard-coded example text, the ASR script contains a placeholder where audio should be supplied, paths point to the original developer's computer, it only routes Hindi/Bengali/Gujarati, and its old fairseq/PyTorch pins do not support this Python 3.12 Windows setup. Its recorder/interface idea is good; this project implements that idea with a maintained backend.
