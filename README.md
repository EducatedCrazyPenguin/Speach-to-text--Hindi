# Private Conversation Transcriber

A small Windows app for turning a conversation recording into searchable text. It records from the microphone or opens an existing audio/video file, runs speech recognition locally, and writes:

- an untouched primary `.raw.txt`, authoritative `.verbatim.txt`, compatible `.transcript.txt`, and separate `.readable.txt` copy;
- `.srt` subtitles;
- structured `.json` data with word timings, confidence, overlap, provenance, and audio diagnostics.
- a readable Markdown `.md` transcript.

The linked **Indic-TTS** repository is not used because TTS converts text *to* speech. Maximum mode provisionally uses **Qwen3-ASR 1.7B** for Hindi/English code-mixing and dialect speech; the permanent corrected benchmark can promote SraVaani TDT, another measured candidate, or a two-model consensus. Faster-Whisper and the earlier AI4Bharat IndicConformer remain available in fast/compatibility mode.

## Install on this PC

Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

For this RTX 4080, double-click **Run Voice to Text (Maximum Accuracy GPU).bat**; the shorter **Run Voice to Text (GPU).bat** delegates to it. It installs the GPU/accuracy runtimes, verifies CUDA and cached speaker credentials, and resumably downloads Qwen3-ASR, Vaani Whisper and the isolated Qwen3.5 readable-copy runtime when needed. The first complete recovery/readable setup is about 15.5 GB. **Run Voice to Text.bat** remains the small legacy launcher.

On the first transcription, the selected model downloads once. Later transcription is local. Approximate model choices are:

- **Small**: quickest download/test, lower Indic-language accuracy.
- **Medium**: recommended starting point.
- **Large v3**: best accuracy, largest download and slowest without GPU support.

System FFmpeg is not required; the installed audio decoder bundles the needed libraries.

## Use the app

1. Get the consent of everyone being recorded.
2. Click **Start microphone**, talk, then **Stop recording**; or choose an existing WAV, MP3, M4A, MP4, or similar file.
3. Keep **Maximum local accuracy** selected. Qwen3-ASR is the provisional single-model default because it materially beat SraVaani on the supplied 47.664-second sample; the corrected permanent benchmark remains the authority. A second model is enabled only after that benchmark proves it helps.
4. Put names, places, family/religious terms and recurring number forms in the local vocabulary field.
5. Click **Transcribe**.

**Experimental evidence recovery** preserves the raw Qwen pass, detects suspect spans, retries shifted 6–18-second windows with Qwen and Vaani, acoustically aligns candidates, and marks unresolved wording `[uncertain]`. It is available but unchecked by default: on the corrected 2:10 Megha reference it made no accepted changes and did not improve the 46.3% baseline WER, so it failed the required two-point promotion gate. Full Vaani scored 52.1% WER on that reference and is therefore a retry source, not the default recognizer.

Results go to the `transcripts` folder by default. The app deletes its temporary copy of the recording when the job finishes unless **Keep the app's local copy** is enabled; an original file selected from elsewhere is never changed. For Hindi mixed with English, try Whisper with both Auto-detect and Hindi and keep the better transcript. IndicConformer does not support English or language auto-detection.

The correction editor can attach the matching original audio later, checks its duration, converts it to a private lossless WAV, and never uploads it. Mark evaluation calls **Held-out** so they are excluded from personal training. The supplied Megha correction is stored this way; `Speaker 1` is Mohit and `Speaker 0` is Wife.

## Optional: label two speakers

The app can run the open-source pyannote `community-1` diarization model locally and constrain it to exactly two voices.

1. Run `.\setup-diarization.ps1 -Gpu` for an NVIDIA GPU (or omit `-Gpu` for CPU). This is a multi-gigabyte optional PyTorch download.
2. Sign in at [Hugging Face](https://huggingface.co/), accept the conditions for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), and create a read token.
3. In the app, enable **Identify exactly two voices**, enter the two names, and paste the token. Leave **Remember securely for this Windows account** enabled. After the first successful speaker-identification run, the token is stored in Windows Credential Manager and the field can remain empty on later runs. Use **Forget saved token** to remove it.

Recognition now runs first on long VAD-aligned chunks (about 25 seconds, with overlap), preserving conversational context. A multilingual acoustic aligner then places Hindi/Hinglish words on the actual waveform before Community-1 assigns them to speakers. Pause-delimited utterances are rechecked against call-level voice embeddings; enrolled familiar-voice profiles also participate when their match is confident. Overlap is retained as `[overlap]` instead of being forced into one voice.

The token is stored once in Windows Credential Manager. Never paste it into chat or commit it to this folder. A token previously pasted into a chat should be revoked and replaced with a read-only token.

## Measured 10-minute accuracy benchmark

Open **Correct the 10-minute benchmark** in the app. It prepares exactly 600 seconds: the complete 47.664-second Mum sample plus five fixed windows from the 38:53 Wifey call. The editor supports looping, J/K navigation, Space play/pause, speaker editing, and autosave. This set is permanently training-excluded.

After all six clips are correct, **Run installed-model benchmark** measures WER, CER, names/numbers, speaker attribution, runtime, peak GPU memory, and original versus conservative telephone processing. It promotes consensus only for at least 1.5 absolute WER points of improvement within the 30-minute/40-minute-call budget. An audio variant needs at least one WER point without an entity regression.

Qwen's own forced aligner does not support Hindi, so maximum mode now uses TorchAudio MMS_FA plus Uroman for Hindi/Hinglish word alignment. On the supplied 47.664-second sample this aligned 119 of 121 words and raised the speaker-attribution proxy against the ElevenLabs intervals from about 81.4% to 93.3%. This is still below the 95% release gate and the ElevenLabs intervals are not hand-corrected ground truth, so the result is reported rather than claimed as solved. Zero-duration or out-of-bounds timings are rejected, and words near diarization boundaries remain counted as ambiguous in JSON.

The public Srota Hindi/Hinglish 0.6B candidate was also tested on the supplied sample. Its proxy WER was 77.4%, worse than Qwen's 55.6%, so it remains evaluation-only. On the user-corrected 2:10 Megha reference, Qwen measured 46.3% WER/30.8% CER and Vaani measured 52.1% WER/40.8% CER. Encrypted Mohit/Wife profiles matched their call clusters at cosine 0.975/0.952 and produced 93.8% scored speaker attribution—much better, but still below the 95% gate. These results are reported rather than promoted, and the short approximately-95%-correct reference is not treated as proof of the global target.

## Personal adaptation

Keep the app audio copy for consented calls, use **Correct for personal adaptation** after transcription, and tag dialect, audio condition, speakers, and overlap. Once three verified hours across at least three calls are available, follow [training/README.md](training/README.md). The preparation script splits whole calls 80/10/10, never includes the permanent benchmark, and creates the tarred 16 kHz mono shards required by the official SraVaani recipe. WSL is used for training only; accepted checkpoints export back to ONNX for normal Windows inference.

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
