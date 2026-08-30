from __future__ import annotations

import os
import socket
import sys
import threading
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any


# pythonw intentionally starts without console streams. A few ML libraries still
# assume stderr exists even when their progress bars are disabled.
_NULL_STREAMS = []
for _stream_name in ("stdout", "stderr"):
    if getattr(sys, _stream_name) is None:
        _stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, _stream_name, _stream)
        _NULL_STREAMS.append(_stream)

from flask import Flask, Response, jsonify, request, send_from_directory
from huggingface_hub import HfApi
from werkzeug.serving import make_server
from werkzeug.utils import secure_filename

from voice_to_text.core import ENGINE_CHOICES, LANGUAGES, MODEL_CHOICES, Transcriber
from voice_to_text.accuracy import (
    AccuracyPipeline,
    AccuracySettings,
    promoted_defaults,
    recommended_candidates,
)
from voice_to_text.benchmark import (
    benchmark_root,
    load_manifest,
    prepare_benchmark,
    save_gold_segments,
)
from voice_to_text.diarization import (
    diarize_then_transcribe_two_speakers,
    diarize_two_speakers,
    speaker_model_is_cached,
)
from voice_to_text.exports import write_outputs
from voice_to_text.exports import result_from_dict
from voice_to_text.models import CANDIDATE_LABELS, optional_model_availability
from voice_to_text.profiles import delete_profile, enroll_profile, list_profiles
from voice_to_text.benchmark_ui import BENCHMARK_HTML
from voice_to_text.correction_ui import CORRECTION_HTML
from voice_to_text.evaluation import run_benchmark
from voice_to_text.secrets import forget_token, has_token, load_token, save_token


ROOT_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
RECORDINGS_DIR = ROOT_DIR / "recordings"
JOBS_DIR = ROOT_DIR / ".jobs"
CORRECTIONS_DIR = ROOT_DIR / "corrections"
HOST = "127.0.0.1"
PORT = 8765

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
TRANSCRIBERS: dict[tuple[str, str, str], Transcriber] = {}
TRANSCRIBE_LOCK = threading.Lock()
ACTIVE_JOB_ID: str | None = None
SERVER = None
PYANNOTE_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
LOCAL_ORIGINS = {f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Private Conversation Transcriber</title>
  <style>
    :root { color-scheme: light; font-family: Inter, "Segoe UI", system-ui, sans-serif; }
    body { margin: 0; background: #f2f5f4; color: #17201d; }
    main { width: min(980px, calc(100% - 32px)); margin: 28px auto; }
    header { background: linear-gradient(135deg, #123c34, #27695b); color: white; padding: 28px; border-radius: 18px; }
    h1 { margin: 0 0 8px; font-size: clamp(1.7rem, 4vw, 2.5rem); }
    header p { margin: 0; color: #d9eee8; }
    .card { background: white; border: 1px solid #dce5e2; border-radius: 16px; padding: 20px; margin-top: 16px; box-shadow: 0 7px 24px #163c3210; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .wide { grid-column: 1 / -1; }
    label { display: block; font-size: .88rem; font-weight: 650; margin-bottom: 6px; }
    input, select, textarea, button { box-sizing: border-box; width: 100%; min-height: 42px; border: 1px solid #bdcbc7; border-radius: 9px; padding: 9px 11px; font: inherit; }
    button { cursor: pointer; border: 0; color: white; background: #196b59; font-weight: 700; }
    button.secondary { background: #e7efed; color: #24443d; }
    button.danger { background: #8a3333; }
    button:disabled { opacity: .55; cursor: wait; }
    .actions { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }
    .notice { padding: 12px 14px; background: #fff7dc; border-left: 4px solid #d19c18; border-radius: 7px; margin-top: 14px; }
    #speakerOptions { display: none; }
    #status { margin: 14px 0 6px; font-weight: 650; }
    progress { width: 100%; height: 16px; accent-color: #196b59; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f8f7; padding: 16px; border-radius: 10px; min-height: 130px; max-height: 420px; overflow: auto; }
    #downloads { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    #downloads a { color: #145d4e; background: #e4f2ee; border-radius: 7px; padding: 7px 10px; text-decoration: none; font-weight: 650; }
    .help { margin: 0; color: #53635e; font-size: .86rem; }
    .help a { color: #145d4e; }
    audio { width: 100%; margin-top: 10px; }
    nav { margin-top: 12px; display:flex; gap:12px; flex-wrap:wrap; }
    nav a { color:#e7fff8; font-weight:700; }
    details { grid-column:1/-1; }
    @media (max-width: 700px) { .grid, .actions { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Private Conversation Transcriber</h1>
    <p>Maximum-accuracy Hindi and dialect speech to timestamped text—processed on this PC.</p>
    <nav><a href="/benchmark">Correct the 10-minute benchmark</a><a href="#profiles">Enroll familiar voices</a></nav>
  </header>

  <section class="card">
    <div class="notice"><strong>Consent first:</strong> record only when everyone in the conversation has agreed.</div>
    <div class="grid" style="margin-top:16px">
      <div class="wide">
        <label for="audioFile">Conversation recording</label>
        <input id="audioFile" type="file" accept="audio/*,video/*">
        <audio id="preview" controls hidden></audio>
      </div>
      <div class="wide"><label><input id="keepAudio" type="checkbox" style="width:auto;min-height:auto"> Keep the app's local copy after transcription</label></div>
    </div>
    <div class="actions">
      <button id="record">Start microphone</button>
      <button id="stop" class="secondary" disabled>Stop recording</button>
      <button id="transcribe">Transcribe</button>
      <button id="shutdown" class="danger">Stop local app</button>
    </div>
  </section>

  <section class="card">
    <div class="grid">
      <div><label for="preset">Accuracy preset</label><select id="preset"><option value="maximum" selected>Maximum local accuracy</option><option value="fast">Fast / legacy</option></select></div>
      <div><label for="audioMode">Audio processing</label><select id="audioMode"><option value="original" selected>Original (safest default)</option><option value="telephone">Conservative telephone preset</option><option value="auto">Automatic, including mild noise reduction</option></select></div>
      <div><label for="primaryCandidate">Primary accuracy model</label><select id="primaryCandidate"></select></div>
      <div><label for="secondaryCandidate">Second model</label><select id="secondaryCandidate"><option value="">None until benchmark promotes it</option></select></div>
      <div class="wide"><label><input id="consensus" type="checkbox" style="width:auto;min-height:auto"> Use two-model consensus (enable only after benchmark improvement)</label></div>
      <div class="wide"><label><input id="readable" type="checkbox" checked style="width:auto;min-height:auto"> Also create a separate readable standard-Hindi copy</label></div>
      <details><summary>Fast/legacy and advanced controls</summary><div class="grid" style="margin-top:12px">
      <div><label for="engine">Legacy engine</label><select id="engine"></select></div>
      <div><label for="language">Conversation language</label><select id="language"></select></div>
      <div><label for="model">Whisper accuracy</label><select id="model"></select></div>
      <div><label for="device">Device</label><select id="device"><option>auto</option><option>cuda</option><option>cpu</option></select></div>
      </div></details>
      <div class="wide"><label for="prompt">Local vocabulary: names, places, family/religious terms and numbers</label><textarea id="prompt" rows="2" placeholder="Mohit, family names, place names, परिक्रमा, निस्सो रंग"></textarea></div>
      <p class="wide help">For your Hindi/English conversations, keep the recommended language selected. Auto-detect is less reliable for very short recordings.</p>
      <div class="wide"><label><input id="diarize" type="checkbox" checked style="width:auto;min-height:auto"> Identify exactly two voices after transcription</label></div>
      <div id="speakerOptions" class="wide grid">
        <div><label for="speaker1">First distinct voice</label><input id="speaker1" value="Speaker 1"></div>
        <div><label for="speaker2">Second distinct voice</label><input id="speaker2" value="Speaker 2"></div>
        <div class="wide"><label for="token">Hugging Face read token</label><input id="token" type="password"></div>
        <div><label><input id="rememberToken" type="checkbox" checked style="width:auto;min-height:auto"> Remember securely for this Windows account</label></div>
        <div><button id="forgetToken" type="button" class="secondary">Forget saved token</button></div>
        <div id="savedTokenStatus" class="wide help"></div>
        <p class="wide help">Before using a token, sign in to the same Hugging Face account, <a href="https://huggingface.co/pyannote/speaker-diarization-community-1" target="_blank" rel="noreferrer">accept the Community-1 access conditions</a>, then <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">create a read token</a>.</p>
      </div>
    </div>
  </section>

  <section class="card" id="profiles">
    <h2>Familiar voice profiles</h2>
    <p class="help">Upload at least 10 seconds of clean speech by one person. Only an encrypted voice embedding is kept; the enrollment audio is deleted.</p>
    <div class="grid" style="margin-top:12px">
      <div><label for="profileName">Person</label><input id="profileName" placeholder="Wife or Mum"></div>
      <div><label for="profileAudio">Clean voice sample</label><input id="profileAudio" type="file" accept="audio/*,video/*"></div>
      <div><button id="enrollProfile" type="button" class="secondary">Enroll voice</button></div>
      <div><label for="profileList">Stored profiles</label><select id="profileList"></select></div>
      <div><button id="deleteProfile" type="button" class="danger">Delete selected profile</button></div>
      <div id="profileStatus" class="help"></div>
    </div>
  </section>

  <section class="card">
    <div id="status">Ready. Choose a file or record from the microphone.</div>
    <progress id="progress" max="100" value="0"></progress>
    <div id="downloads"></div>
    <pre id="transcript">Your transcript will appear here.</pre>
  </section>
</main>
<script>
const config = __CONFIG__;
const $ = id => document.getElementById(id);
function fillSelect(id, entries, selected) {
  const select = $(id);
  for (const [label, value] of entries) {
    const option = document.createElement('option'); option.textContent = label; option.value = value;
    if (value === selected) option.selected = true; select.appendChild(option);
  }
}
fillSelect('engine', config.engines, 'indicconformer');
fillSelect('language', config.languages, 'hi');
fillSelect('model', config.models, 'medium');
fillSelect('primaryCandidate', config.candidates, config.recommended_candidates[0] || 'sravaani');
for (const [label, value, available] of config.candidates) {
  const option = document.createElement('option'); option.textContent = label + (available ? '' : ' — run setup-accuracy.ps1');
  option.value = value; option.disabled = !available; $('secondaryCandidate').appendChild(option);
}
if (config.recommended_candidates.length > 1) $('secondaryCandidate').value = config.recommended_candidates[1];
$('audioMode').value = config.promoted_defaults.audio_mode || 'original';
$('consensus').checked = !!config.promoted_defaults.use_consensus;

let recordedBlob = null, recorder = null, chunks = [], polling = null;
let selectedDuration = null;
let hasSavedToken = config.has_saved_token;
const speakerModelCached = config.speaker_model_cached;
function updateTokenStatus() {
  $('savedTokenStatus').textContent = hasSavedToken ? 'A token is saved securely. Leave the token field empty to reuse it.' : (speakerModelCached ? 'The speaker model is cached locally. No token is needed now.' : 'No token is currently saved.');
  $('token').placeholder = (hasSavedToken || speakerModelCached) ? 'Leave empty to use the local speaker model' : 'Paste a new read token';
}
updateTokenStatus();
$('speakerOptions').style.display = $('diarize').checked ? 'grid' : 'none';
$('diarize').addEventListener('change', () => $('speakerOptions').style.display = $('diarize').checked ? 'grid' : 'none');
$('forgetToken').addEventListener('click', async () => {
  const response = await fetch('/api/token/forget', {method:'POST'});
  if (response.ok) { hasSavedToken = false; $('token').value = ''; updateTokenStatus(); }
});
$('audioFile').addEventListener('change', () => {
  recordedBlob = null;
  const file = $('audioFile').files[0];
  selectedDuration = null;
  if (file) {
    $('preview').src = URL.createObjectURL(file); $('preview').hidden = false;
    $('preview').onloadedmetadata = () => {
      selectedDuration = Number.isFinite($('preview').duration) ? $('preview').duration : null;
      const duration = selectedDuration === null ? '' : ` — ${Math.floor(selectedDuration / 60)}m ${Math.round(selectedDuration % 60)}s`;
      $('status').textContent = `Selected: ${file.name}${duration}`;
    };
  }
});

$('record').addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    chunks = []; recorder = new MediaRecorder(stream);
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      recordedBlob = new Blob(chunks, {type: recorder.mimeType || 'audio/webm'});
      $('preview').src = URL.createObjectURL(recordedBlob); $('preview').hidden = false;
      stream.getTracks().forEach(track => track.stop());
      $('status').textContent = 'Recording captured. Click Transcribe.';
    };
    recorder.start(1000); $('record').disabled = true; $('stop').disabled = false;
    $('status').textContent = 'Recording…';
  } catch (error) { $('status').textContent = 'Microphone error: ' + error.message; }
});
$('stop').addEventListener('click', () => {
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  $('record').disabled = false; $('stop').disabled = true;
});

function setWorking(working) { $('transcribe').disabled = working; $('record').disabled = working; }
function pollJob(id) {
  clearInterval(polling);
  polling = setInterval(async () => {
    const response = await fetch('/api/jobs/' + id); const job = await response.json();
    $('status').textContent = job.message || job.status; $('progress').value = Math.round((job.progress || 0) * 100);
    if (job.status === 'done') {
      clearInterval(polling); setWorking(false); $('transcript').textContent = job.transcript;
      if (job.has_saved_token !== undefined) { hasSavedToken = job.has_saved_token; $('token').value = ''; updateTokenStatus(); }
      $('downloads').replaceChildren();
      for (const [kind, url] of Object.entries(job.files)) {
        const link = document.createElement('a'); link.href = url; link.textContent = 'Download ' + kind.toUpperCase();
        $('downloads').appendChild(link);
      }
      if (job.correction_url) { const link=document.createElement('a'); link.href=job.correction_url; link.textContent='Correct for personal adaptation'; $('downloads').appendChild(link); }
    } else if (job.status === 'error') {
      clearInterval(polling); setWorking(false); $('transcript').textContent = job.error;
      if (job.resumable) {
        const resume = document.createElement('button'); resume.textContent = 'Resume from saved stage'; resume.className = 'secondary';
        resume.onclick = async () => { const r = await fetch('/api/jobs/' + id + '/resume', {method:'POST'}); if (r.ok) { setWorking(true); pollJob(id); } };
        $('downloads').replaceChildren(resume);
      }
    }
  }, 800);
}

$('transcribe').addEventListener('click', async () => {
  const file = $('audioFile').files[0];
  if (!file && !recordedBlob) { $('status').textContent = 'Choose a recording or use the microphone first.'; return; }
  if ($('preset').value === 'fast' && $('engine').value === 'indicconformer' && !$('language').value) { $('status').textContent = 'Choose an Indian language for IndicConformer.'; return; }
  if ($('diarize').checked && selectedDuration !== null && selectedDuration < 10) {
    $('status').textContent = `This file is only ${selectedDuration.toFixed(1)} seconds. Turn off two-voice identification or choose the full conversation recording.`;
    return;
  }
  if ($('diarize').checked && !$('token').value && !hasSavedToken && !speakerModelCached) { $('status').textContent = 'Paste a Hugging Face token for speaker identification.'; return; }
  const form = new FormData();
  form.append('audio', recordedBlob || file, recordedBlob ? 'browser-recording.webm' : file.name);
  for (const id of ['preset','audioMode','primaryCandidate','secondaryCandidate','engine','language','model','device','prompt','speaker1','speaker2','token']) form.append(id, $(id).value);
  form.append('consensus', $('consensus').checked ? 'true' : 'false');
  form.append('readable', $('readable').checked ? 'true' : 'false');
  form.append('diarize', $('diarize').checked ? 'true' : 'false');
  form.append('remember_token', $('rememberToken').checked ? 'true' : 'false');
  form.append('keep_audio', $('keepAudio').checked ? 'true' : 'false');
  setWorking(true); $('progress').value = 1; $('downloads').replaceChildren(); $('transcript').textContent = '';
  $('status').textContent = 'Saving the local recording…';
  try {
    const response = await fetch('/api/transcribe', {method: 'POST', body: form});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not start transcription');
    pollJob(payload.job_id);
  } catch (error) { setWorking(false); $('status').textContent = error.message; }
});

async function refreshProfiles() {
  const response = await fetch('/api/profiles'); const payload = await response.json();
  $('profileList').replaceChildren();
  for (const name of payload.profiles || []) { const option=document.createElement('option'); option.value=name; option.textContent=name; $('profileList').appendChild(option); }
  if (!(payload.profiles || []).length) { const option=document.createElement('option'); option.textContent='No profiles enrolled'; option.value=''; $('profileList').appendChild(option); }
}
$('enrollProfile').addEventListener('click', async () => {
  const file=$('profileAudio').files[0], name=$('profileName').value.trim();
  if (!file || !name) { $('profileStatus').textContent='Choose a clean sample and enter the person’s name.'; return; }
  const form=new FormData(); form.append('audio',file,file.name); form.append('name',name); form.append('token',$('token').value); form.append('device',$('device').value);
  $('profileStatus').textContent='Extracting the encrypted voice profile…';
  const response=await fetch('/api/profiles/enroll',{method:'POST',body:form}); const payload=await response.json();
  $('profileStatus').textContent=response.ok ? `Enrolled ${name}; no audio was retained.` : payload.error;
  if (response.ok) { $('profileAudio').value=''; await refreshProfiles(); }
});
$('deleteProfile').addEventListener('click', async () => {
  const name=$('profileList').value; if (!name) return;
  const response=await fetch('/api/profiles/'+encodeURIComponent(name),{method:'DELETE'});
  if (response.ok) { $('profileStatus').textContent=`Deleted ${name}.`; await refreshProfiles(); }
});
refreshProfiles();
$('shutdown').addEventListener('click', async () => {
  await fetch('/api/shutdown', {method:'POST'}); document.body.innerHTML = '<main><section class="card"><h2>Local app stopped</h2><p>You may close this tab.</p></section></main>';
});
</script>
</body></html>"""


def _set_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)
        snapshot = dict(JOBS[job_id])
    try:
        directory = JOBS_DIR / job_id
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "status.tmp"
        import json

        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "status.json")
    except OSError:
        pass


@app.before_request
def protect_local_actions():
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("Origin")
        if origin and origin not in LOCAL_ORIGINS:
            return jsonify(error="This local action was blocked because it came from another site."), 403


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if (
        "401" in lowered
        or "gated repo" in lowered
        or "access to model" in lowered
        or "invalid user token" in lowered
        or "invalid username or password" in lowered
    ):
        return (
            "Hugging Face denied access to the speaker model (401).\n\n"
            "1. Sign in at Hugging Face using the account that owns the token.\n"
            f"2. Open {PYANNOTE_MODEL_URL} and accept its access/contact-sharing conditions.\n"
            f"3. Create a new read token at {HF_TOKEN_URL}.\n"
            "4. Paste that new token into this app and try again.\n\n"
            "Do not paste the token into chat or save it in this project."
        )
    return message


def _run_job(job_id: str, source: Path, settings: dict[str, str]) -> None:
    global ACTIVE_JOB_ID
    completed = False

    def progress(message: str, value: float | None) -> None:
        update: dict[str, Any] = {"message": message}
        if value is not None:
            update["progress"] = value
        _set_job(job_id, **update)

    try:
        with TRANSCRIBE_LOCK:
            progress("Loading the transcription engine…", 0.01)
            if settings.get("preset", "fast") == "maximum":
                candidates = tuple(
                    item
                    for item in (
                        settings.get("primary_candidate", "sravaani"),
                        settings.get("secondary_candidate", ""),
                    )
                    if item
                )
                glossary = tuple(
                    item.strip()
                    for item in settings.get("prompt", "").replace("\n", ",").split(",")
                    if item.strip()
                )
                result = AccuracyPipeline(ROOT_DIR, progress).transcribe(
                    source,
                    AccuracySettings(
                        device=settings["device"],
                        language=settings["language"] or "hi",
                        audio_mode=settings.get("audio_mode", "original"),
                        candidates=candidates,
                        use_consensus=settings.get("consensus", "false") == "true",
                        diarize=settings["diarize"] == "true",
                        speaker_names=(settings["speaker1"], settings["speaker2"]),
                        glossary=glossary,
                        hf_token=settings["token"],
                        generate_readable=settings.get("readable", "true") == "true",
                        use_readable_model=True,
                        allow_model_downloads=False,
                    ),
                    checkpoint_dir=JOBS_DIR / job_id / "stages",
                )
            else:
                key = (settings["engine"], settings["model"], settings["device"])
                transcriber = TRANSCRIBERS.get(key)
                if transcriber is None:
                    transcriber = Transcriber(
                        model_size=settings["model"],
                        device=settings["device"],
                        engine=settings["engine"],
                        progress_callback=progress,
                    )
                    TRANSCRIBERS[key] = transcriber
                else:
                    transcriber.progress_callback = progress
                result = transcriber.transcribe(
                    source,
                    settings["language"] or None,
                    settings["prompt"] or None,
                )
                if settings["diarize"] == "true":
                    result = diarize_two_speakers(
                        result,
                        settings["token"],
                        (settings["speaker1"], settings["speaker2"]),
                        settings["device"],
                        progress,
                    )
            paths = write_outputs(result, TRANSCRIPTS_DIR)

        files = {kind: f"/files/{path.name}" for kind, path in paths.items()}
        transcript = paths["txt"].read_text(encoding="utf-8")
        _set_job(
            job_id,
            status="done",
            message=f"Complete. Files saved in {TRANSCRIPTS_DIR}",
            progress=1.0,
            transcript=transcript,
            files=files,
            correction_url=(f"/correct/{paths['json'].name}" if "json" in paths else None),
            has_saved_token=has_token(),
        )
        completed = True
    except Exception as exc:
        if settings["diarize"] == "true" and (
            "401" in str(exc).lower() or "gated repo" in str(exc).lower()
        ):
            try:
                forget_token()
            except OSError:
                pass
        _set_job(
            job_id,
            status="error",
            message="Transcription failed",
            error=_friendly_error(exc),
            progress=0.0,
            resumable=source.is_file(),
        )
    finally:
        settings["token"] = ""
        if completed and settings["keep_audio"] != "true":
            source.unlink(missing_ok=True)
        with JOBS_LOCK:
            if ACTIVE_JOB_ID == job_id:
                ACTIVE_JOB_ID = None


@app.get("/")
def index() -> Response:
    import json

    availability = optional_model_availability()
    recommended = recommended_candidates(ROOT_DIR)
    config = {
        "engines": list(ENGINE_CHOICES.items()),
        "languages": [(name, code or "") for name, code in LANGUAGES.items()],
        "models": list(MODEL_CHOICES.items()),
        "has_saved_token": has_token(),
        "speaker_model_cached": speaker_model_is_cached(),
        "candidates": [
            (label, value, availability.get(value, False))
            for value, label in CANDIDATE_LABELS.items()
        ],
        "recommended_candidates": recommended,
        "promoted_defaults": promoted_defaults(ROOT_DIR),
    }
    return Response(HTML.replace("__CONFIG__", json.dumps(config)), mimetype="text/html")


@app.post("/api/transcribe")
def start_transcription():
    global ACTIVE_JOB_ID

    uploaded = request.files.get("audio")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="No audio recording was supplied"), 400
    engine = request.form.get("engine", "whisper")
    language = request.form.get("language", "")
    model = request.form.get("model", "medium")
    device = request.form.get("device", "auto")
    preset = request.form.get("preset", "maximum")
    audio_mode = request.form.get("audioMode", "original")
    primary_candidate = request.form.get("primaryCandidate", "sravaani")
    secondary_candidate = request.form.get("secondaryCandidate", "")
    if engine not in ENGINE_CHOICES.values() or model not in MODEL_CHOICES.values():
        return jsonify(error="Invalid engine or model"), 400
    if language and language not in {value for value in LANGUAGES.values() if value}:
        return jsonify(error="Invalid language"), 400
    if device not in {"auto", "cuda", "cpu"}:
        return jsonify(error="Invalid device"), 400
    if preset not in {"maximum", "fast"} or audio_mode not in {"original", "telephone", "auto"}:
        return jsonify(error="Invalid accuracy or audio preset"), 400
    if primary_candidate not in CANDIDATE_LABELS or (
        secondary_candidate and secondary_candidate not in CANDIDATE_LABELS
    ):
        return jsonify(error="Invalid accuracy model candidate"), 400

    supplied_token = request.form.get("token", "").strip()
    try:
        token = supplied_token or load_token()
    except Exception:
        token = supplied_token
    diarize = request.form.get("diarize", "false")
    if diarize == "true" and not token and not speaker_model_is_cached():
        return jsonify(error="No Hugging Face token is available for speaker identification"), 400

    remember_token = request.form.get("remember_token", "true")
    if supplied_token:
        try:
            HfApi().whoami(token=supplied_token)
        except Exception as exc:
            return jsonify(error=_friendly_error(exc)), 401

    if supplied_token and remember_token == "true":
        try:
            save_token(supplied_token)
        except OSError as exc:
            return jsonify(error=f"Could not save the token in Windows Credential Manager: {exc}"), 500

    original = secure_filename(uploaded.filename) or "conversation.webm"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = RECORDINGS_DIR / f"{stamp}_{uuid.uuid4().hex[:6]}_{original}"
    source.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(source)

    settings = {
        "preset": preset,
        "audio_mode": audio_mode,
        "primary_candidate": primary_candidate,
        "secondary_candidate": secondary_candidate,
        "consensus": request.form.get("consensus", "false"),
        "readable": request.form.get("readable", "true"),
        "engine": engine,
        "language": language,
        "model": model,
        "device": device,
        "prompt": request.form.get("prompt", "").strip(),
        "diarize": diarize,
        "remember_token": remember_token,
        "keep_audio": request.form.get("keep_audio", "false"),
        "speaker1": request.form.get("speaker1", "Speaker 1").strip() or "Speaker 1",
        "speaker2": request.form.get("speaker2", "Speaker 2").strip() or "Speaker 2",
        "token": token,
    }
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        if ACTIVE_JOB_ID is not None:
            active = JOBS.get(ACTIVE_JOB_ID, {})
            if active.get("status") not in {"done", "error"}:
                source.unlink(missing_ok=True)
                return jsonify(
                    error="Another transcription is already running. Wait for it to finish or stop and restart the local app."
                ), 409
        ACTIVE_JOB_ID = job_id
        JOBS[job_id] = {"status": "running", "message": "Starting locally…", "progress": 0.0}
    request_state = {"source": str(source), "settings": {**settings, "token": ""}}
    job_directory = JOBS_DIR / job_id
    job_directory.mkdir(parents=True, exist_ok=True)
    import json

    (job_directory / "request.json").write_text(
        json.dumps(request_state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    threading.Thread(target=_run_job, args=(job_id, source, settings), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.post("/api/token/forget")
def forget_saved_token():
    forget_token()
    return jsonify(ok=True)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="Unknown job"), 404
        return jsonify(job)


@app.post("/api/jobs/<job_id>/resume")
def resume_job(job_id: str):
    global ACTIVE_JOB_ID
    request_path = JOBS_DIR / job_id / "request.json"
    if not request_path.is_file():
        return jsonify(error="This job has no saved resume state"), 404
    import json

    state = json.loads(request_path.read_text(encoding="utf-8"))
    source = Path(state["source"])
    if not source.is_file():
        return jsonify(error="The saved local audio copy is missing"), 409
    settings = {str(key): str(value) for key, value in state["settings"].items()}
    try:
        settings["token"] = load_token()
    except OSError:
        settings["token"] = ""
    if settings.get("diarize") == "true" and not settings["token"] and not speaker_model_is_cached():
        return jsonify(error="Speaker identification needs the saved Hugging Face token"), 400
    with JOBS_LOCK:
        if ACTIVE_JOB_ID and ACTIVE_JOB_ID != job_id:
            active = JOBS.get(ACTIVE_JOB_ID, {})
            if active.get("status") not in {"done", "error"}:
                return jsonify(error="Another transcription is already running"), 409
        ACTIVE_JOB_ID = job_id
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(status="running", message="Resuming saved stages locally…", progress=0.01)
    threading.Thread(target=_run_job, args=(job_id, source, settings), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.get("/api/profiles")
def profiles_status():
    return jsonify(profiles=list(list_profiles()))


@app.post("/api/profiles/enroll")
def enroll_speaker_profile():
    uploaded = request.files.get("audio")
    name = request.form.get("name", "").strip()
    if uploaded is None or not uploaded.filename or not name:
        return jsonify(error="A person name and clean voice sample are required"), 400
    supplied_token = request.form.get("token", "").strip()
    try:
        token = supplied_token or load_token()
    except OSError:
        token = supplied_token
    if not token and not speaker_model_is_cached():
        return jsonify(error="Paste a Hugging Face read token once to load Community-1"), 400
    if supplied_token:
        try:
            HfApi().whoami(token=supplied_token)
            save_token(supplied_token)
        except Exception as exc:
            return jsonify(error=_friendly_error(exc)), 401
    filename = secure_filename(uploaded.filename) or "enrollment.wav"
    temporary = RECORDINGS_DIR / f"enroll_{uuid.uuid4().hex}_{filename}"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(temporary)
    try:
        dimension = enroll_profile(name, temporary, token, request.form.get("device", "auto"))
    except Exception as exc:
        return jsonify(error=_friendly_error(exc)), 400
    finally:
        temporary.unlink(missing_ok=True)
    return jsonify(ok=True, name=name, embedding_dimension=dimension)


@app.delete("/api/profiles/<path:name>")
def remove_speaker_profile(name: str):
    try:
        delete_profile(name)
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(ok=True)


@app.get("/benchmark")
def benchmark_editor() -> Response:
    return Response(BENCHMARK_HTML, mimetype="text/html")


@app.post("/api/benchmark/prepare")
def prepare_accuracy_benchmark():
    try:
        manifest = prepare_benchmark(ROOT_DIR)
    except Exception as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(ok=True, manifest=str(manifest))


@app.get("/api/benchmark/data")
def benchmark_data():
    try:
        manifest = load_manifest(ROOT_DIR)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 404
    import json

    for item in manifest["items"]:
        draft_path = TRANSCRIPTS_DIR / f"{item['id']}.transcript.json"
        item["draft_segments"] = []
        if draft_path.is_file():
            try:
                payload = json.loads(draft_path.read_text(encoding="utf-8"))
                item["draft_segments"] = payload.get("segments", [])
            except (OSError, json.JSONDecodeError):
                pass
    report_path = benchmark_root(ROOT_DIR) / "report.json"
    if report_path.is_file():
        try:
            manifest["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return jsonify(manifest)


@app.put("/api/benchmark/<item_id>")
def update_benchmark_gold(item_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        item = save_gold_segments(ROOT_DIR, item_id, payload.get("segments", []))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(ok=True, item=item)


@app.get("/benchmark/audio/<path:name>")
def benchmark_audio(name: str):
    return send_from_directory(benchmark_root(ROOT_DIR) / "audio", name)


def _benchmark_job_settings(token: str) -> dict[str, str]:
    promoted = recommended_candidates(ROOT_DIR)
    return {
        "preset": "maximum",
        "audio_mode": "original",
        "primary_candidate": promoted[0] if promoted else "sravaani",
        "secondary_candidate": "",
        "consensus": "false",
        "readable": "false",
        "engine": "sravaani",
        "language": "hi",
        "model": "medium",
        "device": "auto",
        "prompt": "",
        "diarize": "true",
        "remember_token": "true",
        "keep_audio": "true",
        "speaker1": "Speaker 1",
        "speaker2": "Speaker 2",
        "token": token,
    }


@app.post("/api/benchmark/<item_id>/draft")
def generate_benchmark_draft(item_id: str):
    global ACTIVE_JOB_ID
    try:
        manifest = load_manifest(ROOT_DIR)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 404
    item = next((entry for entry in manifest["items"] if entry["id"] == item_id), None)
    if item is None:
        return jsonify(error="Unknown benchmark clip"), 404
    try:
        token = load_token()
    except OSError:
        token = ""
    if not token and not speaker_model_is_cached():
        return jsonify(error="Generate a speaker-labelled draft after saving the Hugging Face token once"), 400
    source = ROOT_DIR / item["audio"]
    settings = _benchmark_job_settings(token)
    job_id = f"draft-{item_id}-{uuid.uuid4().hex[:8]}"
    with JOBS_LOCK:
        if ACTIVE_JOB_ID and JOBS.get(ACTIVE_JOB_ID, {}).get("status") not in {"done", "error"}:
            return jsonify(error="Another local job is running"), 409
        ACTIVE_JOB_ID = job_id
        JOBS[job_id] = {"status": "running", "message": "Generating a correction draft…", "progress": 0.0}
    threading.Thread(target=_run_job, args=(job_id, source, settings), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.post("/api/benchmark/run")
def start_model_benchmark():
    global ACTIVE_JOB_ID
    try:
        token = load_token()
    except OSError:
        token = ""
    candidates = tuple(name for name, available in optional_model_availability().items() if available)
    job_id = f"benchmark-{uuid.uuid4().hex[:10]}"
    with JOBS_LOCK:
        if ACTIVE_JOB_ID and JOBS.get(ACTIVE_JOB_ID, {}).get("status") not in {"done", "error"}:
            return jsonify(error="Another local job is running"), 409
        ACTIVE_JOB_ID = job_id
        JOBS[job_id] = {"status": "running", "message": "Starting measured model benchmark…", "progress": 0.0}

    def worker() -> None:
        global ACTIVE_JOB_ID
        try:
            report = run_benchmark(
                ROOT_DIR,
                candidates=candidates,
                hf_token=token,
                progress_callback=lambda message, value: _set_job(
                    job_id, message=message, progress=value if value is not None else 0.0
                ),
            )
            import json

            _set_job(
                job_id,
                status="done",
                message="Benchmark complete; measured configuration saved.",
                progress=1.0,
                transcript=json.dumps(report, ensure_ascii=False, indent=2),
                files={"report": "/benchmark/files/report.json"},
            )
        except Exception as exc:
            _set_job(job_id, status="error", message="Benchmark failed", error=str(exc), progress=0.0)
        finally:
            with JOBS_LOCK:
                if ACTIVE_JOB_ID == job_id:
                    ACTIVE_JOB_ID = None

    threading.Thread(target=worker, daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.get("/benchmark/files/<path:name>")
def benchmark_file(name: str):
    return send_from_directory(benchmark_root(ROOT_DIR), name, as_attachment=True)


def _transcript_json_path(name: str) -> Path:
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("Invalid transcript file")
    path = TRANSCRIPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError("Transcript was not found")
    return path


def _correction_path(name: str) -> Path:
    return CORRECTIONS_DIR / f"{Path(name).stem}.corrected.json"


@app.get("/correct/<path:name>")
def correction_editor(name: str):
    try:
        _transcript_json_path(name)
    except (ValueError, FileNotFoundError) as exc:
        return Response(str(exc), status=404, mimetype="text/plain")
    return Response(CORRECTION_HTML, mimetype="text/html")


@app.get("/api/corrections/<path:name>")
def correction_data(name: str):
    import json

    try:
        transcript_path = _transcript_json_path(name)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(error=str(exc)), 404
    path = _correction_path(name)
    payload = json.loads((path if path.is_file() else transcript_path).read_text(encoding="utf-8"))
    source = Path(payload["source"])
    payload["audio_available"] = source.is_file() and source.resolve().is_relative_to(RECORDINGS_DIR.resolve())
    return jsonify(payload)


@app.put("/api/corrections/<path:name>")
def update_correction(name: str):
    import json

    try:
        transcript_path = _transcript_json_path(name)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify(error=str(exc)), 404
    original = json.loads(transcript_path.read_text(encoding="utf-8"))
    request_payload = request.get_json(silent=True) or {}
    supplied = request_payload.get("segments", [])
    validated = []
    last_start = 0.0
    for item in supplied:
        start, end = float(item["start"]), float(item["end"])
        if start < last_start or end <= start or end > float(original["duration_seconds"]) + 0.01:
            return jsonify(error="Segments must be ordered and inside the recording"), 400
        text = str(item.get("text", "")).strip()
        if text:
            validated.append({
                "start": start,
                "end": end,
                "speaker": str(item.get("speaker", "")).strip() or None,
                "text": text,
                "overlap": bool(item.get("overlap", False)),
            })
        last_start = start
    payload = {
        **original,
        "segments": validated,
        "text": " ".join(item["text"] for item in validated),
        "correction_metadata": {
            "training_eligible": True,
            "held_out": False,
            "split_group": Path(original["source"]).name,
            "verified": True,
            "tags": {
                "dialect": str(request_payload.get("tags", {}).get("dialect", "Hindi + dialect")),
                "noise": str(request_payload.get("tags", {}).get("noise", "Telephone narrowband")),
            },
        },
    }
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _correction_path(name)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return jsonify(ok=True, path=path.name)


@app.get("/correction-audio/<path:name>")
def correction_audio(name: str):
    import json

    try:
        path = _transcript_json_path(name)
    except (ValueError, FileNotFoundError):
        return Response("Not found", status=404)
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = Path(payload["source"]).resolve()
    if not source.is_file() or not source.is_relative_to(RECORDINGS_DIR.resolve()):
        return Response("The local audio copy was not retained", status=404)
    return send_from_directory(source.parent, source.name)


@app.get("/files/<path:name>")
def result_file(name: str):
    return send_from_directory(TRANSCRIPTS_DIR, name, as_attachment=True)


@app.post("/api/shutdown")
def shutdown():
    if SERVER is not None:
        threading.Thread(target=SERVER.shutdown, daemon=True).start()
    return jsonify(ok=True)


def _already_running() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=0.3):
            return True
    except OSError:
        return False


def _restore_jobs() -> None:
    import json

    if not JOBS_DIR.is_dir():
        return
    for status_path in JOBS_DIR.glob("*/status.json"):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        job_id = status_path.parent.name
        if payload.get("status") == "running":
            request_path = status_path.parent / "request.json"
            payload.update(
                status="error",
                message="Interrupted job can be resumed",
                error="The app stopped before this job completed.",
                resumable=request_path.is_file(),
            )
        JOBS[job_id] = payload


def main() -> None:
    global SERVER
    url = f"http://{HOST}:{PORT}/"
    if _already_running():
        webbrowser.open(url)
        return
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _restore_jobs()
    SERVER = make_server(HOST, PORT, app, threaded=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    SERVER.serve_forever()


if __name__ == "__main__":
    main()
