# Personal SraVaani adaptation

This directory implements the personal-adaptation gate without mixing the permanent 10-minute benchmark into training.

1. Keep the app's local audio copy for calls you consent to use, correct them from the result page, and collect at least three verified hours across at least three calls.
2. Prepare deterministic call-level 80/10/10 shards:

   ```powershell
   .\.venv\Scripts\python.exe .\training\prepare_corrections.py
   ```

3. From an Administrator PowerShell, install WSL only when ready:

   ```powershell
   .\setup-wsl-training.ps1 -InstallWsl
   ```

   After any required restart, run `setup-wsl-training.ps1` again without the switch.

4. Download the official `SraVaani-nemo-checkpoint.nemo` described in the [official fine-tuning repository](https://github.com/ARTPARK-Speech-Models/SraVaani). Copy `training/data` into `~/sravaani-personal/SraVaani/data`, set the official script to batch size 2 with gradient accumulation 16 for the RTX 4080, and run its encoder-frozen `finetune.py`. Select the checkpoint with the lowest validation WER; early-stop if validation WER fails to improve for five epochs.
5. Evaluate the candidate on the untouched app benchmark. Accept it only if WER improves by at least two absolute points and names, numbers, and speakers do not regress.
6. Export an accepted `.nemo` checkpoint under WSL:

   ```bash
   python /mnt/c/Users/mohit/Documents/ChatGPT/voice\ to\ text/training/export_sravaani_onnx.py \
     sra-vaani-finetuned.nemo ~/sravaani-personal/exported-onnx
   ```

   Validate the export with `onnx_asr.load_model("nemo-conformer-tdt", path=...)` before configuring it as the Windows default. The released SraVaani model remains the fallback.
