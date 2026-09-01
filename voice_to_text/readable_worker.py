from __future__ import annotations

import json
from pathlib import Path
import sys


def _extract_json(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, list)
            and payload
            and all(isinstance(item, dict) and "id" in item and "text" in item for item in payload)
        ):
            return payload
    raise ValueError("Readable model did not return a structured JSON turn array")


def main() -> int:
    if len(sys.argv) != 4:
        return 2
    input_path, output_path, runtime_path = map(Path, sys.argv[1:])
    sys.path.insert(0, str(runtime_path.resolve()))
    import torch
    import transformers

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    model_id = str(payload["model"])
    turns = list(payload["turns"])
    processor = transformers.AutoProcessor.from_pretrained(model_id, local_files_only=True)
    loader = getattr(transformers, "AutoModelForMultimodalLM", None)
    if loader is None:
        loader = transformers.AutoModelForImageTextToText
    model = loader.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    ).eval()
    source = json.dumps(turns, ensure_ascii=False)
    instruction = (
        "यह फोन बातचीत के numbered turns हैं। केवल विराम-चिह्न, स्पष्ट ASR disfluency और "
        "बहुत हल्का मानक-हिंदी रूप सुधारें। अर्थ, नाम, संख्या, speaker, turn id या तथ्य न बदलें। "
        "हर input id को ठीक एक बार रखते हुए केवल JSON array लौटाएँ: "
        "[{\"id\":1,\"text\":\"...\"}].\n" + source
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=min(3072, max(256, len(source))),
            do_sample=False,
            repetition_penalty=1.05,
        )
    text = processor.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    output_path.write_text(
        json.dumps({"turns": _extract_json(text)}, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
