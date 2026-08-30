from __future__ import annotations

from collections import Counter
import importlib.util
import re
from typing import Iterable

from .core import TranscriptionResult
from .metrics import normalize_text, number_accuracy


READABLE_MODEL = "Qwen/Qwen3.5-4B"


def _tokens(text: str) -> Counter[str]:
    return Counter(normalize_text(text, canonicalize_numbers=False).split())


def validate_readable_copy(source: str, rewritten: str, glossary: Iterable[str] = ()) -> bool:
    if not rewritten.strip() or number_accuracy(source, rewritten) < 1.0:
        return False
    source_tokens = _tokens(source)
    rewritten_tokens = _tokens(rewritten)
    retained = sum(min(count, rewritten_tokens[token]) for token, count in source_tokens.items())
    introduced = sum(max(0, count - source_tokens[token]) for token, count in rewritten_tokens.items())
    source_count = max(1, sum(source_tokens.values()))
    if retained / source_count < 0.88 or introduced / source_count > 0.12:
        return False
    source_normalized = normalize_text(source, canonicalize_numbers=False)
    rewritten_normalized = normalize_text(rewritten, canonicalize_numbers=False)
    for item in glossary:
        normalized = normalize_text(item, canonicalize_numbers=False)
        if normalized and normalized in source_normalized and normalized not in rewritten_normalized:
            return False
    return True


def _safe_fallback(result: TranscriptionResult) -> str:
    lines = []
    for segment in result.segments:
        prefix = f"{segment.speaker}: " if segment.speaker else ""
        text = segment.text.strip()
        if text and text[-1] not in ".!?।":
            text += "।"
        lines.append(prefix + text)
    return "\n".join(lines).strip()


def generate_readable_copy(
    result: TranscriptionResult,
    glossary: Iterable[str] = (),
    *,
    use_local_model: bool = True,
    allow_download: bool = False,
) -> tuple[str, str]:
    """Return a conservative readable copy and the method used.

    The verbatim result is never modified. If the optional language model is not
    installed/cached or changes content, a punctuation-only copy is returned.
    """
    fallback = _safe_fallback(result)
    if not use_local_model or not importlib.util.find_spec("transformers"):
        return fallback, "punctuation-only safeguard"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(READABLE_MODEL, local_files_only=not allow_download)
        model = AutoModelForCausalLM.from_pretrained(
            READABLE_MODEL,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            local_files_only=not allow_download,
        ).eval()
        source = fallback
        prompt = (
            "नीचे फोन पर हुई बातचीत की शब्दशः प्रति है। केवल विराम-चिह्न और बहुत हल्का "
            "मानक-हिंदी रूप सुधारें। कोई तथ्य, नाम, संख्या या वाक्य न जोड़ें और न हटाएँ। "
            "केवल सुधारा हुआ पाठ लौटाएँ।\n\n" + source
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=min(2048, max(128, len(source) // 2)),
                do_sample=False,
                repetition_penalty=1.05,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        rewritten = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if validate_readable_copy(source, rewritten, glossary):
            return rewritten, READABLE_MODEL
    except Exception:
        pass
    return fallback, "punctuation-only safeguard"
