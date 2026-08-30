from __future__ import annotations

import unicodedata


_INDIC_SCRIPT_NAMES = (
    "BENGALI",
    "GURMUKHI",
    "GUJARATI",
    "ORIYA",
    "TAMIL",
    "TELUGU",
    "KANNADA",
    "MALAYALAM",
)

_SCRIPT_MARK_EQUIVALENTS = {
    "GURMUKHI SIGN BINDI": "DEVANAGARI SIGN ANUSVARA",
    "GURMUKHI TIPPI": "DEVANAGARI SIGN ANUSVARA",
    "GURMUKHI ADDAK": "DEVANAGARI SIGN VIRAMA",
    "BENGALI SIGN CANDRABINDU": "DEVANAGARI SIGN CANDRABINDU",
    "GUJARATI SIGN CANDRABINDU": "DEVANAGARI SIGN CANDRABINDU",
}


def to_devanagari(text: str) -> str:
    """Map structurally equivalent Indic-script characters to Devanagari.

    SraVaani occasionally selects a neighbouring script for Hindi telephone
    speech. Unicode character names provide a deterministic phonetic conversion
    without translating or inventing any words.
    """
    converted: list[str] = []
    for character in text:
        try:
            name = unicodedata.name(character)
        except ValueError:
            converted.append(character)
            continue
        replacement = None
        explicit_name = _SCRIPT_MARK_EQUIVALENTS.get(name)
        if explicit_name:
            replacement = unicodedata.lookup(explicit_name)
        for script in _INDIC_SCRIPT_NAMES:
            if replacement:
                break
            if name.startswith(script + " "):
                try:
                    replacement = unicodedata.lookup("DEVANAGARI" + name[len(script) :])
                except KeyError:
                    pass
                break
        converted.append(replacement or character)
    return unicodedata.normalize("NFC", "".join(converted))
