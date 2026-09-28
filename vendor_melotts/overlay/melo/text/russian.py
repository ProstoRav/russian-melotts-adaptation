"""Russian frontend for a pinned MeloTTS checkout.

BERT-off POC implementation. Requires espeak-ng.
"""
from __future__ import annotations

import re
import subprocess
import unicodedata
from functools import lru_cache

import torch

PUNCTUATION = set(".,!?;:'\"()-")
TOKEN_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9+\u0301]+|[.,!?;:'\"()-]")
MULTIGRAPHS = ("ts",)
REPLACEMENTS = {"t͡s": "ts", "t͡ʃ": "tʃ", "d͡z": "dz", "d͡ʒ": "dʒ", "ɡ": "g"}


def text_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace(";", ",")
    text = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s.,!?;:'\"()\-+\u0301]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=50000)
def _espeak(word: str) -> str:
    try:
        result = subprocess.run(
            ["espeak-ng", "-q", "-v", "ru", "--ipa=3", word],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Install espeak-ng before preprocessing Russian text") from exc
    return unicodedata.normalize("NFC", result.stdout.strip())


def _phones(ipa: str) -> list[str]:
    for source, target in REPLACEMENTS.items():
        ipa = ipa.replace(source, target)

    # eSpeak NG may insert Unicode format joiners into IPA output.
    # U+200D (ZWJ) is not a phoneme and must never reach MeloTTS vocabulary.
    #
    # Remove all Unicode Format characters (category Cf), while keeping
    # real IPA combining marks. Remove U+0361 tie bar separately:
    # the underlying phoneme characters remain intact.
    ipa = "".join(
        ch for ch in ipa
        if unicodedata.category(ch) != "Cf"
    )
    ipa = ipa.replace("\u0361", "")

    out = []
    i = 0
    while i < len(ipa):
        if ipa[i].isspace() or ipa[i] == "_":
            i += 1
            continue
        multi = next((x for x in MULTIGRAPHS if ipa.startswith(x, i)), None)
        if multi:
            out.append(multi)
            i += len(multi)
        else:
            out.append(ipa[i])
            i += 1
    return out


def g2p(text: str, pad_start_end=True, tokenized=None):
    tokens = tokenized or TOKEN_RE.findall(text_normalize(text))
    phones, tones, word2ph = [], [], []
    for token in tokens:
        current = [token] if token in PUNCTUATION else (_phones(_espeak(token.lower())) or ["UNK"])
        phones.extend(current)
        tones.extend([0] * len(current))
        word2ph.append(len(current))
    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones) == sum(word2ph)
    return phones, tones, word2ph


def get_bert_feature(text, word2ph, device=None):
    return torch.zeros(768, sum(word2ph), device="cpu")
