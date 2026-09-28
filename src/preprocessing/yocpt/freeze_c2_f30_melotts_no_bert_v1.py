from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path.home() / "projects/ru_melotts_research"
MELO = Path.home() / "projects/MeloTTS"

RAW_ROOT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_melotts_raw_v1"
)

TRAIN_RAW = RAW_ROOT / "train_raw.list"
VAL_RAW = RAW_ROOT / "val_raw.list"
METADATA_RAW = RAW_ROOT / "metadata.list"

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_melotts_no_bert_v1"
)

WAV_DIR = (
    Path.home()
    / "datasets/tts/yocpt_c2_f30_v1/wavs"
)

EXPECTED_TRAIN = 12257
EXPECTED_VAL = 432
EXPECTED_TOTAL = 12689

EXPECTED_SPEAKER = "c2_f30_generic_ru"
EXPECTED_LANGUAGE = "RU"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def count_cache(suffix: str) -> int:
    return sum(
        1
        for _ in WAV_DIR.glob(
            f"*{suffix}"
        )
    )


if OUT.exists():
    raise RuntimeError(
        f"refusing to overwrite frozen artifact: {OUT}"
    )


#
# Use exact checked-out local MeloTTS.
#
sys.path.insert(
    0,
    str(MELO),
)

from melo.text.cleaner import clean_text  # noqa: E402
from melo.text.symbols import symbols as MELO_SYMBOLS  # noqa: E402


symbol_set = set(
    MELO_SYMBOLS
)


train_raw = (
    TRAIN_RAW
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

val_raw = (
    VAL_RAW
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

metadata_raw = (
    METADATA_RAW
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)


if len(train_raw) != EXPECTED_TRAIN:
    raise RuntimeError(
        f"train raw rows={len(train_raw)}"
    )

if len(val_raw) != EXPECTED_VAL:
    raise RuntimeError(
        f"val raw rows={len(val_raw)}"
    )

if len(metadata_raw) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"metadata raw rows={len(metadata_raw)}"
    )

if metadata_raw != (
    train_raw
    + val_raw
):
    raise RuntimeError(
        "metadata.list != train_raw + val_raw"
    )


bert_before = count_cache(
    ".bert.pt"
)

spec_before = count_cache(
    ".spec.pt"
)


phone_counts = []
word2ph_counts = []

unique_phones = Counter()

normalized_changed = 0

unknown_symbols = Counter()


def process_line(
    line: str,
    global_index: int,
) -> str:

    global normalized_changed

    parts = line.split(
        "|",
        3,
    )

    if len(parts) != 4:
        raise RuntimeError(
            f"row {global_index}: "
            "expected 4 raw fields"
        )

    wav, speaker, language, text = parts


    if speaker != EXPECTED_SPEAKER:
        raise RuntimeError(
            f"row {global_index}: "
            f"speaker={speaker!r}"
        )

    if language != EXPECTED_LANGUAGE:
        raise RuntimeError(
            f"row {global_index}: "
            f"language={language!r}"
        )

    path = Path(wav)

    if not path.is_file():
        raise RuntimeError(
            f"row {global_index}: "
            f"missing WAV {path}"
        )


    result = clean_text(
        text,
        language,
    )

    if not isinstance(
        result,
        (tuple, list),
    ):
        raise RuntimeError(
            f"row {global_index}: "
            f"clean_text type={type(result)}"
        )

    if len(result) != 4:
        raise RuntimeError(
            f"row {global_index}: "
            f"clean_text fields={len(result)}"
        )


    norm_text, phones, tones, word2ph = result

    phones = list(phones)
    tones = [
        int(x)
        for x in tones
    ]

    word2ph = [
        int(x)
        for x in word2ph
    ]


    if not norm_text:
        raise RuntimeError(
            f"row {global_index}: "
            "empty normalized text"
        )

    if "|" in norm_text:
        raise RuntimeError(
            f"row {global_index}: "
            "normalized text contains |"
        )

    if "\n" in norm_text or "\r" in norm_text:
        raise RuntimeError(
            f"row {global_index}: "
            "normalized text contains newline"
        )

    if not phones:
        raise RuntimeError(
            f"row {global_index}: "
            "empty phones"
        )

    if len(phones) != len(tones):
        raise RuntimeError(
            f"row {global_index}: "
            f"phones={len(phones)} "
            f"tones={len(tones)}"
        )

    if not word2ph:
        raise RuntimeError(
            f"row {global_index}: "
            "empty word2ph"
        )

    if any(
        x < 0
        for x in word2ph
    ):
        raise RuntimeError(
            f"row {global_index}: "
            "negative word2ph"
        )

    if sum(word2ph) != len(phones):
        raise RuntimeError(
            f"row {global_index}: "
            f"sum(word2ph)={sum(word2ph)} "
            f"phones={len(phones)}"
        )


    for phone in phones:

        unique_phones[
            str(phone)
        ] += 1

        if phone not in symbol_set:
            unknown_symbols[
                str(phone)
            ] += 1


    if norm_text != text:
        normalized_changed += 1


    phone_counts.append(
        len(phones)
    )

    word2ph_counts.append(
        len(word2ph)
    )


    cleaned = "|".join([
        wav,
        speaker,
        language,
        norm_text,
        " ".join(
            str(x)
            for x in phones
        ),
        " ".join(
            str(x)
            for x in tones
        ),
        " ".join(
            str(x)
            for x in word2ph
        ),
    ])


    if len(
        cleaned.split("|")
    ) != 7:
        raise RuntimeError(
            f"row {global_index}: "
            "cleaned field count != 7"
        )


    return cleaned


print(
    "===== C2-F30 MELOTTS NO-BERT PREPROCESS V1 ====="
)

print(
    "raw train:",
    len(train_raw),
)

print(
    "raw val:",
    len(val_raw),
)

print(
    "BERT files before:",
    bert_before,
)

print(
    "SPEC files before:",
    spec_before,
)

print()


train_cleaned = []

for i, line in enumerate(
    train_raw,
    start=1,
):

    train_cleaned.append(
        process_line(
            line,
            i,
        )
    )

    if (
        i % 500 == 0
        or i == len(train_raw)
    ):
        print(
            f"train {i}/{len(train_raw)}"
        )


val_cleaned = []

offset = len(train_raw)

for i, line in enumerate(
    val_raw,
    start=1,
):

    val_cleaned.append(
        process_line(
            line,
            offset + i,
        )
    )

    if (
        i % 100 == 0
        or i == len(val_raw)
    ):
        print(
            f"val {i}/{len(val_raw)}"
        )


metadata_cleaned = (
    train_cleaned
    + val_cleaned
)


if len(train_cleaned) != EXPECTED_TRAIN:
    raise RuntimeError(
        "clean train row mismatch"
    )

if len(val_cleaned) != EXPECTED_VAL:
    raise RuntimeError(
        "clean val row mismatch"
    )

if len(metadata_cleaned) != EXPECTED_TOTAL:
    raise RuntimeError(
        "clean total row mismatch"
    )


if unknown_symbols:
    print()
    print(
        "===== UNKNOWN PHONE SYMBOLS ====="
    )

    for symbol, count in (
        unknown_symbols
        .most_common()
    ):
        print(
            repr(symbol),
            count,
        )

    raise RuntimeError(
        f"unknown MeloTTS phone symbols: "
        f"{len(unknown_symbols)}"
    )


#
# Verify that preprocessing itself did not
# generate either BERT or spectrogram cache.
#
bert_after = count_cache(
    ".bert.pt"
)

spec_after = count_cache(
    ".spec.pt"
)


if bert_after != bert_before:
    raise RuntimeError(
        "BERT cache count changed: "
        f"{bert_before} -> {bert_after}"
    )

if spec_after != spec_before:
    raise RuntimeError(
        "SPEC cache count changed: "
        f"{spec_before} -> {spec_after}"
    )


OUT.mkdir(
    parents=True,
    exist_ok=False,
)

TRAIN = OUT / "train.list"
VAL = OUT / "val.list"

METADATA_CLEANED = (
    OUT
    / "metadata.list.cleaned"
)

SUMMARY = OUT / "summary.json"
README = OUT / "README.md"
SHA = OUT / "SHA256SUMS.txt"


TRAIN.write_text(
    "\n".join(
        train_cleaned
    )
    + "\n",
    encoding="utf-8",
)

VAL.write_text(
    "\n".join(
        val_cleaned
    )
    + "\n",
    encoding="utf-8",
)

METADATA_CLEANED.write_text(
    "\n".join(
        metadata_cleaned
    )
    + "\n",
    encoding="utf-8",
)


#
# Read-back validation.
#
train_rb = TRAIN.read_text(
    encoding="utf-8"
).splitlines()

val_rb = VAL.read_text(
    encoding="utf-8"
).splitlines()

meta_rb = (
    METADATA_CLEANED
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)


if train_rb != train_cleaned:
    raise RuntimeError(
        "train.list read-back mismatch"
    )

if val_rb != val_cleaned:
    raise RuntimeError(
        "val.list read-back mismatch"
    )

if meta_rb != (
    train_rb
    + val_rb
):
    raise RuntimeError(
        "clean metadata != train + val"
    )


#
# Raw WAV/split identity is still carried by
# first 3 fields and exact path.
#
train_wavs = {
    x.split("|", 1)[0]
    for x in train_rb
}

val_wavs = {
    x.split("|", 1)[0]
    for x in val_rb
}

if train_wavs & val_wavs:
    raise RuntimeError(
        "clean train/val WAV leakage"
    )


pc = np.asarray(
    phone_counts,
    dtype=np.float64,
)

wc = np.asarray(
    word2ph_counts,
    dtype=np.float64,
)


summary = {
    "artifact":
        "C2-F30-MeloTTS-no-BERT-v1",

    "frontend":
        "melo.text.cleaner.clean_text",

    "bert_generation":
        False,

    "spectrogram_generation":
        False,

    "format":
        (
            "wav|speaker|language|"
            "normalized_text|phones|tones|word2ph"
        ),

    "train_rows":
        len(train_rb),

    "val_rows":
        len(val_rb),

    "total_rows":
        len(meta_rb),

    "speaker":
        EXPECTED_SPEAKER,

    "language":
        EXPECTED_LANGUAGE,

    "normalized_text_changed_rows":
        normalized_changed,

    "unique_phone_symbols":
        len(unique_phones),

    "unknown_phone_symbols":
        dict(
            unknown_symbols
        ),

    "phone_count": {
        "min":
            int(pc.min()),

        "p50":
            float(
                np.percentile(
                    pc,
                    50,
                )
            ),

        "p95":
            float(
                np.percentile(
                    pc,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    pc,
                    99,
                )
            ),

        "max":
            int(pc.max()),
    },

    "word2ph_length": {
        "min":
            int(wc.min()),

        "p50":
            float(
                np.percentile(
                    wc,
                    50,
                )
            ),

        "p95":
            float(
                np.percentile(
                    wc,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    wc,
                    99,
                )
            ),

        "max":
            int(wc.max()),
    },

    "cache_counts": {
        "bert_before":
            bert_before,

        "bert_after":
            bert_after,

        "spec_before":
            spec_before,

        "spec_after":
            spec_after,
    },

    "source_sha256": {
        str(TRAIN_RAW):
            sha256_file(
                TRAIN_RAW
            ),

        str(VAL_RAW):
            sha256_file(
                VAL_RAW
            ),

        str(METADATA_RAW):
            sha256_file(
                METADATA_RAW
            ),
    },

    "melotts_source":
        str(MELO),
}


SUMMARY.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


README.write_text(
    """# C2-F30 MeloTTS no-BERT preprocessing v1

Frozen 7-field MeloTTS metadata for C2-F foundation.

Files:

- `train.list`: 12,257 frozen training rows
- `val.list`: 432 frozen validation rows
- `metadata.list.cleaned`: train + validation

Format:

    wav|speaker|language|normalized_text|phones|tones|word2ph

Russian frontend:

    melo.text.cleaner.clean_text()

No BERT feature generation is performed.
No spectrogram cache generation is performed.

YO-CPT remains physically multi-speaker; all rows use the
single synthetic C2-F foundation conditioning label only.
""",
    encoding="utf-8",
)


output_files = [
    TRAIN,
    VAL,
    METADATA_CLEANED,
    SUMMARY,
    README,
]


with SHA.open(
    "w",
    encoding="utf-8",
) as f:

    for path in output_files:
        f.write(
            f"{sha256_file(path)}  "
            f"{path.name}\n"
        )


for line in SHA.read_text(
    encoding="utf-8"
).splitlines():

    digest, filename = (
        line.split(
            "  ",
            1,
        )
    )

    actual = sha256_file(
        OUT / filename
    )

    if actual != digest:
        raise RuntimeError(
            f"SHA mismatch: {filename}"
        )


print()
print(
    "===== C2-F30 MELOTTS NO-BERT V1 ====="
)

print(
    "train rows:",
    len(train_rb),
)

print(
    "val rows:",
    len(val_rb),
)

print(
    "total rows:",
    len(meta_rb),
)

print()
print(
    "normalized text changed:",
    normalized_changed,
)

print(
    "unique phone symbols:",
    len(unique_phones),
)

print(
    "unknown phone symbols:",
    len(unknown_symbols),
)

print()
print(
    "phones min/p50/p95/p99/max:",
    int(pc.min()),
    round(
        float(
            np.percentile(
                pc,
                50,
            )
        ),
        2,
    ),
    round(
        float(
            np.percentile(
                pc,
                95,
            )
        ),
        2,
    ),
    round(
        float(
            np.percentile(
                pc,
                99,
            )
        ),
        2,
    ),
    int(pc.max()),
)

print()
print(
    "BERT files before/after:",
    bert_before,
    "/",
    bert_after,
)

print(
    "SPEC files before/after:",
    spec_before,
    "/",
    spec_after,
)

print()
print(
    "train/val WAV overlap:",
    len(
        train_wavs
        & val_wavs
    ),
)

print()
print(
    "train:",
    TRAIN,
)

print(
    "val:",
    VAL,
)

print(
    "summary:",
    SUMMARY,
)

print(
    "sha256:",
    SHA,
)

print()
print(
    "C2-F30 MELOTTS NO-BERT V1: PASS"
)
