from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


PROJECT = Path.home() / "projects/ru_melotts_research"

SPLIT_ROOT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_train_val_split_v1"
)

TRAIN_SRC = (
    SPLIT_ROOT
    / "manifest_train.parquet"
)

VAL_SRC = (
    SPLIT_ROOT
    / "manifest_val.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_melotts_raw_v1"
)

EXPECTED_TRAIN_ROWS = 12257
EXPECTED_VAL_ROWS = 432
EXPECTED_TOTAL_ROWS = 12689

EXPECTED_TRAIN_HOURS = 28.3588043171
EXPECTED_VAL_HOURS = 1.0001173727
EXPECTED_TOTAL_HOURS = 29.3589216898

EXPECTED_SPEAKER_ID = 0
EXPECTED_SPEAKER_NAME = "c2_f30_generic_ru"
LANGUAGE = "RU"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def physical_hours(rows):
    return (
        sum(
            float(r["physical_duration"])
            for r in rows
        )
        / 3600.0
    )


def validate_text(text, utt):
    if text is None:
        raise RuntimeError(
            f"{utt}: training_text is None"
        )

    text = str(text).strip()

    if not text:
        raise RuntimeError(
            f"{utt}: empty training_text"
        )

    #
    # MeloTTS raw metadata uses "|" as delimiter.
    # Do not silently modify canonical training text.
    #
    if "|" in text:
        raise RuntimeError(
            f"{utt}: training_text contains |"
        )

    if "\n" in text or "\r" in text:
        raise RuntimeError(
            f"{utt}: training_text contains newline"
        )

    return text


def validate_rows(rows, expected_split):
    for r in rows:

        utt = r["utt_id"]

        if r["split"] != expected_split:
            raise RuntimeError(
                f"{utt}: split={r['split']} "
                f"expected={expected_split}"
            )

        if (
            int(r["foundation_speaker_id"])
            != EXPECTED_SPEAKER_ID
        ):
            raise RuntimeError(
                f"{utt}: unexpected "
                "foundation_speaker_id"
            )

        if (
            str(r["foundation_speaker_name"])
            != EXPECTED_SPEAKER_NAME
        ):
            raise RuntimeError(
                f"{utt}: unexpected "
                "foundation_speaker_name"
            )

        validate_text(
            r["training_text"],
            utt,
        )

        wav = Path(
            r["wav_path"]
        )

        if not wav.is_absolute():
            raise RuntimeError(
                f"{utt}: wav_path not absolute: {wav}"
            )

        if not wav.is_file():
            raise RuntimeError(
                f"{utt}: WAV missing: {wav}"
            )

        if "|" in str(wav):
            raise RuntimeError(
                f"{utt}: WAV path contains |"
            )


def raw_line(r):
    text = validate_text(
        r["training_text"],
        r["utt_id"],
    )

    wav = Path(
        r["wav_path"]
    )

    return (
        f"{wav}"
        f"|{EXPECTED_SPEAKER_NAME}"
        f"|{LANGUAGE}"
        f"|{text}"
    )


if OUT.exists():
    raise RuntimeError(
        f"refusing to overwrite frozen artifact: {OUT}"
    )


train = (
    pq.read_table(
        TRAIN_SRC
    )
    .to_pylist()
)

val = (
    pq.read_table(
        VAL_SRC
    )
    .to_pylist()
)


if len(train) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        f"train rows={len(train)}"
    )

if len(val) != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        f"val rows={len(val)}"
    )

if (
    len(train)
    + len(val)
    != EXPECTED_TOTAL_ROWS
):
    raise RuntimeError(
        "total row count mismatch"
    )


train_h = physical_hours(train)
val_h = physical_hours(val)
total_h = train_h + val_h


for actual, expected, name in (
    (
        train_h,
        EXPECTED_TRAIN_HOURS,
        "train hours",
    ),
    (
        val_h,
        EXPECTED_VAL_HOURS,
        "val hours",
    ),
    (
        total_h,
        EXPECTED_TOTAL_HOURS,
        "total hours",
    ),
):
    if abs(actual - expected) > 1e-9:
        raise RuntimeError(
            f"{name}: {actual} != {expected}"
        )


validate_rows(
    train,
    "train",
)

validate_rows(
    val,
    "val",
)


train_utts = {
    r["utt_id"]
    for r in train
}

val_utts = {
    r["utt_id"]
    for r in val
}

if train_utts & val_utts:
    raise RuntimeError(
        "train/val utt_id leakage"
    )

if (
    len(train_utts)
    != len(train)
    or len(val_utts)
    != len(val)
):
    raise RuntimeError(
        "duplicate utt_id"
    )


train_wavs = {
    str(r["wav_path"])
    for r in train
}

val_wavs = {
    str(r["wav_path"])
    for r in val
}

if train_wavs & val_wavs:
    raise RuntimeError(
        "train/val WAV leakage"
    )

if (
    len(train_wavs)
    + len(val_wavs)
    != EXPECTED_TOTAL_ROWS
):
    raise RuntimeError(
        "duplicate WAV path"
    )


train_lines = [
    raw_line(r)
    for r in train
]

val_lines = [
    raw_line(r)
    for r in val
]

metadata_lines = (
    train_lines
    + val_lines
)


#
# There must be exactly four raw metadata fields.
#
for i, line in enumerate(
    metadata_lines,
    start=1,
):
    if len(line.split("|")) != 4:
        raise RuntimeError(
            f"metadata line {i}: "
            "expected exactly 4 fields"
        )


OUT.mkdir(
    parents=True,
    exist_ok=False,
)

METADATA = (
    OUT
    / "metadata.list"
)

TRAIN_RAW = (
    OUT
    / "train_raw.list"
)

VAL_RAW = (
    OUT
    / "val_raw.list"
)

SUMMARY = (
    OUT
    / "summary.json"
)

README = (
    OUT
    / "README.md"
)

SHA = (
    OUT
    / "SHA256SUMS.txt"
)


METADATA.write_text(
    "\n".join(
        metadata_lines
    )
    + "\n",
    encoding="utf-8",
)

TRAIN_RAW.write_text(
    "\n".join(
        train_lines
    )
    + "\n",
    encoding="utf-8",
)

VAL_RAW.write_text(
    "\n".join(
        val_lines
    )
    + "\n",
    encoding="utf-8",
)


#
# Read back line counts.
#
metadata_read = (
    METADATA
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

train_read = (
    TRAIN_RAW
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)

val_read = (
    VAL_RAW
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)


if len(metadata_read) != EXPECTED_TOTAL_ROWS:
    raise RuntimeError(
        f"metadata.list rows="
        f"{len(metadata_read)}"
    )

if len(train_read) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        f"train_raw.list rows="
        f"{len(train_read)}"
    )

if len(val_read) != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        f"val_raw.list rows="
        f"{len(val_read)}"
    )


if metadata_read != (
    train_read
    + val_read
):
    raise RuntimeError(
        "metadata.list != train_raw + val_raw"
    )


summary = {
    "artifact":
        "C2-F30-MeloTTS-raw-v1",

    "format":
        "absolute_wav_path|speaker|language|text",

    "language":
        LANGUAGE,

    "conditioning": {
        "speaker_name":
            EXPECTED_SPEAKER_NAME,

        "speaker_id_after_preprocessing":
            EXPECTED_SPEAKER_ID,

        "n_speakers":
            1,

        "note":
            (
                "YO-CPT audio remains physically "
                "multi-speaker. This is a synthetic "
                "foundation conditioning label only."
            ),
    },

    "train": {
        "rows":
            len(train),

        "physical_hours":
            train_h,
    },

    "val": {
        "rows":
            len(val),

        "physical_hours":
            val_h,
    },

    "total": {
        "rows":
            len(metadata_lines),

        "physical_hours":
            total_h,
    },

    "split_policy":
        (
            "Frozen connected-component split by "
            "source_video_id + known global_spk_id."
        ),

    "source_sha256": {
        str(TRAIN_SRC):
            sha256_file(
                TRAIN_SRC
            ),

        str(VAL_SRC):
            sha256_file(
                VAL_SRC
            ),
    },

    "preprocessing_policy":
        (
            "Raw export only. No G2P, BERT, "
            "spectrogram generation, resampling, "
            "or WAV modification performed."
        ),
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
    """# C2-F30 MeloTTS raw v1

Frozen raw MeloTTS metadata export for C2-F foundation.

Format:

    absolute_wav_path|c2_f30_generic_ru|RU|training_text

Counts:

- train: 12,257
- validation: 432
- total: 12,689

The YO-CPT recordings are physically multi-speaker.
`c2_f30_generic_ru` is only a synthetic conditioning label
for the Russian foundation adaptation.

This artifact contains no generated BERT or spectrogram cache.
The frozen train/validation split is preserved exactly.
""",
    encoding="utf-8",
)


output_files = [
    METADATA,
    TRAIN_RAW,
    VAL_RAW,
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
        OUT
        / filename
    )

    if actual != digest:
        raise RuntimeError(
            f"SHA mismatch: {filename}"
        )


print(
    "===== C2-F30 MELOTTS RAW V1 ====="
)

print(
    "format:",
    "absolute_wav_path|speaker|RU|text",
)

print()
print(
    "train rows:",
    len(train_lines),
)

print(
    "train hours:",
    round(
        train_h,
        10,
    ),
)

print()
print(
    "val rows:",
    len(val_lines),
)

print(
    "val hours:",
    round(
        val_h,
        10,
    ),
)

print()
print(
    "total rows:",
    len(metadata_lines),
)

print(
    "total hours:",
    round(
        total_h,
        10,
    ),
)

print()
print(
    "speaker:",
    EXPECTED_SPEAKER_NAME,
)

print(
    "language:",
    LANGUAGE,
)

print()
print(
    "train/val utt overlap:",
    len(
        train_utts
        & val_utts
    ),
)

print(
    "train/val WAV overlap:",
    len(
        train_wavs
        & val_wavs
    ),
)

print()
print(
    "metadata:",
    METADATA,
)

print(
    "train raw:",
    TRAIN_RAW,
)

print(
    "val raw:",
    VAL_RAW,
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
    "C2-F30 MELOTTS RAW V1: PASS"
)
