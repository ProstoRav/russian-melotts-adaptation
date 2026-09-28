from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


PROJECT = Path(
    "${PROJECT_ROOT}"
)

MELO = Path(
    "${MELOTTS_ROOT}"
)

CSVROOT = (
    PROJECT
    / "datasets/tts/raw/dialogs"
)

META = CSVROOT / "metadata.csv"

WAVROOT = Path(
    "${DATA_ROOT}/tts/raw/dialogs/wavs"
)

FOUNDATION = (
    PROJECT
    / "artifacts/qc/"
      "spk298_prod_epoch_qc_v1/checkpoints/"
      "epoch_0027_step_0056645/"
      "G_56645.pth"
)

CONFIG = (
    PROJECT
    / "runs/"
      "spk298_from_englishv3_prod_b2w6_noeval_v1/"
      "model/config.json"
)

RUSSIAN_PY = (
    MELO
    / "melo/text/russian.py"
)

EXPECTED_ESPEAK = (
    PROJECT
    / "tools/"
      "espeak-ng-ru-stress-ab2eab9/"
      "build/src/espeak-ng"
)

OUT = (
    PROJECT
    / "artifacts/frozen/"
      "dialogs_np_source_stress_g2p_v1"
)

TMP = Path(
    str(OUT) + ".tmp"
)


EXPECTED_SHA = {
    "foundation":
        "7d03e90f9b94ecca28ab62246078ef3431708ba5392d129d7ddde5a2420cf2cc",

    "config":
        "9a38d22fa44ab8fd7249bf96fc5871a5bc6aa4fb06c4ef964846d5aba61622ad",

    "russian_py":
        "4cb05d35e213bbdab6c18fa597b6ff01cf9b05712ebe1414ed96d61abcf5451a",

    "espeak":
        "60874d253f54c88df7a838e76be8fd97a3109d097feb6c1b9eeed1cda3151f28",
}


GROUP_MAP = {
    "neutral":
        "dialogs_female_neutral",

    "happy":
        "dialogs_female_positive",

    "laughing":
        "dialogs_female_positive",
}

EXPECTED_COUNTS = {
    "dialogs_female_neutral":
        {
            "total": 4245,
            "train": 4197,
            "val": 21,
            "test": 27,
        },

    "dialogs_female_positive":
        {
            "total": 2299,
            "train": 2249,
            "val": 22,
            "test": 28,
        },
}

EXPECTED_TOTAL = {
    "total": 6544,
    "train": 6446,
    "val": 43,
    "test": 55,
}


RUSSIAN_VOWELS = (
    "АЕЁИОУЫЭЮЯ"
    "аеёиоуыэюя"
)

STRESS_PLUS_RE = re.compile(
    r"\+([" +
    RUSSIAN_VOWELS +
    r"])"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                4 * 1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def norm_rel(value: str) -> str:
    value = (
        value.strip()
        .replace("\\", "/")
    )

    while value.startswith("./"):
        value = value[2:]

    if value.startswith("wavs/"):
        value = value[5:]

    return value


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(
            csv.DictReader(
                f,
                delimiter="|",
            )
        )


def plus_to_acute(text: str):
    text = (
        text or ""
    )

    converted = 0

    def repl(match):
        nonlocal converted
        converted += 1

        return (
            match.group(1)
            + "\u0301"
        )

    result = STRESS_PLUS_RE.sub(
        repl,
        text,
    )

    result = unicodedata.normalize(
        "NFC",
        result,
    )

    return (
        result,
        converted,
        result.count("+"),
    )


def fail(message: str):
    raise RuntimeError(
        message
    )


print(
    "===== DIALOGS NP SOURCE + STRESS/G2P V1 =====",
    flush=True,
)


if OUT.exists():
    fail(
        f"refusing to overwrite frozen artifact: {OUT}"
    )

if TMP.exists():
    fail(
        f"stale tmp exists: {TMP}"
    )


#
# Immutable dependency checks.
#
for path, expected, label in [
    (
        FOUNDATION,
        EXPECTED_SHA["foundation"],
        "E27 foundation",
    ),
    (
        CONFIG,
        EXPECTED_SHA["config"],
        "foundation config",
    ),
    (
        RUSSIAN_PY,
        EXPECTED_SHA["russian_py"],
        "active russian.py",
    ),
    (
        EXPECTED_ESPEAK,
        EXPECTED_SHA["espeak"],
        "pinned espeak-ng",
    ),
]:
    if not path.is_file():
        fail(
            f"{label} missing: {path}"
        )

    actual = sha256(
        path
    )

    if actual != expected:
        fail(
            f"{label} SHA mismatch: "
            f"expected={expected} "
            f"actual={actual}"
        )

    print(
        f"{label} SHA256: PASS",
        flush=True,
    )


which_espeak = shutil.which(
    "espeak-ng"
)

if not which_espeak:
    fail(
        "espeak-ng not present in PATH"
    )

active_espeak = Path(
    which_espeak
).resolve()

if (
    sha256(active_espeak)
    != EXPECTED_SHA["espeak"]
):
    fail(
        "PATH resolves to unexpected espeak-ng: "
        f"{active_espeak}"
    )


version = subprocess.run(
    [
        str(active_espeak),
        "--version",
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()

print(
    f"espeak version: {version}",
    flush=True,
)

if "1.52.0.1" not in version:
    fail(
        "unexpected eSpeak NG version"
    )


#
# Load production vocabulary.
#
with CONFIG.open(
    "r",
    encoding="utf-8",
) as f:
    config = json.load(
        f
    )

symbols = set(
    config["symbols"]
)

add_blank = bool(
    config["data"]["add_blank"]
)


#
# Import exact active frontend.
#
sys.path.insert(
    0,
    str(MELO),
)

from melo.text.cleaner import clean_text


#
# Authoritative split map.
#
metadata = read_csv(
    META
)

split_by_path = {}

for split, filename in [
    ("train", "train.csv"),
    ("val", "val.csv"),
    ("test", "test.csv"),
]:
    for row in read_csv(
        CSVROOT / filename
    ):
        rel = norm_rel(
            row["audio_path"]
        )

        if rel in split_by_path:
            fail(
                f"duplicate split path: {rel}"
            )

        split_by_path[
            rel
        ] = split


print()
print(
    "===== PHYSICAL SOURCE SELECTION =====",
    flush=True,
)


selected = []

physical_counts = defaultdict(
    Counter
)

source_cells = defaultdict(
    Counter
)

physical_seconds = defaultdict(
    float
)


for row in metadata:
    source_speaker = (
        row["speaker_id"]
        .strip()
    )

    emotion = (
        row["emotion"]
        .strip()
        .lower()
    )

    if source_speaker not in {
        "M",
        "S",
    }:
        continue

    pseudo = GROUP_MAP.get(
        emotion
    )

    if pseudo is None:
        continue

    rel = norm_rel(
        row["audio_path"]
    )

    wav = WAVROOT / rel

    #
    # Published physical WAVs only.
    #
    if not wav.is_file():
        continue

    split = split_by_path.get(
        rel
    )

    if split not in {
        "train",
        "val",
        "test",
    }:
        fail(
            f"missing split: {rel}"
        )

    info = sf.info(
        str(wav)
    )

    if (
        info.samplerate != 44100
        or info.channels != 1
        or info.subtype != "PCM_16"
        or info.frames <= 0
    ):
        fail(
            f"unexpected WAV geometry: "
            f"{wav}: "
            f"{info.samplerate} Hz, "
            f"{info.channels} ch, "
            f"{info.subtype}, "
            f"{info.frames} frames"
        )

    duration = (
        info.frames
        / float(info.samplerate)
    )

    physical_counts[
        pseudo
    ]["total"] += 1

    physical_counts[
        pseudo
    ][split] += 1

    source_cells[
        (source_speaker, emotion)
    ]["count"] += 1

    source_cells[
        (source_speaker, emotion)
    ][split] += 1

    physical_seconds[
        pseudo
    ] += duration

    selected.append({
        "audio_rel":
            rel,

        "audio_path":
            str(wav),

        "source_speaker":
            source_speaker,

        "source_emotion":
            emotion,

        "pseudo_speaker":
            pseudo,

        "split":
            split,

        "duration_sec":
            duration,

        "wav_frames":
            int(info.frames),

        "wav_size_bytes":
            wav.stat().st_size,

        "text":
            row.get("text") or "",

        "accent_text":
            row.get("accent_text") or "",
    })


for pseudo, expected in (
    EXPECTED_COUNTS.items()
):
    actual = physical_counts[
        pseudo
    ]

    for key, wanted in (
        expected.items()
    ):
        got = actual[key]

        if got != wanted:
            fail(
                f"{pseudo} {key}: "
                f"expected={wanted}, "
                f"actual={got}"
            )


totals = Counter()

for pseudo in EXPECTED_COUNTS:
    totals.update(
        physical_counts[pseudo]
    )


for key, wanted in (
    EXPECTED_TOTAL.items()
):
    if totals[key] != wanted:
        fail(
            f"total {key}: "
            f"expected={wanted}, "
            f"actual={totals[key]}"
        )


print(
    "pseudo_speaker\tcount\thours\t"
    "train\tval\ttest",
    flush=True,
)

for pseudo in [
    "dialogs_female_neutral",
    "dialogs_female_positive",
]:
    c = physical_counts[
        pseudo
    ]

    print(
        f"{pseudo}\t"
        f"{c['total']}\t"
        f"{physical_seconds[pseudo]/3600:.9f}\t"
        f"{c['train']}\t"
        f"{c['val']}\t"
        f"{c['test']}",
        flush=True,
    )


print(
    f"SELECTED TOTAL: "
    f"{len(selected)} "
    f"{sum(physical_seconds.values())/3600:.9f} h",
    flush=True,
)


#
# Stable deterministic row order.
#
split_order = {
    "train": 0,
    "val": 1,
    "test": 2,
}

selected.sort(
    key=lambda x: (
        split_order[
            x["split"]
        ],
        x["pseudo_speaker"],
        x["audio_rel"],
    )
)


print()
print(
    "===== STRESS + G2P SCAN =====",
    flush=True,
)


stress_plus_total = 0
existing_acute_total = 0
final_acute_total = 0

accent_text_missing = 0
accent_text_equal_text = 0

nonstress_plus_rows = []
accent_lost_rows = []
g2p_errors = []
unknown_phone_rows = []

unique_phones = set()

max_phone_tokens = 0
max_positions_add_blank = 0
max_row = None

generated = []


for idx, row in enumerate(
    selected,
    1,
):
    accent_text = (
        row["accent_text"]
        .strip()
    )

    plain_text = (
        row["text"]
        .strip()
    )

    if accent_text:
        source_text = (
            accent_text
        )
        text_source = (
            "accent_text"
        )
    else:
        source_text = (
            plain_text
        )
        text_source = (
            "text_fallback"
        )

        accent_text_missing += 1


    if (
        accent_text
        and accent_text == plain_text
    ):
        accent_text_equal_text += 1


    existing_acute = (
        source_text.count(
            "\u0301"
        )
    )

    (
        stress_text,
        converted_plus,
        remaining_plus,
    ) = plus_to_acute(
        source_text
    )


    stress_plus_total += (
        converted_plus
    )

    existing_acute_total += (
        existing_acute
    )

    final_acute = (
        stress_text.count(
            "\u0301"
        )
    )

    final_acute_total += (
        final_acute
    )


    if remaining_plus:
        nonstress_plus_rows.append(
            {
                "audio_rel":
                    row["audio_rel"],

                "source_text":
                    source_text,

                "stress_text":
                    stress_text,

                "remaining_plus":
                    remaining_plus,
            }
        )


    try:
        (
            norm_text,
            phones,
            tones,
            word2ph,
        ) = clean_text(
            stress_text,
            "RU",
        )

    except Exception as exc:
        g2p_errors.append(
            {
                "audio_rel":
                    row["audio_rel"],

                "error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            }
        )

        continue


    normalized_accents = (
        norm_text.count(
            "\u0301"
        )
    )

    if (
        normalized_accents
        < final_acute
    ):
        accent_lost_rows.append(
            {
                "audio_rel":
                    row["audio_rel"],

                "input_accents":
                    final_acute,

                "normalized_accents":
                    normalized_accents,

                "stress_text":
                    stress_text,

                "norm_text":
                    norm_text,
            }
        )


    if not (
        len(phones)
        == len(tones)
        == sum(word2ph)
    ):
        fail(
            f"G2P shape invariant failed: "
            f"{row['audio_rel']}"
        )


    unknown = sorted(
        {
            ph
            for ph in phones
            if ph not in symbols
        }
    )

    if unknown:
        unknown_phone_rows.append(
            {
                "audio_rel":
                    row["audio_rel"],

                "unknown":
                    unknown,
            }
        )


    unique_phones.update(
        phones
    )

    phone_tokens = len(
        phones
    )

    positions = (
        (
            2 * phone_tokens
            + 1
        )
        if add_blank
        else phone_tokens
    )


    if phone_tokens > max_phone_tokens:
        max_phone_tokens = (
            phone_tokens
        )

        max_positions_add_blank = (
            positions
        )

        max_row = (
            row["audio_rel"]
        )


    generated.append({
        **row,

        "text_source":
            text_source,

        "stress_text":
            stress_text,

        "converted_plus_markers":
            converted_plus,

        "existing_acute_markers":
            existing_acute,

        "final_acute_markers":
            final_acute,

        "norm_text":
            norm_text,

        "phones":
            phones,

        "tones":
            tones,

        "word2ph":
            word2ph,

        "phone_tokens":
            phone_tokens,

        "positions_after_blank":
            positions,
    })


    if (
        idx == 1
        or idx % 250 == 0
        or idx == len(selected)
    ):
        print(
            f"[{idx:04d}/{len(selected):04d}] "
            f"g2p_ok={len(generated)}",
            flush=True,
        )


print()
print(
    "===== STRESS/G2P SUMMARY =====",
    flush=True,
)

print(
    "selected_rows:",
    len(selected),
    flush=True,
)

print(
    "generated_rows:",
    len(generated),
    flush=True,
)

print(
    "accent_text_missing:",
    accent_text_missing,
    flush=True,
)

print(
    "accent_text_equal_plain_text:",
    accent_text_equal_text,
    flush=True,
)

print(
    "converted_plus_stress_markers:",
    stress_plus_total,
    flush=True,
)

print(
    "preexisting_U+0301_markers:",
    existing_acute_total,
    flush=True,
)

print(
    "final_U+0301_markers:",
    final_acute_total,
    flush=True,
)

print(
    "rows_with_nonstress_plus:",
    len(nonstress_plus_rows),
    flush=True,
)

print(
    "accent_lost_rows:",
    len(accent_lost_rows),
    flush=True,
)

print(
    "g2p_errors:",
    len(g2p_errors),
    flush=True,
)

print(
    "unknown_phone_rows:",
    len(unknown_phone_rows),
    flush=True,
)

print(
    "unique_phone_symbols:",
    len(unique_phones),
    flush=True,
)

print(
    "max_phone_tokens:",
    max_phone_tokens,
    flush=True,
)

print(
    "max_positions_add_blank:",
    max_positions_add_blank,
    flush=True,
)

print(
    "max_phone_row:",
    max_row,
    flush=True,
)


#
# Blocking invariants.
#
if len(generated) != len(selected):
    fail(
        "not every selected row passed G2P"
    )

if accent_lost_rows:
    fail(
        f"stress accents lost in "
        f"{len(accent_lost_rows)} rows"
    )

if unknown_phone_rows:
    fail(
        f"unknown phones in "
        f"{len(unknown_phone_rows)} rows"
    )


#
# A plus not used as +<Russian vowel> is not silently
# reinterpreted. Persist report and stop finalization later.
#
status = (
    "PASS"
    if not nonstress_plus_rows
    else "REVIEW_NONSTRESS_PLUS"
)


TMP.mkdir(
    parents=True,
    exist_ok=False,
)


#
# Complete frozen row catalog.
#
with (
    TMP / "catalog.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in generated:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
            + "\n"
        )


with (
    TMP / "selected_audio.tsv"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "split\tpseudo_speaker\t"
        "source_speaker\t"
        "source_emotion\t"
        "audio_rel\tduration_sec\t"
        "frames\tsize_bytes\n"
    )

    for row in generated:
        f.write(
            f"{row['split']}\t"
            f"{row['pseudo_speaker']}\t"
            f"{row['source_speaker']}\t"
            f"{row['source_emotion']}\t"
            f"{row['audio_rel']}\t"
            f"{row['duration_sec']:.9f}\t"
            f"{row['wav_frames']}\t"
            f"{row['wav_size_bytes']}\n"
        )


with (
    TMP / "source_cells.tsv"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "source_speaker\t"
        "source_emotion\t"
        "count\ttrain\tval\ttest\n"
    )

    for (
        speaker,
        emotion,
    ) in sorted(source_cells):
        c = source_cells[
            (speaker, emotion)
        ]

        f.write(
            f"{speaker}\t"
            f"{emotion}\t"
            f"{c['count']}\t"
            f"{c['train']}\t"
            f"{c['val']}\t"
            f"{c['test']}\n"
        )


for filename, rows in [
    (
        "nonstress_plus_rows.json",
        nonstress_plus_rows,
    ),
    (
        "accent_lost_rows.json",
        accent_lost_rows,
    ),
    (
        "g2p_errors.json",
        g2p_errors,
    ),
    (
        "unknown_phone_rows.json",
        unknown_phone_rows,
    ),
]:
    (
        TMP / filename
    ).write_text(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


input_hashes = {
    "metadata.csv":
        sha256(
            CSVROOT / "metadata.csv"
        ),

    "train.csv":
        sha256(
            CSVROOT / "train.csv"
        ),

    "val.csv":
        sha256(
            CSVROOT / "val.csv"
        ),

    "test.csv":
        sha256(
            CSVROOT / "test.csv"
        ),

    "foundation_G_56645":
        sha256(
            FOUNDATION
        ),

    "config":
        sha256(
            CONFIG
        ),

    "russian.py":
        sha256(
            RUSSIAN_PY
        ),

    "espeak-ng":
        sha256(
            EXPECTED_ESPEAK
        ),
}


report = {
    "status":
        status,

    "artifact":
        "dialogs_np_source_stress_g2p_v1",

    "foundation": {
        "checkpoint":
            str(FOUNDATION),

        "epoch":
            27,

        "global_step":
            56645,

        "sha256":
            EXPECTED_SHA[
                "foundation"
            ],
    },

    "source": {
        "csv_root":
            str(CSVROOT),

        "wav_root":
            str(WAVROOT),

        "policy":
            (
                "published physical WAVs only; "
                "M+S only; "
                "neutral -> dialogs_female_neutral; "
                "happy/laughing -> dialogs_female_positive"
            ),
    },

    "speaker_plan": {
        "dialogs_female_neutral":
            0,

        "dialogs_female_positive":
            1,

        "prestart_embedding_policy":
            (
                "both emb_g rows will be exact copies "
                "of E27 spk_298 emb_g row 0"
            ),
    },

    "input_sha256":
        input_hashes,

    "frontend": {
        "language":
            "RU",

        "espeak_version":
            version,

        "plus_stress_policy":
            (
                "+<Russian vowel> -> "
                "<vowel> + U+0301"
            ),

        "bert":
            "disabled in later training contract",
    },

    "counts": {
        pseudo: dict(
            physical_counts[
                pseudo
            ]
        )
        for pseudo
        in EXPECTED_COUNTS
    },

    "hours": {
        pseudo:
            physical_seconds[
                pseudo
            ] / 3600
        for pseudo
        in EXPECTED_COUNTS
    },

    "stress_g2p": {
        "selected_rows":
            len(selected),

        "generated_rows":
            len(generated),

        "accent_text_missing":
            accent_text_missing,

        "accent_text_equal_plain_text":
            accent_text_equal_text,

        "converted_plus_stress_markers":
            stress_plus_total,

        "preexisting_U+0301_markers":
            existing_acute_total,

        "final_U+0301_markers":
            final_acute_total,

        "rows_with_nonstress_plus":
            len(nonstress_plus_rows),

        "accent_lost_rows":
            len(accent_lost_rows),

        "g2p_errors":
            len(g2p_errors),

        "unknown_phone_rows":
            len(unknown_phone_rows),

        "unique_phone_symbols":
            sorted(
                unique_phones
            ),

        "max_phone_tokens":
            max_phone_tokens,

        "max_positions_add_blank":
            max_positions_add_blank,

        "max_phone_row":
            max_row,
    },
}


(
    TMP / "report.json"
).write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


with (
    TMP / "SHA256SUMS.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    for p in sorted(
        TMP.iterdir()
    ):
        if (
            p.is_file()
            and p.name
            != "SHA256SUMS.txt"
        ):
            f.write(
                f"{sha256(p)}  "
                f"{p.name}\n"
            )


#
# Do not freeze ambiguous '+' semantics.
#
if nonstress_plus_rows:
    print()
    print(
        "DIALOGS NP STRESS/G2P: "
        "REVIEW REQUIRED",
        flush=True,
    )

    print(
        f"rows_with_nonstress_plus="
        f"{len(nonstress_plus_rows)}",
        flush=True,
    )

    print(
        f"temporary report retained at: {TMP}",
        flush=True,
    )

    raise SystemExit(30)


TMP.rename(
    OUT
)


print()
print(
    "frozen:",
    OUT,
    flush=True,
)

print()
print(
    "DIALOGS NP SOURCE + STRESS/G2P V1: PASS",
    flush=True,
)
