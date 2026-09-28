from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(
    "${PROJECT_ROOT}"
)

SRC = (
    PROJECT
    / "artifacts/frozen/"
      "dialogs_np_source_stress_g2p_v1"
)

CATALOG = (
    SRC / "catalog.jsonl"
)

OUT = (
    PROJECT
    / "artifacts/frozen/"
      "dialogs_np_melotts_no_bert_stress_v1"
)

TMP = Path(
    str(OUT) + ".tmp"
)

MAX_TEXT_LEN = 512

SPEAKER_MAP = {
    "dialogs_female_neutral": 0,
    "dialogs_female_positive": 1,
}

EXPECTED_SELECTED_BEFORE_FILTER = {
    "train": 6446,
    "val": 43,
    "test": 55,
    "total": 6544,
}

EXPECTED_FINAL = {
    "train": 6433,
    "val": 43,
    "test": 55,
    "total": 6531,
}

EXPECTED_EXCLUDED = {
    "train": 13,
    "val": 0,
    "test": 0,
    "total": 13,
}

EXPECTED_FINAL_BY_SPEAKER = {
    "dialogs_female_neutral": {
        "train": 4191,
        "val": 21,
        "test": 27,
        "total": 4239,
    },

    "dialogs_female_positive": {
        "train": 2242,
        "val": 22,
        "test": 28,
        "total": 2292,
    },
}


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


def fail(message: str):
    raise RuntimeError(
        message
    )


def validate_field(
    value: str,
    label: str,
    audio_rel: str,
):
    if "|" in value:
        fail(
            f"{audio_rel}: pipe in {label}"
        )

    if "\n" in value or "\r" in value:
        fail(
            f"{audio_rel}: newline in {label}"
        )


print(
    "===== DIALOGS NP MELOTTS LISTS V1 =====",
    flush=True,
)


if OUT.exists():
    fail(
        f"refusing to overwrite frozen artifact: {OUT}"
    )

if TMP.exists():
    fail(
        f"stale temp artifact: {TMP}"
    )

if not CATALOG.is_file():
    fail(
        f"catalog missing: {CATALOG}"
    )


rows = []

with CATALOG.open(
    "r",
    encoding="utf-8",
) as f:

    for line_no, line in enumerate(
        f,
        1,
    ):
        if not line.strip():
            continue

        row = json.loads(
            line
        )

        rows.append(
            row
        )


print(
    f"catalog rows: {len(rows)}",
    flush=True,
)

if len(rows) != 6544:
    fail(
        f"expected 6544 catalog rows, "
        f"got {len(rows)}"
    )


#
# Global uniqueness.
#
audio_rels = [
    row["audio_rel"]
    for row in rows
]

if len(set(audio_rels)) != len(audio_rels):
    fail(
        "duplicate audio_rel in source catalog"
    )


before = Counter(
    row["split"]
    for row in rows
)

before["total"] = len(
    rows
)

for key, wanted in (
    EXPECTED_SELECTED_BEFORE_FILTER.items()
):
    got = before[key]

    if got != wanted:
        fail(
            f"pre-filter {key}: "
            f"expected={wanted} got={got}"
        )


kept = []
excluded = []


for row in rows:
    phones = row["phones"]
    tones = row["tones"]
    word2ph = row["word2ph"]

    if not (
        len(phones)
        == len(tones)
        == sum(word2ph)
    ):
        fail(
            f"G2P geometry mismatch: "
            f"{row['audio_rel']}"
        )

    if (
        len(phones)
        != int(
            row["phone_tokens"]
        )
    ):
        fail(
            f"phone_tokens mismatch: "
            f"{row['audio_rel']}"
        )

    if (
        row["pseudo_speaker"]
        not in SPEAKER_MAP
    ):
        fail(
            f"unknown pseudo speaker: "
            f"{row['pseudo_speaker']}"
        )

    if row["split"] not in {
        "train",
        "val",
        "test",
    }:
        fail(
            f"unknown split: "
            f"{row['split']}"
        )

    audio = Path(
        row["audio_path"]
    )

    if not audio.is_file():
        fail(
            f"physical WAV disappeared: "
            f"{audio}"
        )

    if (
        int(row["phone_tokens"])
        > MAX_TEXT_LEN
    ):
        excluded.append(
            row
        )
    else:
        kept.append(
            row
        )


#
# Validate exclusion geometry.
#
excluded_counts = Counter(
    row["split"]
    for row in excluded
)

excluded_counts["total"] = len(
    excluded
)

for key, wanted in (
    EXPECTED_EXCLUDED.items()
):
    got = excluded_counts[key]

    if got != wanted:
        fail(
            f"excluded {key}: "
            f"expected={wanted} got={got}"
        )


if any(
    row["split"] != "train"
    for row in excluded
):
    fail(
        "non-train sample excluded by max_text_len"
    )


final_counts = Counter(
    row["split"]
    for row in kept
)

final_counts["total"] = len(
    kept
)

for key, wanted in (
    EXPECTED_FINAL.items()
):
    got = final_counts[key]

    if got != wanted:
        fail(
            f"final {key}: "
            f"expected={wanted} got={got}"
        )


by_speaker = defaultdict(
    Counter
)

for row in kept:
    speaker = row[
        "pseudo_speaker"
    ]

    split = row[
        "split"
    ]

    by_speaker[
        speaker
    ][split] += 1

    by_speaker[
        speaker
    ]["total"] += 1


for speaker, expected in (
    EXPECTED_FINAL_BY_SPEAKER.items()
):
    for key, wanted in expected.items():
        got = by_speaker[
            speaker
        ][key]

        if got != wanted:
            fail(
                f"{speaker} {key}: "
                f"expected={wanted} "
                f"got={got}"
            )


#
# Stable ordering independent of source catalog order.
#
kept.sort(
    key=lambda row: (
        row["split"],
        SPEAKER_MAP[
            row["pseudo_speaker"]
        ],
        row["audio_rel"],
    )
)

excluded.sort(
    key=lambda row: (
        row["split"],
        row["pseudo_speaker"],
        -int(
            row["phone_tokens"]
        ),
        row["audio_rel"],
    )
)


TMP.mkdir(
    parents=True,
    exist_ok=False,
)


#
# Speaker map.
#
(
    TMP / "speaker_map.json"
).write_text(
    json.dumps(
        {
            "spk2id":
                SPEAKER_MAP,

            "n_speakers":
                2,

            "policy":
                (
                    "pseudo-speaker conditioning; "
                    "no factorized style field"
                ),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


#
# Build exact 7-field MeloTTS lines.
#
split_handles = {}

try:
    for split in (
        "train",
        "val",
        "test",
    ):
        split_handles[
            split
        ] = (
            TMP / f"{split}.list"
        ).open(
            "w",
            encoding="utf-8",
        )


    for row in kept:
        audio_path = str(
            Path(
                row["audio_path"]
            ).resolve()
        )

        speaker = row[
            "pseudo_speaker"
        ]

        language = "RU"

        #
        # norm_text is the text from which phones,
        # tones and word2ph were produced.
        #
        text = row[
            "norm_text"
        ]

        phones = " ".join(
            str(x)
            for x in row["phones"]
        )

        tones = " ".join(
            str(int(x))
            for x in row["tones"]
        )

        word2ph = " ".join(
            str(int(x))
            for x in row["word2ph"]
        )


        for value, label in [
            (
                audio_path,
                "audio_path",
            ),
            (
                speaker,
                "speaker",
            ),
            (
                language,
                "language",
            ),
            (
                text,
                "text",
            ),
            (
                phones,
                "phones",
            ),
            (
                tones,
                "tones",
            ),
            (
                word2ph,
                "word2ph",
            ),
        ]:
            validate_field(
                value,
                label,
                row["audio_rel"],
            )


        fields = [
            audio_path,
            speaker,
            language,
            text,
            phones,
            tones,
            word2ph,
        ]

        if len(fields) != 7:
            fail(
                "internal field count error"
            )


        split_handles[
            row["split"]
        ].write(
            "|".join(fields)
            + "\n"
        )

finally:
    for handle in (
        split_handles.values()
    ):
        handle.close()


#
# Explicit exclusion report.
#
with (
    TMP
    / "excluded_max_text_len.tsv"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "split\tpseudo_speaker\t"
        "audio_rel\tphone_tokens\t"
        "positions_after_blank\t"
        "duration_sec\tstress_text\n"
    )

    for row in excluded:
        text = (
            row["stress_text"]
            .replace("\t", " ")
            .replace("\n", " ")
        )

        f.write(
            f"{row['split']}\t"
            f"{row['pseudo_speaker']}\t"
            f"{row['audio_rel']}\t"
            f"{row['phone_tokens']}\t"
            f"{row['positions_after_blank']}\t"
            f"{row['duration_sec']:.9f}\t"
            f"{text}\n"
        )


#
# Full selected inventory and WAV hashes.
#
print()
print(
    "===== HASHING 6531 USED WAVS =====",
    flush=True,
)


total_seconds = 0.0
speaker_seconds = defaultdict(
    float
)

with (
    TMP / "wav_sha256.tsv"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "split\tpseudo_speaker\t"
        "audio_rel\tduration_sec\t"
        "size_bytes\tsha256\n"
    )


    for idx, row in enumerate(
        kept,
        1,
    ):
        path = Path(
            row["audio_path"]
        )

        digest = sha256(
            path
        )

        duration = float(
            row["duration_sec"]
        )

        total_seconds += duration

        speaker_seconds[
            row["pseudo_speaker"]
        ] += duration


        f.write(
            f"{row['split']}\t"
            f"{row['pseudo_speaker']}\t"
            f"{row['audio_rel']}\t"
            f"{duration:.9f}\t"
            f"{path.stat().st_size}\t"
            f"{digest}\n"
        )


        if (
            idx == 1
            or idx % 500 == 0
            or idx == len(kept)
        ):
            print(
                f"[{idx:04d}/{len(kept):04d}] "
                f"WAV hashes complete",
                flush=True,
            )


#
# Validate written list line counts and NF=7.
#
for split in (
    "train",
    "val",
    "test",
):
    path = (
        TMP / f"{split}.list"
    )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    if len(lines) != EXPECTED_FINAL[
        split
    ]:
        fail(
            f"{split}.list line count: "
            f"{len(lines)}"
        )

    for line_no, line in enumerate(
        lines,
        1,
    ):
        if len(
            line.split("|")
        ) != 7:
            fail(
                f"{split}.list:"
                f"{line_no} NF != 7"
            )


summary = {
    "artifact":
        "dialogs_np_melotts_no_bert_stress_v1",

    "source_artifact":
        str(SRC),

    "source_catalog_sha256":
        sha256(CATALOG),

    "max_text_len":
        MAX_TEXT_LEN,

    "speaker_map":
        SPEAKER_MAP,

    "metadata_fields":
        [
            "audio_path",
            "speaker",
            "language",
            "text",
            "phones",
            "tones",
            "word2ph",
        ],

    "before_filter":
        dict(before),

    "excluded_max_text_len":
        dict(
            excluded_counts
        ),

    "final":
        dict(
            final_counts
        ),

    "final_by_speaker": {
        speaker:
            dict(
                by_speaker[
                    speaker
                ]
            )
        for speaker
        in SPEAKER_MAP
    },

    "final_hours": {
        "total":
            total_seconds
            / 3600,

        **{
            speaker:
                seconds
                / 3600
            for speaker, seconds
            in speaker_seconds.items()
        },
    },

    "policy": {
        "source_audio":
            (
                "physically published "
                "readable WAVs only"
            ),

        "pseudo_speakers":
            (
                "neutral -> SID 0; "
                "happy/laughing -> SID 1"
            ),

        "style_factorization":
            False,

        "bert":
            False,

        "stress":
            (
                "frozen output from "
                "dialogs_np_source_stress_g2p_v1"
            ),
    },
}


(
    TMP / "summary.json"
).write_text(
    json.dumps(
        summary,
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

    for path in sorted(
        TMP.iterdir()
    ):
        if (
            path.is_file()
            and path.name
            != "SHA256SUMS.txt"
        ):
            f.write(
                f"{sha256(path)}  "
                f"{path.name}\n"
            )


TMP.rename(
    OUT
)


print()
print(
    "===== FINAL =====",
    flush=True,
)

print(
    "train:",
    EXPECTED_FINAL["train"],
    flush=True,
)

print(
    "val:",
    EXPECTED_FINAL["val"],
    flush=True,
)

print(
    "test:",
    EXPECTED_FINAL["test"],
    flush=True,
)

print(
    "total:",
    EXPECTED_FINAL["total"],
    flush=True,
)

print(
    "excluded:",
    EXPECTED_EXCLUDED["total"],
    flush=True,
)

print(
    "final_hours:",
    f"{total_seconds/3600:.9f}",
    flush=True,
)

for speaker in SPEAKER_MAP:
    print(
        speaker,
        f"{speaker_seconds[speaker]/3600:.9f} h",
        flush=True,
    )


print()
print(
    "frozen:",
    OUT,
    flush=True,
)

print()
print(
    "DIALOGS NP MELOTTS LISTS V1: PASS",
    flush=True,
)
