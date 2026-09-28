from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path.home() / "projects/ru_melotts_research"

SRC = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_physical_v1"
    / "manifest_trainable.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_speaker_conditioning_v1"
)

EXPECTED_ROWS = 12689
EXPECTED_HOURS = 29.3589216898


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


if OUT.exists():
    raise RuntimeError(
        f"refusing to overwrite frozen artifact: {OUT}"
    )


rows = (
    pq.read_table(SRC)
    .to_pylist()
)

if len(rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"expected {EXPECTED_ROWS} rows, "
        f"got {len(rows)}"
    )


hours = (
    sum(
        float(r["physical_duration"])
        for r in rows
    )
    / 3600.0
)

if abs(
    hours
    - EXPECTED_HOURS
) > 1e-9:
    raise RuntimeError(
        f"physical hours mismatch: {hours}"
    )


if len({
    r["utt_id"]
    for r in rows
}) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate utt_id"
    )


#
# Explicitly overwrite only the conditioning field.
# Original speaker metadata stays untouched.
#
conditioned = []

for r in rows:

    conditioned.append({
        **r,

        "foundation_speaker_id":
            0,

        "foundation_speaker_name":
            "c2_f30_generic_ru",
    })


if {
    r["foundation_speaker_id"]
    for r in conditioned
} != {0}:
    raise RuntimeError(
        "unexpected foundation speaker IDs"
    )


OUT.mkdir(
    parents=True,
    exist_ok=False,
)

MANIFEST = (
    OUT
    / "manifest.parquet"
)

CONFIG = (
    OUT
    / "speaker_conditioning.json"
)

README = (
    OUT
    / "README.md"
)

SHA = (
    OUT
    / "SHA256SUMS.txt"
)


pq.write_table(
    pa.Table.from_pylist(
        conditioned
    ),
    MANIFEST,
    compression="zstd",
)


config = {
    "artifact":
        "C2-F30-speaker-conditioning-v1",

    "source_manifest":
        str(SRC),

    "source_manifest_sha256":
        sha256_file(SRC),

    "rows":
        len(conditioned),

    "physical_hours":
        hours,

    "conditioning_strategy":
        "single_synthetic_speaker",

    "n_speakers":
        1,

    "speaker_id":
        0,

    "speaker_name":
        "c2_f30_generic_ru",

    "conditioning_field":
        "foundation_speaker_id",

    "source_speaker_metadata_policy":
        (
            "local_spk_id, global_spk_id and speaker_key "
            "are retained as provenance only and are not "
            "used as conditioning labels for C2-F foundation."
        ),

    "rationale": [
        (
            "local_spk_id is video-local: 8070 IDs, "
            "65.12% single-row, 0% span multiple videos."
        ),
        (
            "global_spk_id covers only 3571/12689 rows "
            "and is highly fragmented."
        ),
        (
            "speaker_key has 6814 IDs; "
            "62.56% are single-row keys."
        ),
        (
            "The goal of C2-F is Russian foundation "
            "adaptation rather than learning unreliable "
            "YO-CPT pseudo-speaker identities."
        ),
    ],

    "split":
        None,
}


CONFIG.write_text(
    json.dumps(
        config,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


README.write_text(
    """# C2-F30 speaker conditioning v1

C2-F foundation uses one synthetic speaker identity for every
trainable C2-F30 utterance:

- `n_speakers = 1`
- `foundation_speaker_id = 0`
- `foundation_speaker_name = c2_f30_generic_ru`

YO-CPT `local_spk_id`, `global_spk_id`, and `speaker_key` remain
in the manifest for provenance and analysis only.

This artifact does not define the train/validation split.
""",
    encoding="utf-8",
)


files = [
    MANIFEST,
    CONFIG,
    README,
]

with SHA.open(
    "w",
    encoding="utf-8",
) as f:

    for file in files:
        f.write(
            f"{sha256_file(file)}  "
            f"{file.name}\n"
        )


#
# Verify frozen outputs immediately.
#
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


print(
    "===== C2-F30 SPEAKER CONDITIONING V1 ====="
)

print(
    "rows:",
    len(conditioned),
)

print(
    "physical_hours:",
    round(
        hours,
        10,
    ),
)

print(
    "conditioning:",
    "single_synthetic_speaker",
)

print(
    "n_speakers:",
    1,
)

print(
    "speaker_id:",
    0,
)

print(
    "speaker_name:",
    "c2_f30_generic_ru",
)

print()
print(
    "manifest:",
    MANIFEST,
)

print(
    "config:",
    CONFIG,
)

print(
    "sha256:",
    SHA,
)

print()
print(
    "C2-F30 SPEAKER CONDITIONING V1: PASS"
)
