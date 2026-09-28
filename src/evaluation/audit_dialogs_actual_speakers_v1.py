#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
import hashlib
import json

import pandas as pd


PROJECT = Path(
    "${PROJECT_ROOT}"
)

FROZEN = (
    PROJECT
    / "artifacts/frozen/"
      "dialogs_np_melotts_no_bert_stress_v1"
)

OUT = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_actual_speaker_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DATASET_REVISION = (
    "e25ba617b2b56bd1dbf255d3905c51bd8da3d31f"
)


METADATA_CANDIDATES = [
    Path(
        "${DATA_ROOT}/tts/raw/"
        "dialogs/metadata.csv"
    ),

    (
        Path.home()
        / ".cache/huggingface/hub/"
          "datasets--langswap--"
          "dialogs-ru-emotional-conversations/"
          "snapshots"
        / DATASET_REVISION
        / "metadata.csv"
    ),
]


def sha256(
    path: Path,
):
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


metadata_path = None

for candidate in (
    METADATA_CANDIDATES
):
    if candidate.is_file():
        metadata_path = candidate
        break


if metadata_path is None:
    print(
        "Metadata not found in "
        "expected locations:"
    )

    for p in METADATA_CANDIDATES:
        print(
            "  ",
            p,
        )

    raise RuntimeError(
        "source metadata.csv not found"
    )


meta = pd.read_csv(
    metadata_path,
    sep="|",
)


required = {
    "audio_path",
    "speaker_id",
    "text",
    "emotion",
    "duration",
}


missing_columns = (
    required
    - set(
        meta.columns
    )
)


if missing_columns:
    raise RuntimeError(
        "metadata missing columns: "
        f"{sorted(missing_columns)}"
    )


if len(meta) != 11796:
    raise RuntimeError(
        f"metadata rows="
        f"{len(meta)}, "
        "expected 11796"
    )


meta[
    "basename"
] = meta[
    "audio_path"
].map(
    lambda x:
        Path(
            str(x)
        ).name
)


duplicate_basenames = int(
    meta[
        "basename"
    ].duplicated().sum()
)


if duplicate_basenames:
    raise RuntimeError(
        "duplicate source basenames: "
        f"{duplicate_basenames}"
    )


meta_lookup = (
    meta.set_index(
        "basename",
        drop=False,
    )
)


selected_rows = []


for split in (
    "train",
    "val",
    "test",
):

    list_path = (
        FROZEN
        / f"{split}.list"
    )

    if not list_path.is_file():
        raise RuntimeError(
            f"missing: {list_path}"
        )

    with list_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        lines = [
            line.rstrip(
                "\n"
            )
            for line in f
            if line.strip()
        ]


    for row_index, line in enumerate(
        lines,
        1,
    ):

        fields = line.split(
            "|"
        )

        if len(fields) != 7:
            raise RuntimeError(
                f"{list_path}:"
                f"{row_index}: "
                f"NF={len(fields)}, "
                "expected 7"
            )

        wav_path = Path(
            fields[0]
        )

        artificial_label = (
            fields[1]
        )

        basename = (
            wav_path.name
        )

        if basename not in (
            meta_lookup.index
        ):
            raise RuntimeError(
                f"metadata missing "
                f"{basename}"
            )

        source = (
            meta_lookup.loc[
                basename
            ]
        )

        selected_rows.append(
            {
                "split": split,

                "row_index": (
                    row_index
                ),

                "wav": str(
                    wav_path
                ),

                "basename": (
                    basename
                ),

                "artificial_label": (
                    artificial_label
                ),

                "source_speaker_id": (
                    str(
                        source[
                            "speaker_id"
                        ]
                    )
                ),

                "source_emotion": (
                    str(
                        source[
                            "emotion"
                        ]
                    )
                ),

                "source_audio_path": (
                    str(
                        source[
                            "audio_path"
                        ]
                    )
                ),

                "source_duration": (
                    float(
                        source[
                            "duration"
                        ]
                    )
                ),

                "text_matches_source": (
                    str(
                        fields[3]
                    ).strip()
                    ==
                    str(
                        source[
                            "text"
                        ]
                    ).strip()
                ),
            }
        )


df = pd.DataFrame(
    selected_rows
)


expected_counts = {
    "train": 6433,
    "val": 43,
    "test": 55,
}


actual_counts = (
    df[
        "split"
    ].value_counts().to_dict()
)


for split, expected in (
    expected_counts.items()
):
    actual = int(
        actual_counts.get(
            split,
            0,
        )
    )

    if actual != expected:
        raise RuntimeError(
            f"{split}: "
            f"{actual} != {expected}"
        )


#
# Female-only contract.
#
source_speakers = set(
    df[
        "source_speaker_id"
    ].unique()
)


unexpected = (
    source_speakers
    - {
        "M",
        "S",
    }
)


if unexpected:
    raise RuntimeError(
        "female-only contract failed; "
        f"unexpected source actors: "
        f"{sorted(unexpected)}"
    )


text_mismatch = int(
    (
        ~df[
            "text_matches_source"
        ]
    ).sum()
)


#
# Full selected-row provenance.
#
df.to_csv(
    OUT
    / "selected_rows.csv",
    index=False,
)


#
# Split x actor.
#
split_actor = (
    df.groupby(
        [
            "split",
            "source_speaker_id",
        ]
    )
    .size()
    .reset_index(
        name="rows"
    )
)


split_actor.to_csv(
    OUT
    / "split_actor_counts.csv",
    index=False,
)


#
# Artificial conditioning label x
# actual source actor.
#
label_actor = (
    df.groupby(
        [
            "split",
            "artificial_label",
            "source_speaker_id",
        ]
    )
    .size()
    .reset_index(
        name="rows"
    )
)


label_actor.to_csv(
    OUT
    / "split_label_actor_counts.csv",
    index=False,
)


#
# Include original source emotion too.
#
label_actor_emotion = (
    df.groupby(
        [
            "split",
            "artificial_label",
            "source_speaker_id",
            "source_emotion",
        ]
    )
    .size()
    .reset_index(
        name="rows"
    )
)


label_actor_emotion.to_csv(
    OUT
    / "split_label_actor_emotion_counts.csv",
    index=False,
)


summary = {
    "version": 1,

    "dataset": (
        "langswap/"
        "dialogs-ru-emotional-conversations"
    ),

    "dataset_revision": (
        DATASET_REVISION
    ),

    "source_metadata": (
        str(
            metadata_path
        )
    ),

    "source_metadata_sha256": (
        sha256(
            metadata_path
        )
    ),

    "source_metadata_rows": (
        len(meta)
    ),

    "selected_rows": (
        len(df)
    ),

    "split_rows": {
        k: int(v)
        for k, v in (
            actual_counts.items()
        )
    },

    "source_speakers": sorted(
        source_speakers
    ),

    "text_mismatch_rows": (
        text_mismatch
    ),

    "interpretation": (
        "MeloTTS speaker labels in this "
        "adaptation encode neutral/positive "
        "conditioning; source_speaker_id "
        "contains the real Dialogs actor."
    ),
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


#
# Checksums.
#
files = [
    OUT
    / "selected_rows.csv",

    OUT
    / "split_actor_counts.csv",

    OUT
    / "split_label_actor_counts.csv",

    OUT
    / "split_label_actor_emotion_counts.csv",

    OUT
    / "summary.json",
]


with (
    OUT
    / "SHA256SUMS.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    for path in files:
        f.write(
            f"{sha256(path)}  "
            f"{path.name}\n"
        )


(
    OUT
    / ".complete"
).write_text(
    "PASS\n",
    encoding="utf-8",
)


print("=" * 78)
print(
    "DIALOGS ACTUAL-SPEAKER AUDIT V1"
)
print("=" * 78)

print(
    "source metadata:",
    metadata_path,
)

print(
    "metadata rows:",
    len(meta),
)

print(
    "selected rows:",
    len(df),
)

print(
    "source actors:",
    sorted(
        source_speakers
    ),
)

print(
    "text mismatch rows:",
    text_mismatch,
)


for split in (
    "train",
    "val",
    "test",
):

    x = df[
        df[
            "split"
        ]
        == split
    ]

    print()
    print(
        "=" * 78
    )
    print(
        split.upper()
    )
    print(
        "=" * 78
    )

    print()
    print(
        "ACTUAL ACTOR × "
        "CONDITIONING LABEL"
    )

    print(
        pd.crosstab(
            x[
                "artificial_label"
            ],
            x[
                "source_speaker_id"
            ],
            margins=True,
        ).to_string()
    )

    print()
    print(
        "SOURCE EMOTION × "
        "ACTUAL ACTOR"
    )

    print(
        pd.crosstab(
            x[
                "source_emotion"
            ],
            x[
                "source_speaker_id"
            ],
            margins=True,
        ).to_string()
    )


print()
print("=" * 78)
print(
    "TEST ROW DETAILS"
)
print("=" * 78)

print(
    df[
        df[
            "split"
        ]
        == "test"
    ][
        [
            "basename",
            "artificial_label",
            "source_speaker_id",
            "source_emotion",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 78)
print(
    "DIALOGS ACTUAL-SPEAKER AUDIT V1: PASS"
)
print("=" * 78)

print(
    "output:",
    OUT,
)
