from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = (
    Path.home()
    / "projects/ru_melotts_research"
)

SOURCE = (
    PROJECT
    / "artifacts/c2_yocpt_f30_clean_pool_v1"
    / "clean_pool.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_manifest_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SOURCE_CAP_SEC = 60.0
LOCAL_CAP_SEC = 60.0

TARGETS = [
    (
        "2.5-6",
        2.5,
        6.0,
        3.0 * 3600.0,
    ),
    (
        "6-8",
        6.0,
        8.0,
        7.0 * 3600.0,
    ),
    (
        "8-10",
        8.0,
        10.0,
        9.0 * 3600.0,
    ),
    (
        "10-12",
        10.0,
        12.0001,
        11.0 * 3600.0,
    ),
]

MAX_BIN_OVERSHOOT_SEC = 12.0


def deterministic_jitter(
    utt_id: str,
) -> float:
    digest = hashlib.sha256(
        (
            "C2-F30-v1|"
            + utt_id
        ).encode(
            "utf-8"
        )
    ).digest()

    value = int.from_bytes(
        digest[:8],
        "big",
    )

    return (
        value
        / float(2**64)
    )


def duration_bin(
    duration: float,
):
    for (
        name,
        lo,
        hi,
        _,
    ) in TARGETS:

        if (
            duration >= lo
            and duration < hi
        ):
            return name

    return None


table = pq.read_table(
    SOURCE
)

rows = table.to_pylist()

print(
    "clean pool rows:",
    len(rows),
)

print(
    "clean pool hours:",
    sum(
        float(r["duration"])
        for r in rows
    ) / 3600.0,
)


# ---------------------------------------------------------
# Prepare candidates
# ---------------------------------------------------------

for row in rows:

    mos = float(
        row["mos_score"]
    )

    quality = float(
        row.get(
            "quality_score",
            mos,
        )
    )

    #
    # MOS dominates.
    # Existing quality_score adds only a mild
    # normal-speech-rate preference.
    #
    row["_rank_score"] = (
        0.80 * mos
        +
        0.20 * quality
        +
        1e-6
        * deterministic_jitter(
            row["utt_id"]
        )
    )

    row["_duration_bin"] = (
        duration_bin(
            float(
                row["duration"]
            )
        )
    )


selected = []

selected_ids = set()

source_used = defaultdict(
    float
)

local_used = defaultdict(
    float
)

bin_used = defaultdict(
    float
)


def can_take(
    row,
    bin_target,
):
    duration = float(
        row["duration"]
    )

    source = (
        row.get(
            "source_video_id"
        )
        or "<missing>"
    )

    local = (
        row.get(
            "local_spk_id"
        )
        or "<missing>"
    )

    if (
        source_used[source]
        + duration
        > SOURCE_CAP_SEC
    ):
        return False

    if (
        local_used[local]
        + duration
        > LOCAL_CAP_SEC
    ):
        return False

    b = row[
        "_duration_bin"
    ]

    if (
        bin_used[b]
        + duration
        >
        bin_target
        + MAX_BIN_OVERSHOOT_SEC
    ):
        return False

    return True


def take(row):

    duration = float(
        row["duration"]
    )

    source = (
        row.get(
            "source_video_id"
        )
        or "<missing>"
    )

    local = (
        row.get(
            "local_spk_id"
        )
        or "<missing>"
    )

    b = row[
        "_duration_bin"
    ]

    selected.append(
        row
    )

    selected_ids.add(
        row["utt_id"]
    )

    source_used[
        source
    ] += duration

    local_used[
        local
    ] += duration

    bin_used[
        b
    ] += duration


# ---------------------------------------------------------
# Fill each duration bin using source round-robin.
#
# First best utterance from many sources,
# then second utterance from a source, etc.
# ---------------------------------------------------------

for (
    bin_name,
    lo,
    hi,
    target_sec,
) in TARGETS:

    candidates = [
        r
        for r in rows
        if r[
            "_duration_bin"
        ] == bin_name
    ]

    by_source = defaultdict(
        list
    )

    for row in candidates:

        source = (
            row.get(
                "source_video_id"
            )
            or "<missing>"
        )

        by_source[
            source
        ].append(row)

    for source in by_source:

        by_source[
            source
        ].sort(
            key=lambda r: (
                -r["_rank_score"],
                r["utt_id"],
            )
        )

    source_order = sorted(
        by_source.keys(),
        key=lambda source: (
            -by_source[
                source
            ][0][
                "_rank_score"
            ],

            deterministic_jitter(
                source
            ),
        ),
    )

    round_idx = 0

    while (
        bin_used[
            bin_name
        ]
        < target_sec
    ):

        added_this_round = 0

        for source in source_order:

            queue = (
                by_source[
                    source
                ]
            )

            if (
                round_idx
                >= len(queue)
            ):
                continue

            row = queue[
                round_idx
            ]

            if row[
                "utt_id"
            ] in selected_ids:
                continue

            if can_take(
                row,
                target_sec,
            ):
                take(
                    row
                )

                added_this_round += 1

                if (
                    bin_used[
                        bin_name
                    ]
                    >= target_sec
                ):
                    break

        if (
            bin_used[
                bin_name
            ]
            >= target_sec
        ):
            break

        if added_this_round == 0:

            #
            # Round-robin position may contain
            # blocked rows while later rows remain usable.
            # Do one global-quality fallback pass.
            #
            remaining = sorted(
                (
                    r
                    for r in candidates
                    if r["utt_id"]
                    not in selected_ids
                ),
                key=lambda r: (
                    -r["_rank_score"],
                    r["utt_id"],
                ),
            )

            fallback_added = 0

            for row in remaining:

                if can_take(
                    row,
                    target_sec,
                ):
                    take(
                        row
                    )

                    fallback_added += 1

                    if (
                        bin_used[
                            bin_name
                        ]
                        >= target_sec
                    ):
                        break

            if fallback_added == 0:
                break

        round_idx += 1

    print(
        f"{bin_name}: "
        f"{bin_used[bin_name] / 3600.0:.4f} h "
        f"/ target "
        f"{target_sec / 3600.0:.2f} h"
    )


# ---------------------------------------------------------
# Strip private selection fields
# ---------------------------------------------------------

output_rows = []

for row in selected:

    output_rows.append({
        k: v
        for k, v
        in row.items()
        if not k.startswith("_")
    })


manifest = pa.Table.from_pylist(
    output_rows
)

manifest_path = (
    OUT
    / "manifest.parquet"
)

pq.write_table(
    manifest,
    manifest_path,
    compression="zstd",
)


# ---------------------------------------------------------
# JSONL for easy audit / future downloader
# ---------------------------------------------------------

jsonl_path = (
    OUT
    / "manifest.jsonl"
)

with jsonl_path.open(
    "w",
    encoding="utf-8",
) as f:

    for row in output_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

total_sec = sum(
    float(r["duration"])
    for r in output_rows
)

video_counter = Counter(
    r.get(
        "source_video_id"
    )
    or "<missing>"
    for r in output_rows
)

local_counter = Counter(
    r.get(
        "local_spk_id"
    )
    or "<missing>"
    for r in output_rows
)

global_counter = Counter(
    r.get(
        "global_spk_id"
    )
    for r in output_rows
    if r.get(
        "global_spk_id"
    )
)

mos_bins = Counter()

duration_counts = Counter()

duration_hours = defaultdict(
    float
)

global_seconds = defaultdict(
    float
)

for r in output_rows:

    duration = float(
        r["duration"]
    )

    mos = float(
        r["mos_score"]
    )

    b = duration_bin(
        duration
    )

    duration_counts[
        b
    ] += 1

    duration_hours[
        b
    ] += duration / 3600.0

    if mos < 4.30:
        mos_bins[
            "4.20-4.30"
        ] += duration

    elif mos < 4.40:
        mos_bins[
            "4.30-4.40"
        ] += duration

    elif mos < 4.50:
        mos_bins[
            "4.40-4.50"
        ] += duration

    else:
        mos_bins[
            "4.50+"
        ] += duration

    global_id = r.get(
        "global_spk_id"
    )

    if global_id:
        global_seconds[
            global_id
        ] += duration


summary = {
    "target_hours":
        30.0,

    "selected_rows":
        len(output_rows),

    "selected_hours":
        total_sec
        / 3600.0,

    "source_video_cap_sec":
        SOURCE_CAP_SEC,

    "local_speaker_cap_sec":
        LOCAL_CAP_SEC,

    "global_speaker_cap_applied":
        False,

    "duration_bin_hours": {
        k:
            float(v)
        for k, v
        in duration_hours.items()
    },

    "duration_bin_rows": {
        k:
            int(v)
        for k, v
        in duration_counts.items()
    },

    "mos_bin_hours": {
        k:
            float(v)
            / 3600.0
        for k, v
        in mos_bins.items()
    },

    "unique_source_videos":
        len(
            video_counter
        ),

    "unique_local_speaker_ids":
        len(
            local_counter
        ),

    "global_speaker_ids_present":
        len(
            global_counter
        ),

    "max_source_video_sec":
        max(
            source_used.values(),
            default=0.0,
        ),

    "max_local_speaker_sec":
        max(
            local_used.values(),
            default=0.0,
        ),

    "diagnostic_max_known_global_speaker_sec":
        max(
            global_seconds.values(),
            default=0.0,
        ),

    "manifest_path":
        str(
            manifest_path
        ),
}


summary_path = (
    OUT
    / "summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "C2-F30 MANIFEST V1: PASS"
)
