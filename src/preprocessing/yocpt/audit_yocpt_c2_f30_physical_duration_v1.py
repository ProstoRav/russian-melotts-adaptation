from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = (
    Path.home()
    / "projects/ru_melotts_research"
)

DATASET = (
    Path.home()
    / "datasets/tts/yocpt_c2_f30_v1"
)

MANIFEST = (
    PROJECT
    / "artifacts/c2_yocpt_row_layout_v1"
    / "c2_f30_manifest_with_global_row_idx.parquet"
)

INVENTORY = (
    DATASET
    / "inventory.jsonl"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_physical_duration_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MIN_CPS = 7.0
MAX_CPS = 19.5

MIN_WPS = 1.0
MAX_WPS = 3.6


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    pos = (
        (len(values) - 1)
        * q
    )

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    f = pos - lo

    return (
        values[lo] * (1 - f)
        +
        values[hi] * f
    )


def quantiles(values):
    return {
        "min": min(values),
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def write_tsv(path, header, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\t".join(header)
            + "\n"
        )

        for row in rows:
            f.write(
                "\t".join(
                    str(x)
                    .replace("\t", " ")
                    .replace("\n", " ")
                    for x in row
                )
                + "\n"
            )


# ---------------------------------------------------------
# Load frozen manifest
# ---------------------------------------------------------

manifest_rows = (
    pq.read_table(
        MANIFEST
    )
    .to_pylist()
)

if len(manifest_rows) != 12720:
    raise RuntimeError(
        f"Expected 12720 manifest rows, "
        f"got {len(manifest_rows)}"
    )

manifest = {
    r["utt_id"]: r
    for r in manifest_rows
}

if len(manifest) != 12720:
    raise RuntimeError(
        "Duplicate manifest utt_id"
    )


# ---------------------------------------------------------
# Load physical inventory
# ---------------------------------------------------------

inventory = {}

with INVENTORY.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        row = json.loads(line)

        inventory[
            row["utt_id"]
        ] = row


if len(inventory) != 12720:
    raise RuntimeError(
        f"Expected 12720 inventory rows, "
        f"got {len(inventory)}"
    )


if set(manifest) != set(inventory):
    raise RuntimeError(
        "Manifest / inventory utt_id mismatch"
    )


# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------

audit = []

signed_deltas = []
abs_deltas = []
relative_pct = []

physical_cps_values = []
physical_wps_values = []

severity_counts = Counter()
severity_seconds = defaultdict(float)

direction_counts = Counter()

shard_stats = defaultdict(
    lambda: {
        "rows": 0,
        "physical_sec": 0.0,
        "gt025": 0,
        "gt075": 0,
        "gt100": 0,
    }
)

video_stats = defaultdict(
    lambda: {
        "rows": 0,
        "physical_sec": 0.0,
        "gt100": 0,
    }
)


for utt_id, m in manifest.items():

    inv = inventory[
        utt_id
    ]

    metadata_duration = float(
        m["duration"]
    )

    physical_duration = float(
        inv["physical_duration"]
    )

    signed_delta = (
        physical_duration
        - metadata_duration
    )

    abs_delta = abs(
        signed_delta
    )

    rel_pct = (
        100.0
        * signed_delta
        / metadata_duration
    )


    text = (
        m.get("text")
        or ""
    )

    training_text = (
        m.get("training_text")
        or
        m.get("text_denorm")
        or
        text
    )


    char_count = m.get(
        "char_count"
    )

    if char_count is None:
        char_count = len(text)


    word_count = m.get(
        "word_count"
    )

    if word_count is None:
        word_count = len(
            text.split()
        )


    physical_cps = (
        float(char_count)
        / physical_duration
    )

    physical_wps = (
        float(word_count)
        / physical_duration
    )


    if abs_delta <= 0.25:
        severity = "le_0.25"

    elif abs_delta <= 0.75:
        severity = "0.25_0.75"

    elif abs_delta <= 1.00:
        severity = "0.75_1.00"

    elif abs_delta <= 1.50:
        severity = "1.00_1.50"

    else:
        severity = "gt_1.50"


    severity_counts[
        severity
    ] += 1

    severity_seconds[
        severity
    ] += physical_duration


    if signed_delta < -0.25:
        direction = "physical_shorter"

    elif signed_delta > 0.25:
        direction = "physical_longer"

    else:
        direction = "near_equal"

    direction_counts[
        direction
    ] += 1


    physical_rate_outlier = (
        physical_cps < MIN_CPS
        or
        physical_cps > MAX_CPS
        or
        physical_wps < MIN_WPS
        or
        physical_wps > MAX_WPS
    )


    physical_duration_outlier = (
        physical_duration < 2.5
        or
        physical_duration > 12.0
    )


    shard = (
        m.get("source_shard")
        or "<missing>"
    )

    video = (
        m.get("source_video_id")
        or "<missing>"
    )


    ss = shard_stats[shard]

    ss["rows"] += 1
    ss["physical_sec"] += (
        physical_duration
    )

    if abs_delta > 0.25:
        ss["gt025"] += 1

    if abs_delta > 0.75:
        ss["gt075"] += 1

    if abs_delta > 1.00:
        ss["gt100"] += 1


    vs = video_stats[video]

    vs["rows"] += 1
    vs["physical_sec"] += (
        physical_duration
    )

    if abs_delta > 1.00:
        vs["gt100"] += 1


    row = {
        **m,

        "filename":
            inv["filename"],

        "physical_duration":
            physical_duration,

        "duration_signed_delta":
            signed_delta,

        "duration_abs_delta":
            abs_delta,

        "duration_relative_pct":
            rel_pct,

        "duration_severity":
            severity,

        "physical_chars_per_sec":
            physical_cps,

        "physical_words_per_sec":
            physical_wps,

        "physical_rate_outlier":
            physical_rate_outlier,

        "physical_duration_outlier":
            physical_duration_outlier,

        "training_text_audit":
            training_text,
    }

    audit.append(row)

    signed_deltas.append(
        signed_delta
    )

    abs_deltas.append(
        abs_delta
    )

    relative_pct.append(
        abs(rel_pct)
    )

    physical_cps_values.append(
        physical_cps
    )

    physical_wps_values.append(
        physical_wps
    )


# ---------------------------------------------------------
# Save full audit
# ---------------------------------------------------------

audit_path = (
    OUT
    / "audit.parquet"
)

pq.write_table(
    pa.Table.from_pylist(
        audit
    ),
    audit_path,
    compression="zstd",
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

physical_seconds = sum(
    r["physical_duration"]
    for r in audit
)

metadata_seconds = sum(
    float(r["duration"])
    for r in audit
)


summary = {
    "rows":
        len(audit),

    "metadata_hours":
        metadata_seconds
        / 3600.0,

    "physical_hours":
        physical_seconds
        / 3600.0,

    "net_physical_minus_metadata_sec":
        physical_seconds
        - metadata_seconds,

    "net_physical_minus_metadata_minutes":
        (
            physical_seconds
            - metadata_seconds
        )
        / 60.0,

    "direction_counts":
        dict(
            direction_counts
        ),

    "severity_counts":
        dict(
            severity_counts
        ),

    "severity_physical_hours": {
        k:
            v / 3600.0
        for k, v
        in severity_seconds.items()
    },

    "signed_delta_sec":
        quantiles(
            signed_deltas
        ),

    "absolute_delta_sec":
        quantiles(
            abs_deltas
        ),

    "absolute_relative_pct":
        quantiles(
            relative_pct
        ),

    "physical_chars_per_sec":
        quantiles(
            physical_cps_values
        ),

    "physical_words_per_sec":
        quantiles(
            physical_wps_values
        ),

    "physical_rate_outliers":
        sum(
            bool(
                r[
                    "physical_rate_outlier"
                ]
            )
            for r in audit
        ),

    "physical_duration_lt_2_5":
        sum(
            r[
                "physical_duration"
            ] < 2.5
            for r in audit
        ),

    "physical_duration_gt_12":
        sum(
            r[
                "physical_duration"
            ] > 12.0
            for r in audit
        ),

    "abs_delta_gt_1_and_rate_outlier":
        sum(
            (
                r[
                    "duration_abs_delta"
                ] > 1.0
            )
            and
            r[
                "physical_rate_outlier"
            ]
            for r in audit
        ),

    "audit_path":
        str(audit_path),
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------
# Worst duration mismatches
# ---------------------------------------------------------

worst = sorted(
    audit,
    key=lambda r: (
        -r[
            "duration_abs_delta"
        ],
        r["utt_id"],
    ),
)


write_tsv(
    OUT
    / "worst_duration_mismatch_top200.tsv",

    [
        "rank",
        "utt_id",
        "source_video_id",
        "source_shard",
        "metadata_sec",
        "physical_sec",
        "signed_delta_sec",
        "abs_delta_sec",
        "relative_pct",
        "physical_cps",
        "physical_wps",
        "rate_outlier",
        "text",
        "training_text",
        "filename",
    ],

    [
        (
            i,
            r["utt_id"],
            r.get(
                "source_video_id",
                "",
            ),
            r.get(
                "source_shard",
                "",
            ),
            f"{float(r['duration']):.3f}",
            f"{r['physical_duration']:.3f}",
            f"{r['duration_signed_delta']:.3f}",
            f"{r['duration_abs_delta']:.3f}",
            f"{r['duration_relative_pct']:.2f}",
            f"{r['physical_chars_per_sec']:.3f}",
            f"{r['physical_words_per_sec']:.3f}",
            int(
                r[
                    "physical_rate_outlier"
                ]
            ),
            r.get("text", ""),
            r.get(
                "training_text_audit",
                "",
            ),
            r["filename"],
        )

        for i, r
        in enumerate(
            worst[:200],
            start=1,
        )
    ],
)


# ---------------------------------------------------------
# Physical speech-rate outliers
# ---------------------------------------------------------

rate_outliers = [
    r
    for r in audit
    if r[
        "physical_rate_outlier"
    ]
]

rate_outliers.sort(
    key=lambda r: (
        -r[
            "duration_abs_delta"
        ],
        r["utt_id"],
    )
)


write_tsv(
    OUT
    / "physical_rate_outliers.tsv",

    [
        "utt_id",
        "metadata_sec",
        "physical_sec",
        "abs_delta_sec",
        "physical_cps",
        "physical_wps",
        "text",
        "filename",
    ],

    [
        (
            r["utt_id"],
            f"{float(r['duration']):.3f}",
            f"{r['physical_duration']:.3f}",
            f"{r['duration_abs_delta']:.3f}",
            f"{r['physical_chars_per_sec']:.3f}",
            f"{r['physical_words_per_sec']:.3f}",
            r.get("text", ""),
            r["filename"],
        )
        for r in rate_outliers
    ],
)


# ---------------------------------------------------------
# Shard concentration
# ---------------------------------------------------------

shard_rows = []

for shard, s in shard_stats.items():

    shard_rows.append((
        shard,
        s["rows"],
        s["physical_sec"] / 3600.0,
        s["gt025"],
        s["gt075"],
        s["gt100"],
        (
            s["gt100"]
            / s["rows"]
        ),
    ))


shard_rows.sort(
    key=lambda x: (
        -x[6],
        -x[5],
        x[0],
    )
)


write_tsv(
    OUT
    / "shard_mismatch_stats.tsv",

    [
        "source_shard",
        "rows",
        "physical_hours",
        "abs_gt_0.25",
        "abs_gt_0.75",
        "abs_gt_1.00",
        "gt1_fraction",
    ],

    [
        (
            shard,
            rows,
            f"{hours:.4f}",
            gt025,
            gt075,
            gt100,
            f"{fraction:.4f}",
        )

        for (
            shard,
            rows,
            hours,
            gt025,
            gt075,
            gt100,
            fraction,
        )
        in shard_rows
    ],
)


print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print(
    "C2-F30 PHYSICAL DURATION "
    "AUDIT V1: PASS"
)
