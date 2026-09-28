from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path.home() / "projects/ru_melotts_research"

SRC = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_speaker_conditioning_v1"
    / "manifest.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_train_val_split_v1"
)

EXPECTED_ROWS = 12689
EXPECTED_HOURS = 29.3589216898

TARGET_VAL_HOURS = 1.0
SEED = "c2_f30_split_v1_20260828"


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def valid_global(x):
    if x is None:
        return False

    s = str(x).strip()

    return (
        s != ""
        and s.lower()
        not in {
            "none",
            "nan",
            "null",
        }
    )


class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )
        return self.parent[x]

    def union(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


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
        f"rows={len(rows)}"
    )


total_hours = (
    sum(
        float(r["physical_duration"])
        for r in rows
    )
    / 3600.0
)

if abs(
    total_hours
    - EXPECTED_HOURS
) > 1e-9:
    raise RuntimeError(
        f"hours mismatch: {total_hours}"
    )


if len({
    r["utt_id"]
    for r in rows
}) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate utt_id"
    )


#
# ==========================================================
# Build source_video <-> known global speaker components
# ==========================================================
#
dsu = DSU()

videos = sorted({
    str(r["source_video_id"])
    for r in rows
})

for video in videos:
    dsu.add(
        "video:" + video
    )


for r in rows:

    video_node = (
        "video:"
        + str(r["source_video_id"])
    )

    gid = r.get(
        "global_spk_id"
    )

    if valid_global(gid):
        dsu.union(
            video_node,
            "speaker:"
            + str(gid).strip(),
        )


rows_by_root = defaultdict(list)
videos_by_root = defaultdict(set)
globals_by_root = defaultdict(set)


for r in rows:

    video = str(
        r["source_video_id"]
    )

    root = dsu.find(
        "video:" + video
    )

    rows_by_root[root].append(r)
    videos_by_root[root].add(video)

    gid = r.get(
        "global_spk_id"
    )

    if valid_global(gid):
        globals_by_root[root].add(
            str(gid).strip()
        )


components = []


for root, group in rows_by_root.items():

    component_videos = sorted(
        videos_by_root[root]
    )

    component_globals = sorted(
        globals_by_root[root]
    )

    #
    # Stable component identity independent of DSU root.
    #
    signature = json.dumps(
        {
            "videos": component_videos,
            "global_speakers":
                component_globals,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    component_id = (
        "cc_"
        + hashlib.sha256(
            signature.encode("utf-8")
        ).hexdigest()[:20]
    )

    selection_hash = (
        hashlib.sha256(
            (
                SEED
                + "|"
                + component_id
            ).encode("utf-8")
        ).hexdigest()
    )

    hours = (
        sum(
            float(r["physical_duration"])
            for r in group
        )
        / 3600.0
    )

    components.append({
        "root":
            root,

        "component_id":
            component_id,

        "selection_hash":
            selection_hash,

        "rows":
            len(group),

        "hours":
            hours,

        "videos":
            len(component_videos),

        "global_speakers":
            len(component_globals),

        "stratum":
            (
                "known_global"
                if component_globals
                else "unknown_global"
            ),
    })


if len(components) != 6653:
    raise RuntimeError(
        f"expected 6653 components, "
        f"got {len(components)}"
    )


#
# ==========================================================
# Deterministic stratified validation selection
# ==========================================================
#
hours_by_stratum = defaultdict(float)

for c in components:
    hours_by_stratum[
        c["stratum"]
    ] += c["hours"]


target_by_stratum = {
    stratum:
        TARGET_VAL_HOURS
        * hours
        / total_hours
    for stratum, hours
    in hours_by_stratum.items()
}


val_component_ids = set()


for stratum in sorted(
    hours_by_stratum
):

    candidates = sorted(
        [
            c
            for c in components
            if c["stratum"]
            == stratum
        ],
        key=lambda c:
            c["selection_hash"],
    )

    target = (
        target_by_stratum[
            stratum
        ]
    )

    current = 0.0

    #
    # Include a component only when it gets
    # this stratum closer to its target.
    #
    for c in candidates:

        before = abs(
            current
            - target
        )

        after = abs(
            current
            + c["hours"]
            - target
        )

        if after < before:
            val_component_ids.add(
                c["component_id"]
            )

            current += c["hours"]


#
# Map stable component IDs back to roots.
#
component_id_by_root = {
    c["root"]:
        c["component_id"]
    for c in components
}


component_split = {}

for c in components:
    component_split[
        c["component_id"]
    ] = (
        "val"
        if c["component_id"]
        in val_component_ids
        else "train"
    )


#
# ==========================================================
# Assign rows
# ==========================================================
#
assigned = []

for r in rows:

    root = dsu.find(
        "video:"
        + str(r["source_video_id"])
    )

    component_id = (
        component_id_by_root[root]
    )

    split = (
        component_split[
            component_id
        ]
    )

    assigned.append({
        **r,

        "split_component_id":
            component_id,

        "split":
            split,
    })


train = [
    r
    for r in assigned
    if r["split"] == "train"
]

val = [
    r
    for r in assigned
    if r["split"] == "val"
]


train_hours = (
    sum(
        float(r["physical_duration"])
        for r in train
    )
    / 3600.0
)

val_hours = (
    sum(
        float(r["physical_duration"])
        for r in val
    )
    / 3600.0
)


if abs(
    val_hours
    - TARGET_VAL_HOURS
) > 0.02:
    raise RuntimeError(
        f"validation hours too far "
        f"from target: {val_hours}"
    )


#
# ==========================================================
# Leakage checks
# ==========================================================
#
train_videos = {
    str(r["source_video_id"])
    for r in train
}

val_videos = {
    str(r["source_video_id"])
    for r in val
}

video_overlap = (
    train_videos
    & val_videos
)


train_globals = {
    str(r["global_spk_id"]).strip()
    for r in train
    if valid_global(
        r.get("global_spk_id")
    )
}

val_globals = {
    str(r["global_spk_id"]).strip()
    for r in val
    if valid_global(
        r.get("global_spk_id")
    )
}

global_overlap = (
    train_globals
    & val_globals
)


if video_overlap:
    raise RuntimeError(
        f"source_video leakage: "
        f"{len(video_overlap)}"
    )


if global_overlap:
    raise RuntimeError(
        f"global speaker leakage: "
        f"{len(global_overlap)}"
    )


#
# Every component must belong to exactly one split.
#
component_to_splits = defaultdict(set)

for r in assigned:
    component_to_splits[
        r["split_component_id"]
    ].add(
        r["split"]
    )

bad_components = {
    k: v
    for k, v
    in component_to_splits.items()
    if len(v) != 1
}

if bad_components:
    raise RuntimeError(
        "component leakage detected"
    )


#
# ==========================================================
# Diagnostics
# ==========================================================
#
RISK_FIELDS = [
    "qc_duration_mismatch_gt_0_25",
    "qc_duration_mismatch_gt_1_00",
    "qc_eof_m40_quiet_lt_50ms",
    "qc_eof_m40_quiet_lt_100ms",
    "qc_training_rate_outlier",
    "qc_rare_technical_candidate",
]


def subset_stats(group):

    hours = (
        sum(
            float(r["physical_duration"])
            for r in group
        )
        / 3600.0
    )

    known_rows = sum(
        valid_global(
            r.get("global_spk_id")
        )
        for r in group
    )

    result = {
        "rows":
            len(group),

        "hours":
            hours,

        "source_videos":
            len({
                str(r["source_video_id"])
                for r in group
            }),

        "known_global_speaker_rows":
            known_rows,

        "known_global_speaker_row_fraction":
            known_rows / len(group),

        "known_global_speakers":
            len({
                str(
                    r["global_spk_id"]
                ).strip()
                for r in group
                if valid_global(
                    r.get(
                        "global_spk_id"
                    )
                )
            }),

        "duration_seconds": {
            "p05":
                float(
                    np.percentile(
                        [
                            float(
                                r["physical_duration"]
                            )
                            for r in group
                        ],
                        5,
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        [
                            float(
                                r["physical_duration"]
                            )
                            for r in group
                        ],
                        50,
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        [
                            float(
                                r["physical_duration"]
                            )
                            for r in group
                        ],
                        95,
                    )
                ),
        },

        "risk": {},
    }

    for field in RISK_FIELDS:

        count = sum(
            bool(r[field])
            for r in group
        )

        result["risk"][
            field
        ] = {
            "count":
                count,

            "fraction":
                count
                / len(group),
        }

    return result


train_stats = subset_stats(
    train
)

val_stats = subset_stats(
    val
)


#
# Component-level frozen table.
#
component_rows = []

for c in sorted(
    components,
    key=lambda x:
        x["component_id"],
):

    component_rows.append({
        "component_id":
            c["component_id"],

        "split":
            component_split[
                c["component_id"]
            ],

        "stratum":
            c["stratum"],

        "rows":
            c["rows"],

        "hours":
            c["hours"],

        "source_videos":
            c["videos"],

        "global_speakers":
            c["global_speakers"],

        "selection_hash":
            c["selection_hash"],
    })


#
# ==========================================================
# Freeze
# ==========================================================
#
OUT.mkdir(
    parents=True,
    exist_ok=False,
)

ALL = (
    OUT
    / "manifest_with_split.parquet"
)

TRAIN = (
    OUT
    / "manifest_train.parquet"
)

VAL = (
    OUT
    / "manifest_val.parquet"
)

COMPONENTS = (
    OUT
    / "components.parquet"
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


pq.write_table(
    pa.Table.from_pylist(
        assigned
    ),
    ALL,
    compression="zstd",
)

pq.write_table(
    pa.Table.from_pylist(
        train
    ),
    TRAIN,
    compression="zstd",
)

pq.write_table(
    pa.Table.from_pylist(
        val
    ),
    VAL,
    compression="zstd",
)

pq.write_table(
    pa.Table.from_pylist(
        component_rows
    ),
    COMPONENTS,
    compression="zstd",
)


summary = {
    "artifact":
        "C2-F30-train-val-split-v1",

    "source_manifest":
        str(SRC),

    "source_manifest_sha256":
        sha256_file(SRC),

    "selection_seed":
        SEED,

    "target_val_hours":
        TARGET_VAL_HOURS,

    "grouping_policy":
        (
            "Connected components of source_video_id "
            "and known global_spk_id. No component may "
            "cross train/validation."
        ),

    "speaker_conditioning":
        (
            "Single synthetic foundation speaker ID. "
            "YO-CPT remains physically multi-speaker."
        ),

    "rows_total":
        len(assigned),

    "hours_total":
        total_hours,

    "components_total":
        len(components),

    "components_train":
        sum(
            c["split"] == "train"
            for c in component_rows
        ),

    "components_val":
        sum(
            c["split"] == "val"
            for c in component_rows
        ),

    "train":
        train_stats,

    "val":
        val_stats,

    "leakage_checks": {
        "source_video_overlap":
            len(video_overlap),

        "known_global_speaker_overlap":
            len(global_overlap),

        "component_overlap":
            len(bad_components),
    },

    "stratum_total_hours":
        dict(
            sorted(
                hours_by_stratum.items()
            )
        ),

    "stratum_target_val_hours":
        dict(
            sorted(
                target_by_stratum.items()
            )
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
    """# C2-F30 train/validation split v1

Validation target: approximately 1.0 physical hour.

Atomic split unit is the connected component formed by:

- `source_video_id`
- known `global_spk_id`

Therefore:

1. clips from one source video cannot cross train/validation;
2. where YO-CPT provides a global speaker identity, that speaker
   cannot cross train/validation even across different videos;
3. rows without a global speaker identity still receive strict
   source-video isolation.

YO-CPT is physically multi-speaker. The single synthetic speaker
ID used by C2-F foundation is a conditioning strategy only and
does not imply that these recordings originate from one speaker.
""",
    encoding="utf-8",
)


output_files = [
    ALL,
    TRAIN,
    VAL,
    COMPONENTS,
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

    if (
        sha256_file(
            OUT / filename
        )
        != digest
    ):
        raise RuntimeError(
            f"SHA mismatch: {filename}"
        )


print(
    "===== C2-F30 TRAIN/VAL SPLIT V1 ====="
)

print(
    "rows total:",
    len(assigned),
)

print(
    "hours total:",
    round(
        total_hours,
        10,
    ),
)

print()
print(
    "components total:",
    len(components),
)

print(
    "components train:",
    summary[
        "components_train"
    ],
)

print(
    "components val:",
    summary[
        "components_val"
    ],
)

print()
print(
    "train rows:",
    len(train),
)

print(
    "train hours:",
    round(
        train_hours,
        10,
    ),
)

print(
    "train videos:",
    len(train_videos),
)

print(
    "train known global speakers:",
    len(train_globals),
)

print()
print(
    "val rows:",
    len(val),
)

print(
    "val hours:",
    round(
        val_hours,
        10,
    ),
)

print(
    "val videos:",
    len(val_videos),
)

print(
    "val known global speakers:",
    len(val_globals),
)

print()
print(
    "source_video overlap:",
    len(video_overlap),
)

print(
    "known global speaker overlap:",
    len(global_overlap),
)

print(
    "component overlap:",
    len(bad_components),
)


print()
print(
    "===== RISK FRACTIONS TRAIN / VAL ====="
)

for field in RISK_FIELDS:

    tr = (
        train_stats[
            "risk"
        ][field]["fraction"]
    )

    va = (
        val_stats[
            "risk"
        ][field]["fraction"]
    )

    print(
        field,
        f"{tr:.4f}",
        "/",
        f"{va:.4f}",
    )


print()
print(
    "known-global row fraction train/val:",
    round(
        train_stats[
            "known_global_speaker_row_fraction"
        ],
        4,
    ),
    "/",
    round(
        val_stats[
            "known_global_speaker_row_fraction"
        ],
        4,
    ),
)


print()
print(
    "manifest train:",
    TRAIN,
)

print(
    "manifest val:",
    VAL,
)

print(
    "components:",
    COMPONENTS,
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
    "C2-F30 TRAIN/VAL SPLIT V1: PASS"
)
