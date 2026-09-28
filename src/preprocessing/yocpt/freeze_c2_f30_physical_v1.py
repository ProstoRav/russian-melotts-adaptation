from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path.home() / "projects/ru_melotts_research"

DATASET = (
    Path.home()
    / "datasets/tts/yocpt_c2_f30_v1"
)

FROZEN_MANIFEST = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_manifest_v1"
    / "manifest.parquet"
)

MAPPED_MANIFEST = (
    PROJECT
    / "artifacts/c2_yocpt_row_layout_v1"
    / "c2_f30_manifest_with_global_row_idx.parquet"
)

INVENTORY = (
    DATASET
    / "inventory.jsonl"
)

QC = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_trailing_activity_v1"
    / "trailing_activity.parquet"
)

REVIEW_EOF = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_eof_manual_review_result_v1"
    / "adjudicated_review.tsv"
)

REVIEW_HOLDOUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_alignment_holdout_v1"
    / "blind_review.tsv"
)

REVIEW_RARE = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_rare_physical_review_v1"
    / "review.tsv"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_physical_v1"
)

EXPECTED_FROZEN_SHA256 = (
    "ea926d614380b152da675b074b8b461e"
    "037d7825a3aff9c050fbc151aeb69295"
)

EXPECTED_ROWS = 12720
EXPECTED_MANUAL_ROWS = 116
EXPECTED_HARD_EXCLUDES = 31

MIN_CPS = 7.0
MAX_CPS = 19.5
MIN_WPS = 1.0
MAX_WPS = 3.6

SPACE_RE = re.compile(r"\s+")


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def normalize_space(value):
    if value is None:
        return ""

    return SPACE_RE.sub(
        " ",
        str(value),
    ).strip()


def make_training_text(value):
    """
    Preserve punctuation/case/ё from text_denorm,
    remove YO-CPT stress markup.
    """
    text = normalize_space(value)

    text = text.replace(
        "+",
        "",
    )

    text = text.replace(
        "\u0301",
        "",
    )

    return unicodedata.normalize(
        "NFC",
        SPACE_RE.sub(
            " ",
            text,
        ).strip(),
    )


def read_tsv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )


#
# ------------------------------------------------------------
# Immutable-output guard
# ------------------------------------------------------------
#
if OUT.exists():
    raise RuntimeError(
        f"refusing to overwrite frozen artifact: {OUT}"
    )


#
# ------------------------------------------------------------
# Verify frozen selection identity
# ------------------------------------------------------------
#
frozen_sha = sha256_file(
    FROZEN_MANIFEST
)

if frozen_sha != EXPECTED_FROZEN_SHA256:
    raise RuntimeError(
        "frozen manifest SHA256 mismatch:\n"
        f"actual   {frozen_sha}\n"
        f"expected {EXPECTED_FROZEN_SHA256}"
    )


manifest_rows = (
    pq.read_table(
        FROZEN_MANIFEST
    )
    .to_pylist()
)

if len(manifest_rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"frozen manifest rows={len(manifest_rows)}"
    )


required_manifest = {
    "utt_id",
    "source_shard",
    "row_group",
    "row_in_group",
    "source_video_id",
    "text",
    "text_denorm",
    "duration",
}

missing = (
    required_manifest
    - set(manifest_rows[0])
)

if missing:
    raise RuntimeError(
        f"missing frozen fields: {sorted(missing)}"
    )


manifest_by_utt = {
    r["utt_id"]: r
    for r in manifest_rows
}

if len(manifest_by_utt) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate utt_id in frozen manifest"
    )


#
# ------------------------------------------------------------
# Derived YO-CPT global-row mapping
#
# global_row_idx is NOT the position inside F30.
# It is the row position in the full 1,627,283-row
# YO-CPT Dataset Viewer layout:
#
#   row_group_global_start + row_in_group
#
# ------------------------------------------------------------
#
mapped_rows = (
    pq.read_table(
        MAPPED_MANIFEST
    )
    .to_pylist()
)

if len(mapped_rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"mapped manifest rows={len(mapped_rows)}"
    )


required_mapped = {
    "utt_id",
    "global_row_idx",
    "source_shard",
    "row_group",
    "row_in_group",
}

missing_mapped = (
    required_mapped
    - set(mapped_rows[0])
)

if missing_mapped:
    raise RuntimeError(
        "missing mapped fields: "
        f"{sorted(missing_mapped)}"
    )


mapped_by_utt = {
    r["utt_id"]: r
    for r in mapped_rows
}

if len(mapped_by_utt) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate utt_id in mapped manifest"
    )


if (
    set(mapped_by_utt)
    != set(manifest_by_utt)
):
    raise RuntimeError(
        "frozen/mapped utt_id sets differ"
    )


global_ids = [
    int(r["global_row_idx"])
    for r in mapped_rows
]

if len(global_ids) != len(set(global_ids)):
    raise RuntimeError(
        "duplicate global_row_idx "
        "in mapped manifest"
    )


if min(global_ids) < 0:
    raise RuntimeError(
        "negative global_row_idx"
    )


if max(global_ids) >= 1627283:
    raise RuntimeError(
        "global_row_idx outside "
        "known YO-CPT row layout"
    )


#
# The mapped manifest must be the frozen selection
# plus exactly one derived coordinate.
#
for utt, frozen_row in manifest_by_utt.items():

    mapped_row = mapped_by_utt[utt]

    if (
        str(frozen_row["source_shard"])
        != str(mapped_row["source_shard"])
    ):
        raise RuntimeError(
            f"{utt}: mapped source_shard mismatch"
        )

    for field in (
        "row_group",
        "row_in_group",
    ):
        if (
            int(frozen_row[field])
            != int(mapped_row[field])
        ):
            raise RuntimeError(
                f"{utt}: mapped {field} mismatch"
            )


#
# ------------------------------------------------------------
# Physical inventory
# ------------------------------------------------------------
#
inventory_rows = []

with INVENTORY.open(
    "r",
    encoding="utf-8",
) as f:

    for line_no, line in enumerate(
        f,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        row = json.loads(line)
        inventory_rows.append(row)


if len(inventory_rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"inventory rows={len(inventory_rows)}"
    )


inventory_by_utt = {
    r["utt_id"]: r
    for r in inventory_rows
}

if len(inventory_by_utt) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate utt_id in inventory"
    )


#
# ------------------------------------------------------------
# Objective waveform QC
# ------------------------------------------------------------
#
qc_rows = (
    pq.read_table(QC)
    .to_pylist()
)

if len(qc_rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"QC rows={len(qc_rows)}"
    )


qc_by_utt = {
    r["utt_id"]: r
    for r in qc_rows
}

if len(qc_by_utt) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate utt_id in QC"
    )


if (
    set(manifest_by_utt)
    != set(inventory_by_utt)
    or set(manifest_by_utt)
    != set(qc_by_utt)
):
    raise RuntimeError(
        "utt_id sets differ between "
        "frozen manifest / inventory / QC"
    )


#
# ------------------------------------------------------------
# Manual QC annotations
# ------------------------------------------------------------
#
manual = {}


def ingest_manual(path, source):
    rows = read_tsv(path)

    for r in rows:
        utt = r["utt_id"]
        label = r["label"].strip()

        if not label:
            raise RuntimeError(
                f"empty manual label: "
                f"{source} {utt}"
            )

        previous = manual.get(utt)

        if previous is not None:

            if previous["label"] != label:
                raise RuntimeError(
                    f"conflicting manual label "
                    f"for {utt}: "
                    f"{previous['label']} vs {label}"
                )

            previous["sources"].append(
                source
            )

        else:
            manual[utt] = {
                "label": label,
                "sources": [source],
            }


ingest_manual(
    REVIEW_EOF,
    "eof_calibration_v1",
)

ingest_manual(
    REVIEW_HOLDOUT,
    "alignment_holdout_v1",
)

ingest_manual(
    REVIEW_RARE,
    "rare_physical_review_v1",
)


if len(manual) != EXPECTED_MANUAL_ROWS:
    raise RuntimeError(
        f"expected {EXPECTED_MANUAL_ROWS} "
        f"unique manually reviewed rows, "
        f"got {len(manual)}"
    )


unknown_manual = (
    set(manual)
    - set(manifest_by_utt)
)

if unknown_manual:
    raise RuntimeError(
        f"manual QC contains unknown utts: "
        f"{sorted(unknown_manual)[:10]}"
    )


manual_counts = Counter(
    x["label"]
    for x in manual.values()
)


#
# Only individually confirmed problems are excluded.
#
HARD_EXCLUDE_LABELS = {
    "TRUNCATED",
    "BAD_OTHER",
    "BAD_AUDIO",
}


hard_exclude_utts = {
    utt
    for utt, info in manual.items()
    if info["label"]
    in HARD_EXCLUDE_LABELS
}


if len(hard_exclude_utts) != EXPECTED_HARD_EXCLUDES:
    raise RuntimeError(
        f"expected {EXPECTED_HARD_EXCLUDES} "
        f"manual hard excludes, "
        f"got {len(hard_exclude_utts)}; "
        f"manual counts={dict(manual_counts)}"
    )


#
# ------------------------------------------------------------
# Build canonical physical rows
# ------------------------------------------------------------
#
canonical = []

training_text_changed = 0


for m in manifest_rows:

    utt = m["utt_id"]

    mapped = mapped_by_utt[utt]
    inv = inventory_by_utt[utt]
    q = qc_by_utt[utt]

    #
    # Authoritative derived global coordinate comes
    # from c2_f30_manifest_with_global_row_idx.parquet.
    #
    global_row_idx = int(
        mapped["global_row_idx"]
    )

    #
    # Downloader inventory must carry the identical
    # full-dataset global coordinate.
    #
    if "global_row_idx" not in inv:
        raise RuntimeError(
            f"{utt}: inventory missing global_row_idx"
        )

    if (
        int(inv["global_row_idx"])
        != global_row_idx
    ):
        raise RuntimeError(
            f"{utt}: inventory global_row_idx mismatch: "
            f"{inv['global_row_idx']} "
            f"!= {global_row_idx}"
        )


    #
    # QC was derived from the same indexed manifest.
    #
    if "global_row_idx" in q:
        if (
            int(q["global_row_idx"])
            != global_row_idx
        ):
            raise RuntimeError(
                f"{utt}: QC global_row_idx mismatch"
            )


    #
    # Frozen source coordinates remain the
    # authoritative provenance identity.
    #
    for field in (
        "row_group",
        "row_in_group",
    ):
        if field in q:
            if int(m[field]) != int(q[field]):
                raise RuntimeError(
                    f"{utt}: QC {field} mismatch"
                )


    if (
        str(m["source_shard"])
        != str(q["source_shard"])
    ):
        raise RuntimeError(
            f"{utt}: source_shard mismatch"
        )


    metadata_duration = float(
        m["duration"]
    )

    inventory_manifest_duration = float(
        inv["manifest_duration"]
    )

    if abs(
        metadata_duration
        - inventory_manifest_duration
    ) > 1e-5:
        raise RuntimeError(
            f"{utt}: manifest duration mismatch"
        )


    physical_duration = float(
        inv["physical_duration"]
    )

    if abs(
        physical_duration
        - float(q["physical_duration"])
    ) > 1e-9:
        raise RuntimeError(
            f"{utt}: physical duration mismatch"
        )


    if (
        int(inv["sample_rate"]) != 24000
        or int(inv["channels"]) != 1
        or int(inv["sample_width"]) != 2
    ):
        raise RuntimeError(
            f"{utt}: unexpected physical format"
        )


    wav_path = (
        DATASET
        / "wavs"
        / inv["filename"]
    )

    if not wav_path.is_file():
        raise RuntimeError(
            f"missing WAV: {wav_path}"
        )


    denorm = normalize_space(
        m.get("text_denorm")
    )

    if not denorm:
        raise RuntimeError(
            f"{utt}: empty text_denorm"
        )


    training_text = (
        make_training_text(
            denorm
        )
    )

    if not training_text:
        raise RuntimeError(
            f"{utt}: empty training_text"
        )


    if training_text != denorm:
        training_text_changed += 1


    training_chars = len(
        training_text
    )

    training_words = len(
        training_text.split()
    )

    physical_training_cps = (
        training_chars
        / physical_duration
    )

    physical_training_wps = (
        training_words
        / physical_duration
    )


    training_rate_outlier = (
        physical_training_cps < MIN_CPS
        or physical_training_cps > MAX_CPS
        or physical_training_wps < MIN_WPS
        or physical_training_wps > MAX_WPS
    )


    #
    # Diagnostic risk flags only.
    #
    risk_duration_gt025 = (
        float(
            q["duration_abs_delta"]
        )
        > 0.25
    )

    risk_duration_gt1 = (
        float(
            q["duration_abs_delta"]
        )
        > 1.0
    )

    risk_eof_m40_lt50 = (
        float(
            q["trail_quiet_ms_m40"]
        )
        < 50.0
    )

    risk_eof_m40_lt100 = (
        float(
            q["trail_quiet_ms_m40"]
        )
        < 100.0
    )


    rare_technical_candidate = any([
        physical_duration < 2.5,
        float(
            q["qc_clipping_fraction"]
        ) > 0,
        float(
            q["qc_rms_dbfs"]
        ) < -30.0,
        float(
            q["qc_rms_dbfs"]
        ) > -12.0,
        float(
            q["qc_abs_dc_offset"]
        ) > 0.005,
        float(
            q["qc_exact_zero_fraction"]
        ) > 0.50,
        float(
            q["qc_proxy_snr_db"]
        ) < 10.0,
    ])


    manual_info = manual.get(
        utt
    )

    manual_label = (
        manual_info["label"]
        if manual_info
        else None
    )

    manual_source = (
        ",".join(
            manual_info["sources"]
        )
        if manual_info
        else None
    )


    hard_exclude = (
        utt in hard_exclude_utts
    )

    hard_exclude_reason = (
        f"manual_confirmed_{manual_label}"
        if hard_exclude
        else None
    )


    #
    # Carry every objective QC field produced by
    # waveform/trailing-activity audits.
    #
    objective_qc = {}

    for key, value in q.items():

        if (
            key.startswith("qc_")
            or key.startswith(
                "trail_quiet_ms_"
            )
            or key.startswith(
                "tail100_active_fraction_"
            )
            or key.startswith(
                "tail250_active_fraction_"
            )
            or key.startswith(
                "tail500_active_fraction_"
            )
        ):
            objective_qc[key] = value


    row = {
        **m,

        #
        # Derived physical/download provenance.
        #
        "global_row_idx":
            global_row_idx,

        #
        # Canonical text actually intended for
        # the MeloTTS RU frontend.
        #
        "training_text":
            training_text,

        "training_char_count":
            training_chars,

        "training_word_count":
            training_words,

        #
        # Physical inventory.
        #
        "filename":
            inv["filename"],

        "wav_path":
            str(wav_path),

        "wav_sha256":
            inv["sha256"],

        "wav_bytes":
            int(inv["bytes"]),

        "wav_frames":
            int(inv["frames"]),

        "sample_rate":
            int(inv["sample_rate"]),

        "channels":
            int(inv["channels"]),

        "sample_width_bytes":
            int(inv["sample_width"]),

        "download_status":
            inv["status"],

        #
        # Explicit metadata vs physical timing.
        #
        "metadata_duration":
            metadata_duration,

        "physical_duration":
            physical_duration,

        "duration_signed_delta":
            float(
                q[
                    "duration_signed_delta"
                ]
            ),

        "duration_abs_delta":
            float(
                q[
                    "duration_abs_delta"
                ]
            ),

        "duration_relative_pct":
            float(
                q[
                    "duration_relative_pct"
                ]
            ),

        #
        # Frozen-audit rate semantics based on
        # raw text counts.
        #
        "physical_chars_per_sec":
            float(
                q[
                    "physical_chars_per_sec"
                ]
            ),

        "physical_words_per_sec":
            float(
                q[
                    "physical_words_per_sec"
                ]
            ),

        #
        # MeloTTS-ready rate using training_text.
        #
        "physical_training_cps":
            physical_training_cps,

        "physical_training_wps":
            physical_training_wps,

        "qc_training_rate_outlier":
            training_rate_outlier,

        #
        # Risk flags: NEVER automatic exclusions.
        #
        "qc_duration_mismatch_gt_0_25":
            risk_duration_gt025,

        "qc_duration_mismatch_gt_1_00":
            risk_duration_gt1,

        "qc_eof_m40_quiet_lt_50ms":
            risk_eof_m40_lt50,

        "qc_eof_m40_quiet_lt_100ms":
            risk_eof_m40_lt100,

        "qc_rare_technical_candidate":
            rare_technical_candidate,

        #
        # Partial manual ground truth.
        #
        "manual_qc_reviewed":
            manual_info is not None,

        "manual_qc_label":
            manual_label,

        "manual_qc_source":
            manual_source,

        #
        # Only this controls membership in the
        # trainable physical subset.
        #
        "hard_exclude":
            hard_exclude,

        "hard_exclude_reason":
            hard_exclude_reason,

        **objective_qc,
    }

    canonical.append(row)


#
# ------------------------------------------------------------
# Final invariants
# ------------------------------------------------------------
#
if len(canonical) != EXPECTED_ROWS:
    raise RuntimeError(
        "canonical row count mismatch"
    )


if len({
    r["utt_id"]
    for r in canonical
}) != EXPECTED_ROWS:
    raise RuntimeError(
        "duplicate canonical utt_id"
    )


excluded = [
    r
    for r in canonical
    if r["hard_exclude"]
]

trainable = [
    r
    for r in canonical
    if not r["hard_exclude"]
]


if len(excluded) != EXPECTED_HARD_EXCLUDES:
    raise RuntimeError(
        f"excluded rows={len(excluded)}"
    )


if len(trainable) != (
    EXPECTED_ROWS
    - EXPECTED_HARD_EXCLUDES
):
    raise RuntimeError(
        f"trainable rows={len(trainable)}"
    )


#
# No objective flag is allowed to create an
# unreviewed hard exclusion.
#
for r in excluded:
    if not (
        r["manual_qc_reviewed"]
        and r["manual_qc_label"]
        in HARD_EXCLUDE_LABELS
    ):
        raise RuntimeError(
            f"non-manual exclusion detected: "
            f"{r['utt_id']}"
        )


#
# ------------------------------------------------------------
# Freeze artifact
# ------------------------------------------------------------
#
OUT.mkdir(
    parents=True,
    exist_ok=False,
)

ALL_PARQUET = (
    OUT
    / "manifest_all.parquet"
)

TRAIN_PARQUET = (
    OUT
    / "manifest_trainable.parquet"
)

EXCLUDED_TSV = (
    OUT
    / "excluded_manual_confirmed.tsv"
)

SUMMARY = (
    OUT
    / "summary.json"
)

README = (
    OUT
    / "README.md"
)

PROVENANCE = (
    OUT
    / "PROVENANCE_SHA256SUMS.txt"
)

SHA = (
    OUT
    / "SHA256SUMS.txt"
)


pq.write_table(
    pa.Table.from_pylist(
        canonical
    ),
    ALL_PARQUET,
    compression="zstd",
)

pq.write_table(
    pa.Table.from_pylist(
        trainable
    ),
    TRAIN_PARQUET,
    compression="zstd",
)


excluded_fields = [
    "utt_id",
    "source_video_id",
    "filename",
    "physical_duration",
    "duration_abs_delta",
    "manual_qc_label",
    "manual_qc_source",
    "hard_exclude_reason",
    "text",
    "training_text",
]


with EXCLUDED_TSV.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=excluded_fields,
        delimiter="\t",
    )

    writer.writeheader()

    for r in excluded:

        writer.writerow({
            field: r.get(field)
            for field in excluded_fields
        })


all_hours = sum(
    float(r["physical_duration"])
    for r in canonical
) / 3600.0

excluded_hours = sum(
    float(r["physical_duration"])
    for r in excluded
) / 3600.0

trainable_hours = sum(
    float(r["physical_duration"])
    for r in trainable
) / 3600.0


risk_counts = {
    "duration_mismatch_gt_0_25":
        sum(
            r[
                "qc_duration_mismatch_gt_0_25"
            ]
            for r in canonical
        ),

    "duration_mismatch_gt_1_00":
        sum(
            r[
                "qc_duration_mismatch_gt_1_00"
            ]
            for r in canonical
        ),

    "eof_m40_quiet_lt_50ms":
        sum(
            r[
                "qc_eof_m40_quiet_lt_50ms"
            ]
            for r in canonical
        ),

    "eof_m40_quiet_lt_100ms":
        sum(
            r[
                "qc_eof_m40_quiet_lt_100ms"
            ]
            for r in canonical
        ),

    "training_rate_outlier":
        sum(
            r[
                "qc_training_rate_outlier"
            ]
            for r in canonical
        ),

    "rare_technical_candidate":
        sum(
            r[
                "qc_rare_technical_candidate"
            ]
            for r in canonical
        ),
}


source_hashes = {
    str(FROZEN_MANIFEST):
        frozen_sha,

    str(MAPPED_MANIFEST):
        sha256_file(
            MAPPED_MANIFEST
        ),

    str(INVENTORY):
        sha256_file(
            INVENTORY
        ),

    str(QC):
        sha256_file(
            QC
        ),

    str(REVIEW_EOF):
        sha256_file(
            REVIEW_EOF
        ),

    str(REVIEW_HOLDOUT):
        sha256_file(
            REVIEW_HOLDOUT
        ),

    str(REVIEW_RARE):
        sha256_file(
            REVIEW_RARE
        ),
}


summary = {
    "artifact":
        "C2-F30-physical-v1",

    "rows_all":
        len(canonical),

    "rows_trainable":
        len(trainable),

    "rows_hard_excluded":
        len(excluded),

    "physical_hours_all":
        all_hours,

    "physical_hours_hard_excluded":
        excluded_hours,

    "physical_hours_trainable":
        trainable_hours,

    "sample_rate":
        24000,

    "channels":
        1,

    "sample_width_bytes":
        2,

    "frozen_selection_manifest_sha256":
        frozen_sha,

    "manual_review_unique_rows":
        len(manual),

    "manual_label_counts":
        dict(
            sorted(
                manual_counts.items()
            )
        ),

    "hard_exclude_policy":
        (
            "Only individually manually confirmed "
            "TRUNCATED/BAD_OTHER/BAD_AUDIO rows. "
            "No automatic exclusion from duration mismatch, "
            "EOF activity, speech rate, ASR, RMS, DC offset, "
            "or other diagnostic risk flags."
        ),

    "risk_counts_all_rows":
        risk_counts,

    "training_text_policy":
        (
            "Derived from text_denorm; preserve "
            "punctuation/case/ё; remove '+' stress markers "
            "and combining acute U+0301; normalize NFC/space."
        ),

    "training_text_rows_changed_from_text_denorm":
        training_text_changed,

    "split":
        None,

    "speaker_conditioning":
        None,

    "source_hashes":
        source_hashes,
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
    """# C2-F30-physical-v1

Canonical physical freeze of the C2-F30 YO-CPT selection.

- `manifest_all.parquet`: all 12,720 physically downloaded WAV rows.
- `manifest_trainable.parquet`: same catalog minus individually
  manually confirmed problematic rows.
- Diagnostic duration/EOF/rate/audio-QC flags are retained but do
  not cause automatic exclusion.
- `training_text` is derived from `text_denorm` with YO-CPT stress
  markup removed.
- Train/validation split and speaker-conditioning strategy are not
  encoded in this artifact and must be frozen separately.
""",
    encoding="utf-8",
)


with PROVENANCE.open(
    "w",
    encoding="utf-8",
) as f:

    for path, digest in sorted(
        source_hashes.items()
    ):
        f.write(
            f"{digest}  {path}\n"
        )


output_files = [
    ALL_PARQUET,
    TRAIN_PARQUET,
    EXCLUDED_TSV,
    SUMMARY,
    README,
    PROVENANCE,
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


#
# Verify our newly written hashes.
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
        OUT
        / filename
    )

    if actual != digest:
        raise RuntimeError(
            f"output SHA mismatch: {filename}"
        )


print(
    "===== C2-F30 PHYSICAL V1 FREEZE ====="
)

print(
    "frozen selection SHA256:",
    frozen_sha,
)

print()
print(
    "rows all:",
    len(canonical),
)

print(
    "rows trainable:",
    len(trainable),
)

print(
    "rows hard excluded:",
    len(excluded),
)

print()
print(
    "physical hours all:",
    round(
        all_hours,
        10,
    ),
)

print(
    "excluded hours:",
    round(
        excluded_hours,
        10,
    ),
)

print(
    "trainable hours:",
    round(
        trainable_hours,
        10,
    ),
)

print()
print(
    "manual labels:",
    dict(
        sorted(
            manual_counts.items()
        )
    ),
)

print(
    "training_text changed:",
    training_text_changed,
)

print()
print(
    "risk counts:",
    risk_counts,
)

print()
print(
    "manifest all:",
    ALL_PARQUET,
)

print(
    "manifest trainable:",
    TRAIN_PARQUET,
)

print(
    "excluded:",
    EXCLUDED_TSV,
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
    "C2-F30 PHYSICAL V1: PASS"
)
