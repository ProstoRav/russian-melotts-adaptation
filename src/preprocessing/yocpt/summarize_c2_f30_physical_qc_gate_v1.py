from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT = Path.home() / "projects/ru_melotts_research"

SRC = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_trailing_activity_v1"
    / "trailing_activity.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_physical_qc_gate_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY = OUT / "summary.json"

rows = pq.read_table(SRC).to_pylist()

if len(rows) != 12720:
    raise RuntimeError(
        f"expected 12720 rows, got {len(rows)}"
    )


FIELDS = [
    "physical_duration",
    "duration_abs_delta",
    "qc_peak_dbfs",
    "qc_rms_dbfs",
    "qc_abs_dc_offset",
    "qc_clipping_fraction",
    "qc_exact_zero_fraction",
    "qc_proxy_snr_db",
    "trail_quiet_ms_m40",
]

missing = [
    x
    for x in FIELDS
    if x not in rows[0]
]

if missing:
    raise RuntimeError(
        f"missing fields: {missing}"
    )


def values(field):
    return np.asarray(
        [
            float(r[field])
            for r in rows
        ],
        dtype=np.float64,
    )


def q(field):
    x = values(field)

    return {
        "min": float(np.min(x)),
        "p001": float(np.percentile(x, 0.1)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "p999": float(np.percentile(x, 99.9)),
        "max": float(np.max(x)),
    }


def count(predicate):
    return sum(
        bool(predicate(r))
        for r in rows
    )


checks = {
    #
    # Physical duration.
    #
    "duration_lt_2_0":
        count(
            lambda r:
                float(r["physical_duration"]) < 2.0
        ),

    "duration_lt_2_5":
        count(
            lambda r:
                float(r["physical_duration"]) < 2.5
        ),

    "duration_gt_12":
        count(
            lambda r:
                float(r["physical_duration"]) > 12.0
        ),

    #
    # Amplitude / clipping.
    #
    "clipping_any":
        count(
            lambda r:
                float(r["qc_clipping_fraction"]) > 0
        ),

    "peak_gt_minus_1_dbfs":
        count(
            lambda r:
                float(r["qc_peak_dbfs"]) > -1.0
        ),

    "peak_gt_minus_0_25_dbfs":
        count(
            lambda r:
                float(r["qc_peak_dbfs"]) > -0.25
        ),

    #
    # RMS extremes.
    #
    "rms_lt_minus_30_dbfs":
        count(
            lambda r:
                float(r["qc_rms_dbfs"]) < -30.0
        ),

    "rms_lt_minus_26_dbfs":
        count(
            lambda r:
                float(r["qc_rms_dbfs"]) < -26.0
        ),

    "rms_gt_minus_14_dbfs":
        count(
            lambda r:
                float(r["qc_rms_dbfs"]) > -14.0
        ),

    "rms_gt_minus_12_dbfs":
        count(
            lambda r:
                float(r["qc_rms_dbfs"]) > -12.0
        ),

    #
    # DC offset.
    #
    "abs_dc_gt_0_001":
        count(
            lambda r:
                float(r["qc_abs_dc_offset"]) > 0.001
        ),

    "abs_dc_gt_0_005":
        count(
            lambda r:
                float(r["qc_abs_dc_offset"]) > 0.005
        ),

    "abs_dc_gt_0_01":
        count(
            lambda r:
                float(r["qc_abs_dc_offset"]) > 0.01
        ),

    #
    # Exact digital zeros.
    #
    "zero_fraction_gt_0_25":
        count(
            lambda r:
                float(r["qc_exact_zero_fraction"]) > 0.25
        ),

    "zero_fraction_gt_0_50":
        count(
            lambda r:
                float(r["qc_exact_zero_fraction"]) > 0.50
        ),

    #
    # Crude frame-energy dynamic-range proxy.
    #
    "proxy_snr_lt_10":
        count(
            lambda r:
                float(r["qc_proxy_snr_db"]) < 10.0
        ),

    "proxy_snr_lt_15":
        count(
            lambda r:
                float(r["qc_proxy_snr_db"]) < 15.0
        ),

    "proxy_snr_lt_20":
        count(
            lambda r:
                float(r["qc_proxy_snr_db"]) < 20.0
        ),

    #
    # Already-established risk flags.
    # These are explicitly NOT drop criteria.
    #
    "duration_delta_gt_0_25":
        count(
            lambda r:
                float(r["duration_abs_delta"]) > 0.25
        ),

    "duration_delta_gt_1":
        count(
            lambda r:
                float(r["duration_abs_delta"]) > 1.0
        ),

    "quiet40_lt_50ms":
        count(
            lambda r:
                float(r["trail_quiet_ms_m40"]) < 50
        ),

    "quiet40_lt_100ms":
        count(
            lambda r:
                float(r["trail_quiet_ms_m40"]) < 100
        ),
}


#
# Conservative rare-anomaly union.
# This is only a review candidate set.
# It is NOT a hard-exclude set.
#
def rare_candidate(r):
    return any([
        float(r["physical_duration"]) < 2.5,
        float(r["qc_clipping_fraction"]) > 0,
        float(r["qc_rms_dbfs"]) < -30.0,
        float(r["qc_rms_dbfs"]) > -12.0,
        float(r["qc_abs_dc_offset"]) > 0.005,
        float(r["qc_exact_zero_fraction"]) > 0.50,
        float(r["qc_proxy_snr_db"]) < 10.0,
    ])


rare = [
    r
    for r in rows
    if rare_candidate(r)
]


def anomaly_score(r):
    score = 0.0

    dur = float(r["physical_duration"])
    rms = float(r["qc_rms_dbfs"])
    dc = float(r["qc_abs_dc_offset"])
    zero = float(r["qc_exact_zero_fraction"])
    snr = float(r["qc_proxy_snr_db"])
    clip = float(r["qc_clipping_fraction"])

    if dur < 2.5:
        score += (2.5 - dur) * 10

    if rms < -30:
        score += (-30 - rms)

    if rms > -12:
        score += (rms + 12)

    score += dc * 1000
    score += zero * 10

    if snr < 10:
        score += (10 - snr)

    if clip > 0:
        score += 100 + clip * 1000

    return score


rare_sorted = sorted(
    rare,
    key=anomaly_score,
    reverse=True,
)


summary = {
    "rows": len(rows),

    "physical_hours":
        sum(
            float(r["physical_duration"])
            for r in rows
        )
        / 3600.0,

    "quantiles": {
        field: q(field)
        for field in FIELDS
    },

    "diagnostic_counts":
        checks,

    "rare_review_candidate_rows":
        len(rare),

    "note":
        (
            "All thresholds in this artifact are diagnostic. "
            "No keep/drop decision is encoded."
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


print(
    "===== C2-F30 PHYSICAL QC GATE V1 ====="
)

print(
    "rows:",
    len(rows),
)

print(
    "physical_hours:",
    round(
        summary["physical_hours"],
        10,
    ),
)


print()
print(
    "===== DIAGNOSTIC COUNTS ====="
)

for name, value in checks.items():
    print(
        f"{name:30s}",
        value,
    )


print()
print(
    "===== KEY QUANTILES ====="
)

for field in (
    "physical_duration",
    "qc_peak_dbfs",
    "qc_rms_dbfs",
    "qc_abs_dc_offset",
    "qc_exact_zero_fraction",
    "qc_proxy_snr_db",
):

    x = summary[
        "quantiles"
    ][field]

    print()
    print(field)

    print(
        "  min/p01/p50/p99/max =",
        round(x["min"], 6),
        round(x["p01"], 6),
        round(x["p50"], 6),
        round(x["p99"], 6),
        round(x["max"], 6),
    )


print()
print(
    "rare review candidates:",
    len(rare),
)


print()
print(
    "===== TOP RARE TECHNICAL CANDIDATES ====="
)

for r in rare_sorted[:50]:

    print(
        r["utt_id"],
        "dur=",
        round(
            float(r["physical_duration"]),
            3,
        ),
        "rms=",
        round(
            float(r["qc_rms_dbfs"]),
            2,
        ),
        "peak=",
        round(
            float(r["qc_peak_dbfs"]),
            2,
        ),
        "dc=",
        round(
            float(r["qc_abs_dc_offset"]),
            6,
        ),
        "zero=",
        round(
            float(r["qc_exact_zero_fraction"]),
            4,
        ),
        "snr=",
        round(
            float(r["qc_proxy_snr_db"]),
            2,
        ),
        "delta=",
        round(
            float(r["duration_abs_delta"]),
            3,
        ),
        "file=",
        r["filename"],
    )


print()
print(
    "summary:",
    SUMMARY,
)

print()
print(
    "C2-F30 PHYSICAL QC GATE V1: PASS"
)
