from __future__ import annotations

import json
import math
import os
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path.home() / "projects/ru_melotts_research"

WAV_DIR = (
    Path.home()
    / "datasets/tts/yocpt_c2_f30_v1/wavs"
)

SRC = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_physical_duration_audit_v1"
    / "audit.parquet"
)

OUT = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_waveform_qc_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_ROWS = 12720
SR = 24000
WORKERS = min(
    8,
    os.cpu_count() or 1,
)

EPS = 1e-12


def db(value):
    return (
        20.0
        * math.log10(
            max(
                float(value),
                EPS,
            )
        )
    )


def rms_db(x):
    if not x.size:
        return -240.0

    rms = np.sqrt(
        np.mean(
            np.square(
                x,
                dtype=np.float64,
            )
        )
    )

    return db(rms)


def percentile(values, p):
    values = sorted(
        float(x)
        for x in values
    )

    pos = (
        (len(values) - 1)
        * p
    )

    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    f = pos - lo

    return (
        values[lo] * (1.0 - f)
        +
        values[hi] * f
    )


def quantiles(values):
    return {
        "min": percentile(values, 0.00),
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": percentile(values, 1.00),
    }


def edge_rms(
    x,
    seconds,
    tail=False,
):
    n = min(
        len(x),
        max(
            1,
            int(
                round(
                    SR * seconds
                )
            ),
        ),
    )

    part = (
        x[-n:]
        if tail
        else x[:n]
    )

    return rms_db(part)


def inspect_one(
    index,
    row,
):
    path = (
        WAV_DIR
        / row["filename"]
    )

    if not path.is_file():
        raise RuntimeError(
            f"missing WAV: {path}"
        )

    with wave.open(
        str(path),
        "rb",
    ) as wav:

        channels = (
            wav.getnchannels()
        )

        sample_rate = (
            wav.getframerate()
        )

        sample_width = (
            wav.getsampwidth()
        )

        frames = (
            wav.getnframes()
        )

        raw = wav.readframes(
            frames
        )


    if (
        channels,
        sample_rate,
        sample_width,
    ) != (
        1,
        24000,
        2,
    ):
        raise RuntimeError(
            f"{path.name}: "
            "unexpected format "
            f"{channels}/"
            f"{sample_rate}/"
            f"{sample_width}"
        )


    samples = np.frombuffer(
        raw,
        dtype="<i2",
    )

    if (
        len(samples) != frames
        or
        len(samples) == 0
    ):
        raise RuntimeError(
            f"{path.name}: "
            "invalid frame count"
        )


    samples_i32 = (
        samples.astype(
            np.int32,
            copy=False,
        )
    )

    x = (
        samples.astype(
            np.float32
        )
        / 32768.0
    )

    abs_samples = np.abs(
        samples_i32
    )


    #
    # 10 ms frame RMS distribution.
    #
    frame_samples = 240

    frame_count = (
        len(x)
        // frame_samples
    )

    framed = (
        x[
            :frame_count
            * frame_samples
        ]
        .reshape(
            frame_count,
            frame_samples,
        )
    )

    frame_rms = np.sqrt(
        np.mean(
            np.square(
                framed,
                dtype=np.float64,
            ),
            axis=1,
        )
    )

    frame_db = (
        20.0
        * np.log10(
            np.maximum(
                frame_rms,
                EPS,
            )
        )
    )

    frame_p10 = float(
        np.percentile(
            frame_db,
            10,
        )
    )

    frame_p50 = float(
        np.percentile(
            frame_db,
            50,
        )
    )

    frame_p90 = float(
        np.percentile(
            frame_db,
            90,
        )
    )


    metrics = {
        "qc_peak_dbfs":
            db(
                float(
                    abs_samples.max()
                )
                / 32768.0
            ),

        "qc_rms_dbfs":
            rms_db(x),

        "qc_abs_dc_offset":
            abs(
                float(
                    np.mean(
                        x,
                        dtype=np.float64,
                    )
                )
            ),

        "qc_clipping_fraction":
            float(
                np.mean(
                    abs_samples
                    >= 32760
                )
            ),

        "qc_exact_zero_fraction":
            float(
                np.mean(
                    samples == 0
                )
            ),

        "qc_frame_rms_p10_dbfs":
            frame_p10,

        "qc_frame_rms_p50_dbfs":
            frame_p50,

        "qc_frame_rms_p90_dbfs":
            frame_p90,

        "qc_proxy_snr_db":
            frame_p90
            - frame_p10,

        "qc_head_20ms_rms_dbfs":
            edge_rms(
                x,
                0.020,
            ),

        "qc_tail_20ms_rms_dbfs":
            edge_rms(
                x,
                0.020,
                tail=True,
            ),

        "qc_head_100ms_rms_dbfs":
            edge_rms(
                x,
                0.100,
            ),

        "qc_tail_100ms_rms_dbfs":
            edge_rms(
                x,
                0.100,
                tail=True,
            ),

        "qc_head_250ms_rms_dbfs":
            edge_rms(
                x,
                0.250,
            ),

        "qc_tail_250ms_rms_dbfs":
            edge_rms(
                x,
                0.250,
                tail=True,
            ),

        "qc_endpoint_abs_norm":
            (
                abs(
                    int(
                        samples[-1]
                    )
                )
                / 32768.0
            ),
    }


    return (
        index,
        {
            **row,
            **metrics,
        },
    )


METRIC_FIELDS = [
    "qc_peak_dbfs",
    "qc_rms_dbfs",
    "qc_abs_dc_offset",
    "qc_clipping_fraction",
    "qc_exact_zero_fraction",
    "qc_proxy_snr_db",
    "qc_head_20ms_rms_dbfs",
    "qc_tail_20ms_rms_dbfs",
    "qc_head_100ms_rms_dbfs",
    "qc_tail_100ms_rms_dbfs",
    "qc_head_250ms_rms_dbfs",
    "qc_tail_250ms_rms_dbfs",
    "qc_endpoint_abs_norm",
]


def cohort(rows):
    return {
        "rows":
            len(rows),

        "physical_hours":
            sum(
                float(
                    r[
                        "physical_duration"
                    ]
                )
                for r in rows
            )
            / 3600.0,

        "metrics": {
            field:
                quantiles(
                    [
                        r[field]
                        for r in rows
                    ]
                )

            for field
            in METRIC_FIELDS
        },
    }


rows = (
    pq.read_table(
        SRC
    )
    .to_pylist()
)


if len(rows) != EXPECTED_ROWS:
    raise RuntimeError(
        f"expected "
        f"{EXPECTED_ROWS} rows, "
        f"got {len(rows)}"
    )


if (
    len(
        {
            r["utt_id"]
            for r in rows
        }
    )
    != EXPECTED_ROWS
):
    raise RuntimeError(
        "duplicate utt_id"
    )


print(
    "===== C2-F30 "
    "WAVEFORM QC V1 ====="
)

print(
    "rows:",
    len(rows),
)

print(
    "workers:",
    WORKERS,
)


results = [
    None
] * len(rows)


with ThreadPoolExecutor(
    max_workers=WORKERS
) as pool:

    futures = [
        pool.submit(
            inspect_one,
            i,
            row,
        )

        for i, row
        in enumerate(rows)
    ]


    for done, future in enumerate(
        as_completed(
            futures
        ),
        start=1,
    ):

        index, result = (
            future.result()
        )

        results[index] = result


        if (
            done == 1
            or
            done % 500 == 0
            or
            done == len(rows)
        ):
            print(
                f"[{done:05d}/"
                f"{len(rows):05d}]"
            )


if any(
    r is None
    for r in results
):
    raise RuntimeError(
        "incomplete QC result"
    )


qc_path = (
    OUT
    / "audio_qc.parquet"
)

pq.write_table(
    pa.Table.from_pylist(
        results
    ),
    qc_path,
    compression="zstd",
)


cohort_clean = [
    r
    for r in results
    if float(
        r[
            "duration_abs_delta"
        ]
    ) <= 0.25
]

cohort_mid = [
    r
    for r in results
    if (
        0.25
        <
        float(
            r[
                "duration_abs_delta"
            ]
        )
        <=
        1.0
    )
]

cohort_tail = [
    r
    for r in results
    if float(
        r[
            "duration_abs_delta"
        ]
    ) > 1.0
]


summary = {
    "rows":
        len(results),

    "physical_hours":
        sum(
            float(
                r[
                    "physical_duration"
                ]
            )
            for r in results
        )
        / 3600.0,

    "cohorts": {
        "abs_delta_le_0.25":
            cohort(
                cohort_clean
            ),

        "abs_delta_0.25_to_1.0":
            cohort(
                cohort_mid
            ),

        "abs_delta_gt_1.0":
            cohort(
                cohort_tail
            ),
    },

    "note":
        "Measurement only: "
        "no keep/drop flags and "
        "no WAV modification.",
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
    "===== COHORT COMPARISON ====="
)

for name, data in (
    summary["cohorts"].items()
):

    m = data["metrics"]

    print()
    print(
        name,
        "rows=",
        data["rows"],
    )

    print(
        "  rms_dbfs"
        " p05/p50/p95 =",
        round(
            m[
                "qc_rms_dbfs"
            ]["p05"],
            3,
        ),
        round(
            m[
                "qc_rms_dbfs"
            ]["p50"],
            3,
        ),
        round(
            m[
                "qc_rms_dbfs"
            ]["p95"],
            3,
        ),
    )

    print(
        "  proxy_snr"
        " p05/p50/p95 =",
        round(
            m[
                "qc_proxy_snr_db"
            ]["p05"],
            3,
        ),
        round(
            m[
                "qc_proxy_snr_db"
            ]["p50"],
            3,
        ),
        round(
            m[
                "qc_proxy_snr_db"
            ]["p95"],
            3,
        ),
    )

    print(
        "  tail100_dbfs"
        " p05/p50/p95 =",
        round(
            m[
                "qc_tail_100ms_rms_dbfs"
            ]["p05"],
            3,
        ),
        round(
            m[
                "qc_tail_100ms_rms_dbfs"
            ]["p50"],
            3,
        ),
        round(
            m[
                "qc_tail_100ms_rms_dbfs"
            ]["p95"],
            3,
        ),
    )

    print(
        "  clipping"
        " p95/p99/max =",
        m[
            "qc_clipping_fraction"
        ]["p95"],
        m[
            "qc_clipping_fraction"
        ]["p99"],
        m[
            "qc_clipping_fraction"
        ]["max"],
    )

    print(
        "  abs_dc"
        " p95/p99/max =",
        m[
            "qc_abs_dc_offset"
        ]["p95"],
        m[
            "qc_abs_dc_offset"
        ]["p99"],
        m[
            "qc_abs_dc_offset"
        ]["max"],
    )

    print(
        "  endpoint_abs"
        " p95/p99/max =",
        m[
            "qc_endpoint_abs_norm"
        ]["p95"],
        m[
            "qc_endpoint_abs_norm"
        ]["p99"],
        m[
            "qc_endpoint_abs_norm"
        ]["max"],
    )


print()
print(
    "audio_qc:",
    qc_path,
)

print(
    "summary:",
    summary_path,
)

print()
print(
    "C2-F30 WAVEFORM QC V1: PASS"
)
