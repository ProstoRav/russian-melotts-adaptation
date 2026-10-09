import copy
import csv
import hashlib
import itertools
import json
import statistics
import sys
from pathlib import Path

import jiwer
import numpy as np


dst = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()

out.mkdir(parents=True, exist_ok=True)


SYSTEMS = [
    "melotts_yocpt_e27",
    "melotts_dialogs_e38_neutral",
    "melotts_dialogs_e38_positive",
    "dialogs_ru_vits2",
    "silero_v5_5_ru",
    "xtts_v2_spk298",
    "piper_ru_irina_medium",
]

DISPLAY = {
    "melotts_yocpt_e27":
        "MeloTTS YO-CPT E27",

    "melotts_dialogs_e38_neutral":
        "MeloTTS E38 neutral",

    "melotts_dialogs_e38_positive":
        "MeloTTS E38 positive",

    "dialogs_ru_vits2":
        "dialogs-ru-vits2",

    "silero_v5_5_ru":
        "Silero v5_5_ru",

    "xtts_v2_spk298":
        "XTTS-v2 spk298",

    "piper_ru_irina_medium":
        "Piper irina-medium",
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


# ============================================================
# Frozen source metadata
# ============================================================

contract_path = (
    dst /
    "benchmark_contract.draft.json"
)

systems_path = (
    dst /
    "systems.draft.json"
)

prompts_path = (
    dst /
    "prompts.benchmark.jsonl"
)

for p in [
    contract_path,
    systems_path,
    prompts_path,
]:
    assert p.is_file(), p


contract = read_json(
    contract_path
)

protocols = contract.get(
    "input_protocols",
    {},
)

if (
    "canonical_cross_model_stress_target"
    not in protocols
):
    raise RuntimeError(
        "benchmark contract still has "
        "the obsolete input policy: "
        f"{list(protocols)}"
    )


systems_doc = read_json(
    systems_path
)

system_meta = {
    x["id"]: x
    for x in systems_doc["systems"]
}

for sid in SYSTEMS:
    assert sid in system_meta

    unresolved = (
        "TO_RESOLVE"
        in json.dumps(
            system_meta[sid],
            ensure_ascii=False,
        )
    )

    if unresolved:
        raise RuntimeError(
            f"systems.draft.json still "
            f"contains TO_RESOLVE for {sid}"
        )


prompts = read_jsonl(
    prompts_path
)

assert [
    x["id"]
    for x in prompts
] == [
    f"S{i:02d}"
    for i in range(1, 10)
]

PROMPT_IDS = [
    x["id"]
    for x in prompts
]


# ============================================================
# Quality inputs
# ============================================================

WHISPER_DIR = (
    dst /
    "quality/"
    "whisper_small_ru_wer_cer"
)

UTMOS_DIR = (
    dst /
    "quality/"
    "utmosv2_5fold_5tta_gpu"
)

DISTILL_DIR = (
    dst /
    "quality/"
    "distillmos_v7_gpu"
)


def quality_files(root):
    pp = root / "per_prompt.jsonl"
    summary = root / "system_summary.json"

    assert pp.is_file(), pp
    assert summary.is_file(), summary

    return pp, summary


W_PP_PATH, W_SUM_PATH = quality_files(
    WHISPER_DIR
)

U_PP_PATH, U_SUM_PATH = quality_files(
    UTMOS_DIR
)

D_PP_PATH, D_SUM_PATH = quality_files(
    DISTILL_DIR
)


W_ROWS = read_jsonl(
    W_PP_PATH
)

U_ROWS = read_jsonl(
    U_PP_PATH
)

D_ROWS = read_jsonl(
    D_PP_PATH
)

W_SUM = read_json(
    W_SUM_PATH
)

U_SUM = read_json(
    U_SUM_PATH
)

D_SUM = read_json(
    D_SUM_PATH
)


def make_index(rows):
    result = {}

    for row in rows:
        key = (
            row["system_id"],
            row["prompt_id"],
        )

        assert key not in result
        result[key] = row

    return result


W = make_index(W_ROWS)
U = make_index(U_ROWS)
D = make_index(D_ROWS)


EXPECTED_KEYS = {
    (sid, pid)
    for sid in SYSTEMS
    for pid in PROMPT_IDS
}

assert set(W) == EXPECTED_KEYS
assert set(U) == EXPECTED_KEYS
assert set(D) == EXPECTED_KEYS


# Exact same canonical WAV must underlie all quality axes.
for key in EXPECTED_KEYS:
    assert (
        W[key]["wav_sha256"]
        ==
        U[key]["wav_sha256"]
        ==
        D[key]["wav_sha256"]
    ), key


# ============================================================
# Performance discovery
# ============================================================

PERF_ROOT = (
    dst /
    "performance"
)

assert PERF_ROOT.is_dir()


perf_candidates = []

for path in sorted(
    PERF_ROOT.rglob(
        "summary.json"
    )
):
    try:
        obj = read_json(path)
    except Exception:
        continue

    sid = obj.get(
        "system_id"
    )

    if sid not in SYSTEMS:
        for candidate in SYSTEMS:
            if candidate in str(path):
                sid = candidate
                break

    if sid not in SYSTEMS:
        continue

    mode = obj.get(
        "mode"
    )

    if mode is None:
        dirname = str(
            path.parent
        ).lower()

        if (
            "nonstream"
            in dirname
        ):
            mode = "nonstream"

        elif (
            "stream"
            in dirname
        ):
            mode = "stream"

        else:
            mode = "nonstream"

    perf_candidates.append(
        (
            sid,
            mode,
            path,
            obj,
        )
    )


def median_summary(
    obj,
    names,
):
    for name in names:
        value = obj.get(name)

        if (
            isinstance(
                value,
                dict,
            )
            and
            "median"
            in value
        ):
            return float(
                value["median"]
            )

        if isinstance(
            value,
            (int, float),
        ):
            return float(value)

    return None


def median_rows(
    path,
    names,
):
    if not path.is_file():
        return None

    rows = read_jsonl(path)

    for name in names:
        values = [
            float(x[name])
            for x in rows
            if isinstance(
                x.get(name),
                (int, float),
            )
        ]

        if values:
            return float(
                statistics.median(
                    values
                )
            )

    return None


def primary_performance(sid):
    candidates = [
        x
        for x in perf_candidates
        if (
            x[0] == sid
            and
            x[1] != "stream"
        )
    ]

    if (
        sid
        == "xtts_v2_spk298"
    ):
        explicit = [
            x
            for x in candidates
            if x[1] == "nonstream"
        ]

        if explicit:
            candidates = explicit

    if not candidates:
        raise RuntimeError(
            "No official non-stream "
            f"performance summary: {sid}"
        )

    # Prefer a summary that directly contains RTF.
    candidates.sort(
        key=lambda x: (
            0
            if median_summary(
                x[3],
                [
                    "RTF",
                    "rtf",
                ],
            )
            is not None
            else 1,
            len(str(x[2])),
        )
    )

    _, _, summary_path, obj = (
        candidates[0]
    )

    pp_path = (
        summary_path.parent /
        "per_prompt.jsonl"
    )

    rtf = median_summary(
        obj,
        [
            "RTF",
            "rtf",
        ],
    )

    if rtf is None:
        rtf = median_rows(
            pp_path,
            [
                "RTF",
                "rtf",
            ],
        )

    latency = median_summary(
        obj,
        [
            "full_waveform_availability_seconds",
            "wall_inference_seconds",
            "wall_seconds",
            "inference_seconds",
        ],
    )

    if latency is None:
        latency = median_rows(
            pp_path,
            [
                "full_waveform_availability_seconds",
                "wall_inference_seconds",
                "wall_seconds",
                "inference_seconds",
                "infer_seconds",
            ],
        )

    if (
        rtf is None
        or latency is None
    ):
        raise RuntimeError(
            f"Could not extract performance "
            f"from {summary_path}: "
            f"RTF={rtf}, latency={latency}"
        )

    return {
        "summary_path":
            summary_path,

        "summary_sha256":
            sha256(
                summary_path
            ),

        "rtf_median":
            rtf,

        "full_waveform_latency_median_seconds":
            latency,
    }


PERF = {
    sid:
        primary_performance(
            sid
        )
    for sid in SYSTEMS
}


# XTTS true native streaming metrics.
xtts_stream_candidates = [
    x
    for x in perf_candidates
    if (
        x[0]
        == "xtts_v2_spk298"
        and
        x[1] == "stream"
    )
]

XTTS_STREAM = None

if xtts_stream_candidates:
    _, _, path, obj = (
        xtts_stream_candidates[0]
    )

    pp_path = (
        path.parent /
        "per_prompt.jsonl"
    )

    ttfa = median_summary(
        obj,
        [
            "TTFA_seconds",
            "ttfa_seconds",
        ],
    )

    if ttfa is None:
        ttfa = median_rows(
            pp_path,
            [
                "TTFA_seconds",
                "ttfa_seconds",
                "TTFA",
            ],
        )

    stream_rtf = median_summary(
        obj,
        [
            "stream_RTF",
            "RTF",
            "rtf",
        ],
    )

    if stream_rtf is None:
        stream_rtf = median_rows(
            pp_path,
            [
                "stream_RTF",
                "RTF",
                "rtf",
            ],
        )

    assert ttfa is not None

    XTTS_STREAM = {
        "summary_path":
            path,

        "summary_sha256":
            sha256(path),

        "ttfa_median_seconds":
            ttfa,

        "stream_rtf_median":
            stream_rtf,
    }


# ============================================================
# Headline 7-system table
# ============================================================

table_rows = []

for sid in SYSTEMS:
    ws = W_SUM["systems"][sid]
    us = U_SUM["systems"][sid]
    ds = D_SUM["systems"][sid]

    table_rows.append({
        "system_id":
            sid,

        "display_name":
            DISPLAY[sid],

        "micro_WER":
            float(
                ws["micro_WER"]
            ),

        "micro_CER":
            float(
                ws["micro_CER"]
            ),

        "UTMOSv2_mean":
            float(
                us["mean"]
            ),

        "UTMOSv2_median":
            float(
                us["median"]
            ),

        "DistillMOS_mean":
            float(
                ds["mean"]
            ),

        "DistillMOS_median":
            float(
                ds["median"]
            ),

        "RTF_median_CPU2":
            PERF[sid][
                "rtf_median"
            ],

        "full_waveform_latency_median_seconds_CPU2":
            PERF[sid][
                "full_waveform_latency_median_seconds"
            ],

        "native_streaming_TTFA_median_seconds_CPU2":
            (
                XTTS_STREAM[
                    "ttfa_median_seconds"
                ]
                if (
                    sid
                    == "xtts_v2_spk298"
                    and
                    XTTS_STREAM
                )
                else None
            ),

        "native_streaming_RTF_median_CPU2":
            (
                XTTS_STREAM[
                    "stream_rtf_median"
                ]
                if (
                    sid
                    == "xtts_v2_spk298"
                    and
                    XTTS_STREAM
                )
                else None
            ),
    })


TABLE_PATH = (
    out /
    "benchmark_table.csv"
)

with TABLE_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            table_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        table_rows
    )


# ============================================================
# Joined 63-row quality table
# ============================================================

joined = []

for sid in SYSTEMS:
    for pid in PROMPT_IDS:
        key = (sid, pid)

        joined.append({
            "system_id":
                sid,

            "prompt_id":
                pid,

            "wav_sha256":
                W[key][
                    "wav_sha256"
                ],

            "WER":
                float(
                    W[key]["WER"]
                ),

            "CER":
                float(
                    W[key]["CER"]
                ),

            "UTMOSv2":
                float(
                    U[key]["score"]
                ),

            "DistillMOS":
                float(
                    D[key]["score"]
                ),

            "reference_normalized":
                W[key][
                    "reference_normalized"
                ],

            "hypothesis_normalized":
                W[key][
                    "hypothesis_normalized"
                ],
        })


JOINED_PATH = (
    out /
    "per_prompt_joined.csv"
)

with JOINED_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            joined[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(joined)


# ============================================================
# Deterministic paired bootstrap
#
# Same S01-S09 resampled together.
# WER/CER output is in percentage points.
# AutoMOS output is in MOS points.
# ============================================================

METRICS = {
    "WER": (
        W,
        "WER",
        100.0,
        "percentage_points",
        "lower",
    ),

    "CER": (
        W,
        "CER",
        100.0,
        "percentage_points",
        "lower",
    ),

    "UTMOSv2": (
        U,
        "score",
        1.0,
        "MOS_points",
        "higher",
    ),

    "DistillMOS": (
        D,
        "score",
        1.0,
        "MOS_points",
        "higher",
    ),
}

BOOT_N = 50000
BOOT_SEED = 20261006

bootstrap_rows = []

pair_counter = 0

for metric_index, (
    metric,
    spec,
) in enumerate(
    METRICS.items()
):
    (
        source,
        field,
        scale,
        unit,
        better,
    ) = spec

    for a, b in itertools.combinations(
        SYSTEMS,
        2,
    ):
        av = np.asarray(
            [
                float(
                    source[
                        (a, pid)
                    ][field]
                )
                for pid in PROMPT_IDS
            ],
            dtype=np.float64,
        )

        bv = np.asarray(
            [
                float(
                    source[
                        (b, pid)
                    ][field]
                )
                for pid in PROMPT_IDS
            ],
            dtype=np.float64,
        )

        diff = av - bv

        seed = (
            BOOT_SEED
            + metric_index * 1000
            + pair_counter
        )

        rng = (
            np.random.default_rng(
                seed
            )
        )

        indices = rng.integers(
            0,
            9,
            size=(
                BOOT_N,
                9,
            ),
        )

        boot = (
            diff[
                indices
            ].mean(
                axis=1
            )
        )

        lo, hi = np.quantile(
            boot,
            [
                0.025,
                0.975,
            ],
        )

        bootstrap_rows.append({
            "metric":
                metric,

            "system_A":
                a,

            "system_B":
                b,

            "difference_definition":
                "A_minus_B",

            "better_direction":
                better,

            "unit":
                unit,

            "mean_difference":
                float(
                    diff.mean()
                ) * scale,

            "ci95_low":
                float(lo) * scale,

            "ci95_high":
                float(hi) * scale,

            "n_prompts":
                9,

            "bootstrap_resamples":
                BOOT_N,

            "bootstrap_seed":
                seed,
        })

        pair_counter += 1


BOOT_PATH = (
    out /
    "paired_bootstrap.csv"
)

with BOOT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            bootstrap_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        bootstrap_rows
    )


# ============================================================
# S04 numeric-orthography sensitivity
# ============================================================

def micro_counts(rows):
    wc = {
        "hits": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }

    cc = {
        "hits": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }

    for row in rows:
        wr = jiwer.process_words(
            row[
                "reference_normalized"
            ],
            row[
                "hypothesis_normalized"
            ],
        )

        cr = (
            jiwer
            .process_characters(
                row[
                    "reference_cer_normalized"
                ],
                row[
                    "hypothesis_cer_normalized"
                ],
            )
        )

        for key in wc:
            wc[key] += int(
                getattr(
                    wr,
                    key,
                )
            )

            cc[key] += int(
                getattr(
                    cr,
                    key,
                )
            )

    word_ref = (
        wc["hits"]
        + wc["substitutions"]
        + wc["deletions"]
    )

    char_ref = (
        cc["hits"]
        + cc["substitutions"]
        + cc["deletions"]
    )

    word_errors = (
        wc["substitutions"]
        + wc["deletions"]
        + wc["insertions"]
    )

    char_errors = (
        cc["substitutions"]
        + cc["deletions"]
        + cc["insertions"]
    )

    return {
        "micro_WER":
            word_errors
            / word_ref,

        "micro_CER":
            char_errors
            / char_ref,

        "word_errors":
            word_errors,

        "char_errors":
            char_errors,
    }


s04_rows = []

for sid in SYSTEMS:
    all_rows = [
        W[(sid, pid)]
        for pid in PROMPT_IDS
    ]

    without_s04 = [
        W[(sid, pid)]
        for pid in PROMPT_IDS
        if pid != "S04"
    ]

    s04_only = [
        W[(sid, "S04")]
    ]

    primary = micro_counts(
        all_rows
    )

    sensitivity = micro_counts(
        without_s04
    )

    s04 = micro_counts(
        s04_only
    )

    # Audit that recomputation reproduces
    # the frozen Whisper summary exactly.
    assert abs(
        primary["micro_WER"]
        -
        float(
            W_SUM[
                "systems"
            ][sid][
                "micro_WER"
            ]
        )
    ) < 1e-12

    assert abs(
        primary["micro_CER"]
        -
        float(
            W_SUM[
                "systems"
            ][sid][
                "micro_CER"
            ]
        )
    ) < 1e-12

    s04_rows.append({
        "system_id":
            sid,

        "primary_micro_WER":
            primary[
                "micro_WER"
            ],

        "primary_micro_CER":
            primary[
                "micro_CER"
            ],

        "micro_WER_without_S04":
            sensitivity[
                "micro_WER"
            ],

        "micro_CER_without_S04":
            sensitivity[
                "micro_CER"
            ],

        "S04_word_error_share":
            (
                s04["word_errors"]
                /
                primary[
                    "word_errors"
                ]
                if primary[
                    "word_errors"
                ]
                else 0.0
            ),

        "S04_char_error_share":
            (
                s04["char_errors"]
                /
                primary[
                    "char_errors"
                ]
                if primary[
                    "char_errors"
                ]
                else 0.0
            ),
    })


S04_PATH = (
    out /
    "s04_numeric_orthography_sensitivity.csv"
)

with S04_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(
            s04_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        s04_rows
    )


# ============================================================
# Frozen contract copy
#
# Do not modify source draft.
# Also replace any stale reporting rules from
# the earlier plain-input protocol.
# ============================================================

frozen_contract = (
    copy.deepcopy(
        contract
    )
)

frozen_contract[
    "status"
] = "FROZEN_FINAL_RESULTS"

frozen_contract[
    "final_artifact"
] = (
    "cross_model_tts_benchmark_v1"
)

frozen_contract[
    "reporting_rules"
] = [
    (
        "Headline cross-model results use "
        "canonical_cross_model_stress_target."
    ),
    (
        "Project MeloTTS systems receive "
        "literal text_stressed with U+0301; "
        "external systems use text_plain "
        "through their native frontend "
        "pronunciation/stress handling."
    ),
    (
        "WER/CER always use text_plain "
        "as the ASR reference."
    ),
    (
        "WER/CER use the single pinned "
        "Whisper-small evaluator and "
        "ru_simple_v1 normalization."
    ),
    (
        "UTMOSv2 and DistillMOS are "
        "automatic MOS predictions, "
        "not human MOS."
    ),
    (
        "S04-excluded WER/CER values are "
        "diagnostic sensitivity analysis "
        "only and must not replace the "
        "primary metrics."
    ),
    (
        "Performance results are local "
        "CPU two-thread measurements on "
        "the same machine."
    ),
    (
        "Only native streaming systems "
        "receive a streaming TTFA entry."
    ),
]


frozen_contract[
    "quality_status"
] = {
    "Whisper_small_WER_CER":
        "PASS",

    "UTMOSv2_5fold_5tta":
        "PASS",

    "DistillMOS_v7":
        "PASS",

    "canonical_wavs":
        63,

    "systems":
        7,

    "prompts_per_system":
        9,
}


CONTRACT_OUT = (
    out /
    "benchmark_contract.frozen.json"
)

CONTRACT_OUT.write_text(
    json.dumps(
        frozen_contract,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


frozen_systems = (
    copy.deepcopy(
        systems_doc
    )
)

frozen_systems[
    "status"
] = "FROZEN_FINAL_RESULTS"


SYSTEMS_OUT = (
    out /
    "systems.frozen.json"
)

SYSTEMS_OUT.write_text(
    json.dumps(
        frozen_systems,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


# ============================================================
# Machine-readable summary
# ============================================================

summary = {
    "benchmark_id":
        "cross_model_tts_benchmark_v1",

    "status":
        "FROZEN_FINAL_RESULTS",

    "systems": {
        x["system_id"]: x
        for x in table_rows
    },

    "quality_inputs": {
        "Whisper_small": {
            "per_prompt":
                str(
                    W_PP_PATH
                ),

            "per_prompt_sha256":
                sha256(
                    W_PP_PATH
                ),

            "summary":
                str(
                    W_SUM_PATH
                ),

            "summary_sha256":
                sha256(
                    W_SUM_PATH
                ),
        },

        "UTMOSv2": {
            "per_prompt":
                str(
                    U_PP_PATH
                ),

            "per_prompt_sha256":
                sha256(
                    U_PP_PATH
                ),

            "summary":
                str(
                    U_SUM_PATH
                ),

            "summary_sha256":
                sha256(
                    U_SUM_PATH
                ),
        },

        "DistillMOS": {
            "per_prompt":
                str(
                    D_PP_PATH
                ),

            "per_prompt_sha256":
                sha256(
                    D_PP_PATH
                ),

            "summary":
                str(
                    D_SUM_PATH
                ),

            "summary_sha256":
                sha256(
                    D_SUM_PATH
                ),
        },
    },

    "performance_inputs": {
        sid: {
            "summary":
                str(
                    PERF[sid][
                        "summary_path"
                    ]
                ),

            "summary_sha256":
                PERF[sid][
                    "summary_sha256"
                ],
        }
        for sid in SYSTEMS
    },

    "paired_bootstrap": {
        "method":
            (
                "paired prompt bootstrap "
                "over S01-S09; percentile "
                "95% CI of mean A-B"
            ),

        "resamples":
            BOOT_N,

        "base_seed":
            BOOT_SEED,
    },

    "reporting_notes": [
        (
            "UTMOSv2 and DistillMOS are "
            "automatic MOS predictions, "
            "not human MOS."
        ),
        (
            "Headline comparison uses "
            "the frozen intended "
            "stress/pronunciation target."
        ),
        (
            "WER/CER references use "
            "text_plain and historical "
            "ru_simple_v1."
        ),
        (
            "S04 contains verbalized "
            "numbers while Whisper often "
            "emits digits; primary metrics "
            "retain this mismatch."
        ),
        (
            "Performance is CPU two-thread "
            "measurement; quality evaluators "
            "may use GPU and are not used "
            "for TTS performance ranking."
        ),
        (
            "Piper sentence-level yields "
            "are not reported as equivalent "
            "to native sub-sentence TTFA."
        ),
    ],
}


SUMMARY_OUT = (
    out /
    "benchmark_summary.json"
)

SUMMARY_OUT.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


# ============================================================
# README
# ============================================================

rows_for_readme = sorted(
    table_rows,
    key=lambda x:
        x["micro_WER"],
)


def fmt(value, digits=3):
    if value is None:
        return "—"

    return (
        f"{value:.{digits}f}"
    )


lines = [
    (
        "# Cross-model Russian TTS "
        "benchmark v1 — final"
    ),
    "",
    "Status: **FROZEN_FINAL_RESULTS**",
    "",
    "## Headline comparison",
    "",
    (
        "| System | WER ↓ | CER ↓ | "
        "UTMOSv2 ↑ | Distill-MOS ↑ | "
        "Median RTF CPU×2 ↓ | "
        "Full-wave latency, s ↓ | "
        "Native stream TTFA, s ↓ |"
    ),
    (
        "|---|---:|---:|---:|---:|"
        "---:|---:|---:|"
    ),
]


for row in rows_for_readme:
    lines.append(
        "| "
        + row["display_name"]
        + f" | {100 * row['micro_WER']:.2f}%"
        + f" | {100 * row['micro_CER']:.2f}%"
        + f" | {row['UTMOSv2_mean']:.3f}"
        + f" | {row['DistillMOS_mean']:.3f}"
        + f" | {row['RTF_median_CPU2']:.3f}"
        + " | "
        + fmt(
            row[
                "full_waveform_latency_median_seconds_CPU2"
            ]
        )
        + " | "
        + fmt(
            row[
                "native_streaming_TTFA_median_seconds_CPU2"
            ]
        )
        + " |"
    )


lines += [
    "",
    (
        "UTMOSv2 and Distill-MOS are "
        "automatic MOS predictions, "
        "not formal human MOS."
    ),
    "",
    "## Evaluation contract",
    "",
    (
        "The headline comparison uses the "
        "same frozen semantic utterances "
        "and intended pronunciation/stress "
        "target. Project MeloTTS receives "
        "explicit U+0301 stress marks; "
        "external systems use their native "
        "frontends. WER/CER always use "
        "`text_plain` as reference."
    ),
    "",
    (
        "WER/CER use pinned Whisper-small "
        "with historical `ru_simple_v1`. "
        "UTMOSv2 5 folds × 5 repetitions "
        "is the primary AutoMOS metric; "
        "Distill-MOS v7 is the secondary "
        "cross-check."
    ),
    "",
    "## S04 sensitivity",
    "",
    (
        "S04 contains verbalized numbers. "
        "Whisper frequently outputs the "
        "same content as digits. Primary "
        "WER/CER retain this frozen "
        "orthographic mismatch. "
        "`s04_numeric_orthography_"
        "sensitivity.csv` is diagnostic "
        "only and must not replace the "
        "headline metrics."
    ),
    "",
    "## Statistics",
    "",
    (
        "`paired_bootstrap.csv` contains "
        "all pairwise paired prompt-level "
        "95% bootstrap CIs across S01-S09 "
        "for WER, CER, UTMOSv2 and "
        "Distill-MOS."
    ),
    "",
]


README_OUT = (
    out /
    "README.md"
)

README_OUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)


# ============================================================
# Manifest
# ============================================================

input_paths = [
    contract_path,
    systems_path,
    prompts_path,
    W_PP_PATH,
    W_SUM_PATH,
    U_PP_PATH,
    U_SUM_PATH,
    D_PP_PATH,
    D_SUM_PATH,
]

input_paths += [
    PERF[sid][
        "summary_path"
    ]
    for sid in SYSTEMS
]

if XTTS_STREAM:
    input_paths.append(
        XTTS_STREAM[
            "summary_path"
        ]
    )

input_paths = sorted(
    set(input_paths),
    key=lambda x: str(x),
)


output_paths = [
    TABLE_PATH,
    JOINED_PATH,
    BOOT_PATH,
    S04_PATH,
    CONTRACT_OUT,
    SYSTEMS_OUT,
    SUMMARY_OUT,
    README_OUT,
]


manifest = {
    "inputs": {
        str(p): {
            "sha256":
                sha256(p),

            "size_bytes":
                p.stat().st_size,
        }
        for p in input_paths
    },

    "outputs": {
        p.name: {
            "sha256":
                sha256(p),

            "size_bytes":
                p.stat().st_size,
        }
        for p in output_paths
    },
}


MANIFEST_OUT = (
    out /
    "manifest.json"
)

MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


# ============================================================
# Final console summary
# ============================================================

print("=" * 78)
print(
    "FINAL CROSS-MODEL "
    "BENCHMARK ARTIFACT"
)
print("=" * 78)

print(
    "systems =",
    len(SYSTEMS),
)

print(
    "prompts_per_system =",
    len(PROMPT_IDS),
)

print(
    "canonical_rows =",
    len(joined),
)

print(
    "paired_bootstrap_rows =",
    len(bootstrap_rows),
)

print(
    "output =",
    out,
)

print()

for row in rows_for_readme:
    print(
        f"{row['display_name']:28s} "
        f"WER={100 * row['micro_WER']:6.2f}% "
        f"CER={100 * row['micro_CER']:6.2f}% "
        f"UTMOS={row['UTMOSv2_mean']:.3f} "
        f"Distill={row['DistillMOS_mean']:.3f} "
        f"RTF={row['RTF_median_CPU2']:.3f}"
    )

print()

print(
    "manifest_sha256 =",
    sha256(
        MANIFEST_OUT
    ),
)

print(
    "FINAL BENCHMARK ARTIFACT: PASS"
)
