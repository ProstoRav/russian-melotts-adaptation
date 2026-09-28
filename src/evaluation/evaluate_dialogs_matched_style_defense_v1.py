#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import soundfile as sf
import torch


PROJECT = Path(
    "${PROJECT_ROOT}"
)

INPUT = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_eval_v2"
)

DEFENSE_MANIFEST = (
    INPUT
    / "defense_manifest.jsonl"
)

CHECKPOINT_CONTRACT = (
    PROJECT
    / "artifacts/frozen/"
      "tts_trajectory_expanded_checkpoint_contract_v2/"
      "contract.json"
)

PROMPT_CONTRACT = (
    PROJECT
    / "artifacts/frozen/"
      "defense_qc_prompts_v2/"
      "prompts.json"
)

OUT = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_matched_style_defense_v1"
)

PLOTS = (
    OUT
    / "plots"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_PROMPT_SHA = (
    "95f01b936bd30d120fced130cf2d2b10"
    "d5a1c8777cf36d4295d17e2a7a9863aa"
)

EXPECTED_CHECKPOINT_CONTRACT_SHA = (
    "c46b5433f5ea566961d3a25aff3f3b3b"
    "7127efed8badbc1df82b112d5a647d91"
)


EPOCHS = [
    1,
    10,
    20,
    30,
    38,
]

CHOSEN = 38

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 29838


C = {
    "main_fill": "#DCEAF7",
    "main_edge": "#35698F",

    "chosen_fill": "#FFE3A1",
    "chosen_edge": "#9A6200",

    "median": "#173B57",
    "trend": "#008C95",

    "zero": "#68737E",
    "whisker": "#98A3AF",
    "grid": "#D9DEE5",
    "text": "#5F6872",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


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


def load_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def append_jsonl(
    path: Path,
    row,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


def acoustic_metrics(
    path: Path,
):
    info = sf.info(
        str(path)
    )

    duration = (
        info.frames
        / info.samplerate
    )

    y, sr = librosa.load(
        str(path),
        sr=16000,
        mono=True,
    )

    if not len(y):
        raise RuntimeError(
            f"empty WAV: {path}"
        )

    f0, _, _ = librosa.pyin(
        y,
        fmin=60.0,
        fmax=600.0,
        sr=sr,
        frame_length=1024,
        hop_length=160,
    )

    finite = f0[
        np.isfinite(
            f0
        )
    ]

    if not len(finite):
        raise RuntimeError(
            f"no voiced F0: {path}"
        )

    f0_p10 = float(
        np.percentile(
            finite,
            10,
        )
    )

    f0_p90 = float(
        np.percentile(
            finite,
            90,
        )
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=160,
    )[0]

    mean_rms = float(
        np.mean(
            rms
        )
    )

    return {
        "duration_sec": float(
            duration
        ),

        "f0_median_hz": float(
            np.median(
                finite
            )
        ),

        "f0_mean_hz": float(
            np.mean(
                finite
            )
        ),

        "f0_p10_hz": (
            f0_p10
        ),

        "f0_p90_hz": (
            f0_p90
        ),

        "f0_range_hz": (
            f0_p90
            - f0_p10
        ),

        "voiced_fraction": float(
            len(finite)
            / max(
                len(f0),
                1,
            )
        ),

        "rms_db": float(
            20.0
            * math.log10(
                max(
                    mean_rms,
                    1e-12,
                )
            )
        ),
    }


def bootstrap_median_ci(
    values,
    seed,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    samples = rng.choice(
        values,
        size=(
            BOOTSTRAP_SAMPLES,
            n,
        ),
        replace=True,
    )

    medians = np.median(
        samples,
        axis=1,
    )

    return (
        float(
            np.percentile(
                medians,
                2.5,
            )
        ),
        float(
            np.percentile(
                medians,
                97.5,
            )
        ),
    )


def direction_consistency(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    pos = float(
        np.mean(
            values > 0
        )
    )

    neg = float(
        np.mean(
            values < 0
        )
    )

    return max(
        pos,
        neg,
    )


def export(
    fig,
    stem,
):
    fig.savefig(
        PLOTS
        / f"{stem}.svg",
        bbox_inches="tight",
    )

    fig.savefig(
        PLOTS
        / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        PLOTS
        / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


#
# Validate frozen contracts.
#
if (
    sha256(
        PROMPT_CONTRACT
    )
    != EXPECTED_PROMPT_SHA
):
    raise RuntimeError(
        "prompt contract SHA mismatch"
    )

if (
    sha256(
        CHECKPOINT_CONTRACT
    )
    != EXPECTED_CHECKPOINT_CONTRACT_SHA
):
    raise RuntimeError(
        "checkpoint contract SHA mismatch"
    )


manifest = load_jsonl(
    DEFENSE_MANIFEST
)

if len(manifest) != 90:
    raise RuntimeError(
        f"defense manifest rows="
        f"{len(manifest)}, expected 90"
    )


#
# Strict matched-pair validation.
#
for epoch in EPOCHS:

    x = [
        r
        for r in manifest
        if int(
            r["epoch"]
        ) == epoch
    ]

    if len(x) != 18:
        raise RuntimeError(
            f"E{epoch}: "
            f"{len(x)} rows != 18"
        )

    for prompt_num in range(
        1,
        10,
    ):

        prompt_id = (
            f"S{prompt_num:02d}"
        )

        p = [
            r
            for r in x
            if r[
                "prompt_id"
            ] == prompt_id
        ]

        if len(p) != 2:
            raise RuntimeError(
                f"E{epoch} "
                f"{prompt_id}: "
                f"{len(p)} rows"
            )

        p = sorted(
            p,
            key=lambda r:
                int(
                    r[
                        "speaker_id"
                    ]
                ),
        )

        if [
            int(
                r[
                    "speaker_id"
                ]
            )
            for r in p
        ] != [
            0,
            1,
        ]:
            raise RuntimeError(
                f"E{epoch} "
                f"{prompt_id}: "
                "expected SID0/SID1"
            )

        if (
            p[0][
                "text"
            ]
            != p[1][
                "text"
            ]
        ):
            raise RuntimeError(
                f"E{epoch} "
                f"{prompt_id}: "
                "text mismatch"
            )

        if (
            int(
                p[0][
                    "seed"
                ]
            )
            != int(
                p[1][
                    "seed"
                ]
            )
        ):
            raise RuntimeError(
                f"E{epoch} "
                f"{prompt_id}: "
                "seed mismatch"
            )

        if (
            p[0][
                "generator_sha256"
            ]
            != p[1][
                "generator_sha256"
            ]
        ):
            raise RuntimeError(
                f"E{epoch} "
                f"{prompt_id}: "
                "checkpoint mismatch"
            )


print("=" * 78)
print(
    "DIALOGS MATCHED-STYLE DEFENSE V1"
)
print("=" * 78)

print(
    "manifest_rows:",
    len(
        manifest
    ),
)

print(
    "epochs:",
    EPOCHS,
)

print(
    "prompts_per_epoch: 9"
)

print(
    "matched_audio_per_epoch: 18"
)

print(
    "pair contract: "
    "same text + same seed + same checkpoint"
)

print(
    "PAIR CONTRACT: PASS"
)


#
# Resumable acoustic cache.
#
CACHE = (
    OUT
    / "acoustic_cache.jsonl"
)

cache = {}

if CACHE.is_file():
    for row in load_jsonl(
        CACHE
    ):
        cache[
            row[
                "wav_sha256"
            ]
        ] = row


audio_rows = []

for idx, row in enumerate(
    manifest,
    1,
):
    wav = Path(
        row[
            "wav"
        ]
    )

    if not wav.is_file():
        raise RuntimeError(
            f"missing WAV: {wav}"
        )

    digest = sha256(
        wav
    )

    expected_digest = row[
        "wav_sha256"
    ]

    if (
        digest
        != expected_digest
    ):
        raise RuntimeError(
            f"WAV SHA mismatch: "
            f"{wav}"
        )

    if digest in cache:
        metrics = cache[
            digest
        ]

        status = "CACHE"

    else:
        metrics = {
            "wav_sha256": digest,
            "wav": str(
                wav
            ),
            **acoustic_metrics(
                wav
            ),
        }

        append_jsonl(
            CACHE,
            metrics,
        )

        cache[
            digest
        ] = metrics

        status = (
            "ACOUSTIC"
        )

    audio_rows.append(
        {
            **row,
            **{
                k: v
                for k, v
                in metrics.items()
                if k not in (
                    "wav",
                    "wav_sha256",
                )
            },
        }
    )

    print(
        f"[{idx:02d}/90] "
        f"{status:8s} "
        f"E{int(row['epoch']):02d} "
        f"{row['prompt_id']} "
        f"SID{int(row['speaker_id'])}",
        flush=True,
    )


audio_df = pd.DataFrame(
    audio_rows
)

audio_df.to_csv(
    OUT
    / "per_audio_acoustics.csv",
    index=False,
)


#
# Build 45 exact SID0/SID1 pairs.
#
pair_rows = []


for epoch in EPOCHS:

    for prompt_num in range(
        1,
        10,
    ):

        prompt_id = (
            f"S{prompt_num:02d}"
        )

        x = audio_df[
            (
                audio_df[
                    "epoch"
                ]
                == epoch
            )
            &
            (
                audio_df[
                    "prompt_id"
                ]
                == prompt_id
            )
        ].sort_values(
            "speaker_id"
        )

        if len(x) != 2:
            raise RuntimeError(
                f"pair missing: "
                f"E{epoch} "
                f"{prompt_id}"
            )

        neutral = x.iloc[0]
        positive = x.iloc[1]


        def pct_delta(
            a,
            b,
        ):
            return (
                (
                    b
                    / a
                )
                - 1.0
            ) * 100.0


        pair_rows.append(
            {
                "epoch": epoch,
                "step": int(
                    neutral[
                        "step"
                    ]
                ),

                "checkpoint_role": (
                    neutral[
                        "checkpoint_role"
                    ]
                ),

                "prompt_id": (
                    prompt_id
                ),

                "category": (
                    neutral[
                        "category"
                    ]
                ),

                "text": (
                    neutral[
                        "text"
                    ]
                ),

                "seed": int(
                    neutral[
                        "seed"
                    ]
                ),

                #
                # Positive - neutral
                #
                "delta_f0_median_hz": (
                    positive[
                        "f0_median_hz"
                    ]
                    -
                    neutral[
                        "f0_median_hz"
                    ]
                ),

                "delta_f0_median_pct": (
                    pct_delta(
                        neutral[
                            "f0_median_hz"
                        ],
                        positive[
                            "f0_median_hz"
                        ],
                    )
                ),

                "delta_f0_range_hz": (
                    positive[
                        "f0_range_hz"
                    ]
                    -
                    neutral[
                        "f0_range_hz"
                    ]
                ),

                "delta_f0_range_pct": (
                    pct_delta(
                        neutral[
                            "f0_range_hz"
                        ],
                        positive[
                            "f0_range_hz"
                        ],
                    )
                ),

                "delta_duration_sec": (
                    positive[
                        "duration_sec"
                    ]
                    -
                    neutral[
                        "duration_sec"
                    ]
                ),

                "delta_duration_pct": (
                    pct_delta(
                        neutral[
                            "duration_sec"
                        ],
                        positive[
                            "duration_sec"
                        ],
                    )
                ),

                "delta_rms_db": (
                    positive[
                        "rms_db"
                    ]
                    -
                    neutral[
                        "rms_db"
                    ]
                ),

                "delta_voiced_fraction_pp": (
                    (
                        positive[
                            "voiced_fraction"
                        ]
                        -
                        neutral[
                            "voiced_fraction"
                        ]
                    )
                    * 100.0
                ),
            }
        )


pairs = pd.DataFrame(
    pair_rows
)

if len(pairs) != 45:
    raise RuntimeError(
        f"pair rows="
        f"{len(pairs)}, expected 45"
    )


pairs.to_csv(
    OUT
    / "paired_style_metrics.csv",
    index=False,
)


#
# Summary and deterministic bootstrap CIs.
#
METRICS = {
    "delta_f0_median_hz": (
        "ΔF0 Median",
        "Hz",
    ),

    "delta_f0_range_pct": (
        "ΔF0 Range",
        "%",
    ),

    "delta_duration_pct": (
        "ΔDuration",
        "%",
    ),

    "delta_rms_db": (
        "ΔRMS",
        "dB",
    ),

    "delta_voiced_fraction_pp": (
        "ΔVoiced Fraction",
        "pp",
    ),
}


summary_rows = []


for epoch_idx, epoch in enumerate(
    EPOCHS
):

    x = pairs[
        pairs[
            "epoch"
        ]
        == epoch
    ]

    for metric_idx, (
        metric,
        (
            label,
            unit,
        ),
    ) in enumerate(
        METRICS.items()
    ):

        values = x[
            metric
        ].to_numpy(
            dtype=float
        )

        seed = (
            BOOTSTRAP_SEED
            + epoch_idx * 100
            + metric_idx
        )

        ci_low, ci_high = (
            bootstrap_median_ci(
                values,
                seed,
            )
        )

        summary_rows.append(
            {
                "epoch": epoch,

                "checkpoint_role": (
                    x.iloc[0][
                        "checkpoint_role"
                    ]
                ),

                "metric": metric,
                "label": label,
                "unit": unit,

                "n_pairs": (
                    len(values)
                ),

                "median": float(
                    np.median(
                        values
                    )
                ),

                "mean": float(
                    np.mean(
                        values
                    )
                ),

                "q1": float(
                    np.percentile(
                        values,
                        25,
                    )
                ),

                "q3": float(
                    np.percentile(
                        values,
                        75,
                    )
                ),

                "p10": float(
                    np.percentile(
                        values,
                        10,
                    )
                ),

                "p90": float(
                    np.percentile(
                        values,
                        90,
                    )
                ),

                "median_abs_separation": float(
                    np.median(
                        np.abs(
                            values
                        )
                    )
                ),

                "direction_consistency": (
                    direction_consistency(
                        values
                    )
                ),

                "bootstrap_median_ci95_low": (
                    ci_low
                ),

                "bootstrap_median_ci95_high": (
                    ci_high
                ),
            }
        )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT
    / "summary_by_epoch_metric.csv",
    index=False,
)


#
# Wide presentation table.
#
wide_rows = []


for epoch in EPOCHS:

    r = {
        "epoch": epoch
    }

    for metric in METRICS:

        s = summary[
            (
                summary[
                    "epoch"
                ]
                == epoch
            )
            &
            (
                summary[
                    "metric"
                ]
                == metric
            )
        ].iloc[0]

        r[
            metric
            + "_median"
        ] = s[
            "median"
        ]

        r[
            metric
            + "_abs_median"
        ] = s[
            "median_abs_separation"
        ]

        r[
            metric
            + "_direction_consistency"
        ] = s[
            "direction_consistency"
        ]

        r[
            metric
            + "_ci95_low"
        ] = s[
            "bootstrap_median_ci95_low"
        ]

        r[
            metric
            + "_ci95_high"
        ] = s[
            "bootstrap_median_ci95_high"
        ]

    wide_rows.append(
        r
    )


wide = pd.DataFrame(
    wide_rows
)

wide.to_csv(
    OUT
    / "trajectory_summary.csv",
    index=False,
)


#
# Supporting parameter-space diagnostic:
# distance between SID0/SID1 embeddings.
#
checkpoint_contract = json.loads(
    CHECKPOINT_CONTRACT.read_text(
        encoding="utf-8"
    )
)

dialogs_cp = {
    int(
        cp[
            "epoch"
        ]
    ): cp
    for cp in
    checkpoint_contract[
        "experiments"
    ][
        "dialogs"
    ][
        "checkpoints"
    ]
}


embedding_rows = []


for epoch in EPOCHS:

    cp = dialogs_cp[
        epoch
    ]

    checkpoint = torch.load(
        cp[
            "G"
        ],
        map_location="cpu",
        weights_only=False,
    )

    emb = (
        checkpoint[
            "model"
        ][
            "emb_g.weight"
        ]
        .detach()
        .float()
    )

    if tuple(
        emb.shape
    ) != (
        2,
        256,
    ):
        raise RuntimeError(
            f"E{epoch}: "
            f"unexpected emb shape "
            f"{tuple(emb.shape)}"
        )

    a = emb[0]
    b = emb[1]

    l2 = float(
        torch.linalg.vector_norm(
            a - b
        )
    )

    cosine = float(
        torch.nn.functional.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
    )

    embedding_rows.append(
        {
            "epoch": epoch,
            "step": int(
                cp[
                    "step"
                ]
            ),
            "l2_distance": l2,
            "cosine_similarity": (
                cosine
            ),
        }
    )


pd.DataFrame(
    embedding_rows
).to_csv(
    OUT
    / "embedding_separation.csv",
    index=False,
)


#
# Publication-style boxplots.
#
def make_plot(
    metric,
    english_name,
    russian_name,
    ylabel,
    stem,
):
    groups = [
        pairs[
            pairs[
                "epoch"
            ]
            == epoch
        ][
            metric
        ].to_numpy(
            dtype=float
        )
        for epoch in EPOCHS
    ]

    positions = list(
        range(
            1,
            len(
                EPOCHS
            )
            + 1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.7,
            5.7,
        )
    )

    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.58,
        whis=(
            10,
            90,
        ),
        showfliers=False,
        patch_artist=True,

        medianprops={
            "color": C[
                "median"
            ],
            "linewidth": 2.2,
        },

        whiskerprops={
            "color": C[
                "whisker"
            ],
            "linewidth": 0.9,
        },

        capprops={
            "color": C[
                "whisker"
            ],
            "linewidth": 0.9,
        },
    )


    for box, epoch in zip(
        bp[
            "boxes"
        ],
        EPOCHS,
    ):

        if epoch == CHOSEN:

            box.set_facecolor(
                C[
                    "chosen_fill"
                ]
            )

            box.set_edgecolor(
                C[
                    "chosen_edge"
                ]
            )

            box.set_linewidth(
                2.0
            )

        else:

            box.set_facecolor(
                C[
                    "main_fill"
                ]
            )

            box.set_edgecolor(
                C[
                    "main_edge"
                ]
            )

            box.set_linewidth(
                1.2
            )


    medians = [
        float(
            np.median(
                x
            )
        )
        for x in groups
    ]


    ax.plot(
        positions,
        medians,
        color=C[
            "trend"
        ],
        marker="o",
        linewidth=2.3,
        markersize=5.2,
        zorder=6,
    )


    ax.axhline(
        0,
        color=C[
            "zero"
        ],
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
    )


    labels = [
        (
            f"E{epoch}\n"
            "chosen candidate"
            if epoch == CHOSEN
            else f"E{epoch}"
        )
        for epoch in EPOCHS
    ]


    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )


    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        (
            f"Dialogs — "
            f"{english_name} "
            f"({russian_name}) "
            "по эпохам"
        ),
        loc="left",
        fontweight="semibold",
        pad=12,
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_color(
        "#69747F"
    )

    ax.spines[
        "bottom"
    ].set_color(
        "#69747F"
    )

    ax.grid(
        axis="y",
        linestyle="-",
        linewidth=0.6,
        color=C[
            "grid"
        ],
        alpha=0.85,
    )

    ax.set_axisbelow(
        True
    )


    handle = Line2D(
        [0],
        [0],
        color=C[
            "trend"
        ],
        marker="o",
        linewidth=2.3,
        label=(
            "Median trend "
            "(динамика медианы)"
        ),
    )


    ax.legend(
        handles=[
            handle
        ],
        frameon=False,
        loc="upper right",
    )


    fig.text(
        0.105,
        0.037,
        (
            "Difference = positive − neutral; "
            "box = Q1–Q3; center line = Median "
            "(медиана); whiskers = P10–P90."
        ),
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=C[
            "text"
        ],
    )

    fig.text(
        0.105,
        0.018,
        (
            "Каждая точка распределения — одна из 9 "
            "одинаковых фраз, синтезированная с одинаковым seed."
        ),
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=C[
            "text"
        ],
    )


    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.90,
        bottom=0.205,
    )

    export(
        fig,
        stem,
    )


make_plot(
    metric=(
        "delta_f0_median_hz"
    ),
    english_name="ΔF0 Median",
    russian_name=(
        "разница медианы F0"
    ),
    ylabel=(
        "ΔF0 Median "
        "(разница медианы F0), Hz"
    ),
    stem=(
        "f0_median_delta_boxplot"
    ),
)


make_plot(
    metric=(
        "delta_f0_range_pct"
    ),
    english_name="ΔF0 Range",
    russian_name=(
        "разница диапазона F0"
    ),
    ylabel=(
        "ΔF0 Range "
        "(разница диапазона F0), %"
    ),
    stem=(
        "f0_range_delta_boxplot"
    ),
)


make_plot(
    metric=(
        "delta_duration_pct"
    ),
    english_name="ΔDuration",
    russian_name=(
        "разница длительности"
    ),
    ylabel=(
        "ΔDuration "
        "(разница длительности), %"
    ),
    stem=(
        "duration_delta_boxplot"
    ),
)


make_plot(
    metric=(
        "delta_rms_db"
    ),
    english_name="ΔRMS",
    russian_name=(
        "разница RMS"
    ),
    ylabel=(
        "ΔRMS "
        "(разница RMS), dB"
    ),
    stem=(
        "rms_delta_boxplot"
    ),
)


make_plot(
    metric=(
        "delta_voiced_fraction_pp"
    ),
    english_name=(
        "ΔVoiced Fraction"
    ),
    russian_name=(
        "разница доли voiced frames"
    ),
    ylabel=(
        "ΔVoiced Fraction "
        "(разница доли voiced frames), п.п."
    ),
    stem=(
        "voiced_fraction_delta_boxplot"
    ),
)


#
# Human-readable summary.
#
print()
print("=" * 78)
print(
    "MATCHED-STYLE TRAJECTORY"
)
print("=" * 78)


for epoch in EPOCHS:

    x = summary[
        summary[
            "epoch"
        ]
        == epoch
    ]

    print()
    print(
        f"E{epoch:02d}"
        + (
            "  CHOSEN_CANDIDATE"
            if epoch == CHOSEN
            else ""
        )
    )

    for metric in METRICS:

        r = x[
            x[
                "metric"
            ]
            == metric
        ].iloc[0]

        print(
            f"  {r['label']:18s} "
            f"median={r['median']:9.3f} "
            f"CI95=["
            f"{r['bootstrap_median_ci95_low']:9.3f}, "
            f"{r['bootstrap_median_ci95_high']:9.3f}] "
            f"|Δ|med="
            f"{r['median_abs_separation']:9.3f} "
            f"consistency="
            f"{r['direction_consistency']:.3f}"
        )


print()
print("=" * 78)
print(
    "EMBEDDING SEPARATION"
)
print("=" * 78)

print(
    pd.DataFrame(
        embedding_rows
    ).to_string(
        index=False
    )
)


metadata = {
    "version": 1,

    "input": str(
        INPUT
    ),

    "defense_manifest": (
        str(
            DEFENSE_MANIFEST
        )
    ),

    "prompt_contract_sha256": (
        EXPECTED_PROMPT_SHA
    ),

    "checkpoint_contract_sha256": (
        EXPECTED_CHECKPOINT_CONTRACT_SHA
    ),

    "epochs": (
        EPOCHS
    ),

    "chosen_candidate": (
        CHOSEN
    ),

    "n_prompts_per_epoch": 9,

    "n_audio_per_epoch": 18,

    "pairing": {
        "same_text": True,
        "same_seed": True,
        "same_checkpoint": True,
        "difference": (
            "positive - neutral"
        ),
    },

    "acoustics": {
        "analysis_sr": 16000,
        "f0_method": (
            "librosa.pyin"
        ),
        "f0_min_hz": 60,
        "f0_max_hz": 600,
        "f0_range": (
            "P90 - P10"
        ),
    },

    "bootstrap": {
        "samples": (
            BOOTSTRAP_SAMPLES
        ),
        "seed": (
            BOOTSTRAP_SEED
        ),
        "statistic": (
            "median"
        ),
        "ci": (
            "percentile 95%"
        ),
    },

    "interpretation": (
        "Measures acoustic style separation "
        "between neutral and positive under "
        "matched text/seed conditions; does not "
        "by itself establish perceptual emotion "
        "accuracy."
    ),
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


(
    OUT
    / ".complete"
).write_text(
    "PASS\n",
    encoding="utf-8",
)


print()
print("=" * 78)
print(
    "DIALOGS MATCHED-STYLE DEFENSE V1: PASS"
)
print("=" * 78)

print(
    "paired_rows:",
    len(
        pairs
    ),
)

print(
    "plots:",
    PLOTS,
)

print(
    "output:",
    OUT,
)
