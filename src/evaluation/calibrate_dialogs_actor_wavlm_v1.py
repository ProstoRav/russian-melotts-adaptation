#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)

from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMForXVector,
)


os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)


PROJECT = Path(
    "${PROJECT_ROOT}"
)

AUDIT = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_actual_speaker_audit_v1/"
      "selected_rows.csv"
)

BACKEND = (
    PROJECT
    / "artifacts/frozen/"
      "speaker_embedding_backend_"
      "wavlm_base_plus_sv_v1/"
      "contract.json"
)

EXPECTED_BACKEND_SHA = (
    "bfba1bdd37d25e9de22a9dce9d07a86"
    "fb433adc8a214f2f0b504aa5bba96b98e"
)

OUT = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_actor_wavlm_calibration_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


CACHE = (
    OUT
    / "embedding_cache.jsonl"
)


ACTORS = [
    "M",
    "S",
]

NEUTRAL_LABEL = (
    "dialogs_female_neutral"
)

POSITIVE_LABEL = (
    "dialogs_female_positive"
)

ENROLL_PER_ACTOR_LABEL = 100

MIN_ENROLL_SEC = 1.5
MAX_ENROLL_SEC = 15.0

SELECTION_SEED = 2982026

ANALYSIS_SR = 16000
TRIM_TOP_DB = 40
WINDOW_SEC = 8.0
HOP_SEC = 4.0

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 29840


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


def file_rank(
    basename: str,
):
    payload = (
        f"{SELECTION_SEED}|"
        f"{basename}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_jsonl(
    path: Path,
):
    if not path.is_file():
        return []

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

        f.flush()
        os.fsync(
            f.fileno()
        )


def normalize_vector(
    x,
):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    norm = float(
        np.linalg.norm(
            x
        )
    )

    if norm <= 0:
        raise RuntimeError(
            "zero embedding"
        )

    return (
        x
        / norm
    )


def bootstrap_ci(
    values,
    statistic,
    seed,
):
    values = np.asarray(
        values
    )

    rng = (
        np.random
        .default_rng(
            seed
        )
    )

    n = len(values)

    indices = rng.integers(
        0,
        n,
        size=(
            BOOTSTRAP_SAMPLES,
            n,
        ),
    )

    samples = values[
        indices
    ]

    if statistic == "mean":
        stats = np.mean(
            samples,
            axis=1,
        )

    elif statistic == "median":
        stats = np.median(
            samples,
            axis=1,
        )

    else:
        raise ValueError(
            statistic
        )

    return (
        float(
            np.percentile(
                stats,
                2.5,
            )
        ),
        float(
            np.percentile(
                stats,
                97.5,
            )
        ),
    )


#
# Contract validation.
#
if sha256(
    BACKEND
) != EXPECTED_BACKEND_SHA:
    raise RuntimeError(
        "speaker backend contract SHA mismatch"
    )


backend = json.loads(
    BACKEND.read_text(
        encoding="utf-8"
    )
)


snapshot = Path(
    backend[
        "backend"
    ][
        "snapshot"
    ]
)


if not snapshot.is_dir():
    raise RuntimeError(
        f"snapshot missing: "
        f"{snapshot}"
    )


for required in (
    "config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
):
    if not (
        snapshot
        / required
    ).is_file():
        raise RuntimeError(
            f"missing model file: "
            f"{required}"
        )


#
# Source audit.
#
audit = pd.read_csv(
    AUDIT
)


required_columns = {
    "split",
    "wav",
    "basename",
    "artificial_label",
    "source_speaker_id",
    "source_emotion",
    "source_duration",
}


missing = (
    required_columns
    - set(
        audit.columns
    )
)


if missing:
    raise RuntimeError(
        f"audit missing columns: "
        f"{sorted(missing)}"
    )


#
# Deterministic enrollment selection.
#
train = audit[
    audit[
        "split"
    ]
    == "train"
].copy()


eligible = train[
    (
        train[
            "source_duration"
        ]
        >= MIN_ENROLL_SEC
    )
    &
    (
        train[
            "source_duration"
        ]
        <= MAX_ENROLL_SEC
    )
].copy()


enrollment_parts = []


for actor in ACTORS:

    for label in (
        NEUTRAL_LABEL,
        POSITIVE_LABEL,
    ):

        bucket = eligible[
            (
                eligible[
                    "source_speaker_id"
                ]
                == actor
            )
            &
            (
                eligible[
                    "artificial_label"
                ]
                == label
            )
        ].copy()


        print(
            "eligible",
            actor,
            label,
            len(bucket),
        )


        if (
            len(bucket)
            <
            ENROLL_PER_ACTOR_LABEL
        ):
            raise RuntimeError(
                f"not enough enrollment: "
                f"{actor} "
                f"{label} "
                f"{len(bucket)}"
            )


        bucket[
            "_rank"
        ] = bucket[
            "basename"
        ].map(
            file_rank
        )


        chosen = (
            bucket
            .sort_values(
                "_rank"
            )
            .head(
                ENROLL_PER_ACTOR_LABEL
            )
            .drop(
                columns=[
                    "_rank"
                ]
            )
        )


        enrollment_parts.append(
            chosen
        )


enrollment = pd.concat(
    enrollment_parts,
    ignore_index=True,
)


if len(enrollment) != 400:
    raise RuntimeError(
        f"enrollment rows="
        f"{len(enrollment)}, "
        "expected 400"
    )


if enrollment[
    "basename"
].duplicated().any():
    raise RuntimeError(
        "duplicate enrollment basename"
    )


heldout = audit[
    audit[
        "split"
    ].isin(
        [
            "val",
            "test",
        ]
    )
].copy()


if len(heldout) != 98:
    raise RuntimeError(
        f"heldout rows="
        f"{len(heldout)}, "
        "expected 98"
    )


if set(
    enrollment[
        "basename"
    ]
) & set(
    heldout[
        "basename"
    ]
):
    raise RuntimeError(
        "enrollment / heldout overlap"
    )


enrollment.to_csv(
    OUT
    / "enrollment_selection.csv",
    index=False,
)


heldout.to_csv(
    OUT
    / "heldout_selection.csv",
    index=False,
)


print()
print("=" * 78)
print(
    "ENROLLMENT SELECTION"
)
print("=" * 78)

print(
    pd.crosstab(
        enrollment[
            "source_speaker_id"
        ],
        enrollment[
            "artificial_label"
        ],
        margins=True,
    ).to_string()
)

print()
print(
    "heldout:",
    len(
        heldout
    ),
)

print(
    pd.crosstab(
        heldout[
            "split"
        ],
        heldout[
            "source_speaker_id"
        ],
        margins=True,
    ).to_string()
)


#
# Load backend strictly locally.
#
feature_extractor = (
    Wav2Vec2FeatureExtractor
    .from_pretrained(
        str(
            snapshot
        ),
        local_files_only=True,
    )
)


model = (
    WavLMForXVector
    .from_pretrained(
        str(
            snapshot
        ),
        local_files_only=True,
    )
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = model.to(
    device=device,
    dtype=torch.float32,
)

model.eval()


print()
print("=" * 78)
print(
    "WAVLM RUNTIME"
)
print("=" * 78)

print(
    "device:",
    device,
)

if device == "cuda":
    print(
        "device_name:",
        torch.cuda.get_device_name(
            0
        ),
    )

print(
    "dtype:",
    next(
        model.parameters()
    ).dtype,
)


def load_audio(
    path: Path,
):
    y, sr = librosa.load(
        str(
            path
        ),
        sr=ANALYSIS_SR,
        mono=True,
    )

    if sr != ANALYSIS_SR:
        raise RuntimeError(
            "sampling-rate contract failed"
        )

    if not len(y):
        raise RuntimeError(
            f"empty WAV: {path}"
        )


    trimmed, _ = (
        librosa.effects.trim(
            y,
            top_db=TRIM_TOP_DB,
        )
    )


    if len(trimmed) >= 1600:
        y = trimmed


    return np.asarray(
        y,
        dtype=np.float32,
    )


def make_windows(
    y,
):
    max_len = int(
        WINDOW_SEC
        * ANALYSIS_SR
    )

    hop = int(
        HOP_SEC
        * ANALYSIS_SR
    )


    if len(y) <= max_len:
        return [
            y
        ]


    starts = list(
        range(
            0,
            len(y)
            - max_len
            + 1,
            hop,
        )
    )


    last_start = (
        len(y)
        - max_len
    )


    if (
        not starts
        or starts[-1]
        != last_start
    ):
        starts.append(
            last_start
        )


    return [
        y[
            start:
            start
            + max_len
        ]
        for start in starts
    ]


@torch.inference_mode()
def embed_wav(
    path: Path,
):
    y = load_audio(
        path
    )

    windows = make_windows(
        y
    )

    vectors = []


    for window in windows:

        inputs = (
            feature_extractor(
                window,
                sampling_rate=(
                    ANALYSIS_SR
                ),
                return_tensors="pt",
            )
        )


        kwargs = {
            key:
                value.to(
                    device
                )
            for key, value
            in inputs.items()
        }


        output = model(
            **kwargs
        )


        vector = (
            output
            .embeddings[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )


        vectors.append(
            normalize_vector(
                vector
            )
        )


    result = normalize_vector(
        np.mean(
            np.stack(
                vectors,
                axis=0,
            ),
            axis=0,
        )
    )


    return {
        "embedding": (
            result.tolist()
        ),

        "windows": (
            len(
                windows
            )
        ),

        "trimmed_duration_sec": (
            len(y)
            / ANALYSIS_SR
        ),
    }


#
# Resumable cache.
#
cache = {
    row[
        "wav_sha256"
    ]: row
    for row in load_jsonl(
        CACHE
    )
}


combined = pd.concat(
    [
        enrollment.assign(
            evaluation_role=(
                "enrollment"
            )
        ),

        heldout.assign(
            evaluation_role=(
                "heldout"
            )
        ),
    ],
    ignore_index=True,
)


embedding_rows = []

new_embeddings = 0


for idx, row in combined.iterrows():

    path = Path(
        row[
            "wav"
        ]
    )


    if not path.is_file():
        raise RuntimeError(
            f"missing WAV: "
            f"{path}"
        )


    digest = sha256(
        path
    )


    cached = cache.get(
        digest
    )


    if cached is not None:

        emb = np.asarray(
            cached[
                "embedding"
            ],
            dtype=np.float32,
        )

        emb = normalize_vector(
            emb
        )

        windows = int(
            cached[
                "windows"
            ]
        )

        trimmed_sec = float(
            cached[
                "trimmed_duration_sec"
            ]
        )

        status = "CACHE"

    else:

        result = embed_wav(
            path
        )

        emb = np.asarray(
            result[
                "embedding"
            ],
            dtype=np.float32,
        )

        windows = int(
            result[
                "windows"
            ]
        )

        trimmed_sec = float(
            result[
                "trimmed_duration_sec"
            ]
        )


        cached = {
            "wav_sha256": (
                digest
            ),

            "wav": str(
                path
            ),

            "embedding": (
                emb.tolist()
            ),

            "windows": (
                windows
            ),

            "trimmed_duration_sec": (
                trimmed_sec
            ),
        }


        append_jsonl(
            CACHE,
            cached,
        )

        cache[
            digest
        ] = cached

        new_embeddings += 1

        status = "EMBED"


    embedding_rows.append(
        {
            "basename": (
                row[
                    "basename"
                ]
            ),

            "wav": str(
                path
            ),

            "wav_sha256": (
                digest
            ),

            "evaluation_role": (
                row[
                    "evaluation_role"
                ]
            ),

            "split": (
                row[
                    "split"
                ]
            ),

            "actor": (
                row[
                    "source_speaker_id"
                ]
            ),

            "conditioning_label": (
                row[
                    "artificial_label"
                ]
            ),

            "source_emotion": (
                row[
                    "source_emotion"
                ]
            ),

            "windows": (
                windows
            ),

            "trimmed_duration_sec": (
                trimmed_sec
            ),

            "embedding": (
                emb
            ),
        }
    )


    print(
        f"[{idx + 1:03d}/"
        f"{len(combined):03d}] "
        f"{status:5s} "
        f"{row['evaluation_role']:10s} "
        f"{row['split']:5s} "
        f"{row['source_speaker_id']} "
        f"{row['basename']}",
        flush=True,
    )


#
# Enrollment matrices.
#
def matrix_for(
    actor,
    label=None,
):
    rows = [
        r
        for r in embedding_rows
        if (
            r[
                "evaluation_role"
            ]
            == "enrollment"
            and
            r[
                "actor"
            ]
            == actor
            and
            (
                label is None
                or
                r[
                    "conditioning_label"
                ]
                == label
            )
        )
    ]


    matrix = np.stack(
        [
            r[
                "embedding"
            ]
            for r in rows
        ],
        axis=0,
    )


    return matrix


M = matrix_for(
    "M"
)

S = matrix_for(
    "S"
)

M_neutral = matrix_for(
    "M",
    NEUTRAL_LABEL,
)

M_positive = matrix_for(
    "M",
    POSITIVE_LABEL,
)

S_neutral = matrix_for(
    "S",
    NEUTRAL_LABEL,
)

S_positive = matrix_for(
    "S",
    POSITIVE_LABEL,
)


for name, matrix, expected in (
    (
        "M",
        M,
        200,
    ),
    (
        "S",
        S,
        200,
    ),
    (
        "M neutral",
        M_neutral,
        100,
    ),
    (
        "M positive",
        M_positive,
        100,
    ),
    (
        "S neutral",
        S_neutral,
        100,
    ),
    (
        "S positive",
        S_positive,
        100,
    ),
):
    if (
        matrix.shape
        != (
            expected,
            512,
        )
    ):
        raise RuntimeError(
            f"{name}: "
            f"{matrix.shape}"
        )


def centroid(
    matrix,
):
    return normalize_vector(
        np.mean(
            matrix,
            axis=0,
        )
    )


centroid_M = centroid(
    M
)

centroid_S = centroid(
    S
)

centroid_M_neutral = (
    centroid(
        M_neutral
    )
)

centroid_M_positive = (
    centroid(
        M_positive
    )
)

centroid_S_neutral = (
    centroid(
        S_neutral
    )
)

centroid_S_positive = (
    centroid(
        S_positive
    )
)


np.savez_compressed(
    OUT
    / "actor_centroids.npz",

    M=centroid_M,
    S=centroid_S,

    M_neutral=(
        centroid_M_neutral
    ),

    M_positive=(
        centroid_M_positive
    ),

    S_neutral=(
        centroid_S_neutral
    ),

    S_positive=(
        centroid_S_positive
    ),
)


#
# Pairwise distributions.
#
def upper_triangle(
    matrix,
):
    similarity = (
        matrix
        @ matrix.T
    )

    idx = np.triu_indices(
        len(matrix),
        k=1,
    )

    return similarity[
        idx
    ]


mm = upper_triangle(
    M
)

ss = upper_triangle(
    S
)

ms = (
    M
    @ S.T
).reshape(
    -1
)


m_cross_condition = (
    M_neutral
    @ M_positive.T
).reshape(
    -1
)

s_cross_condition = (
    S_neutral
    @ S_positive.T
).reshape(
    -1
)


def distribution_row(
    name,
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "distribution": name,

        "n": int(
            len(values)
        ),

        "mean": float(
            np.mean(
                values
            )
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "std": float(
            np.std(
                values
            )
        ),

        "p10": float(
            np.percentile(
                values,
                10,
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

        "p90": float(
            np.percentile(
                values,
                90,
            )
        ),
    }


pairwise_summary = pd.DataFrame(
    [
        distribution_row(
            "M-M",
            mm,
        ),

        distribution_row(
            "S-S",
            ss,
        ),

        distribution_row(
            "M-S",
            ms,
        ),

        distribution_row(
            "M neutral-positive",
            m_cross_condition,
        ),

        distribution_row(
            "S neutral-positive",
            s_cross_condition,
        ),
    ]
)


pairwise_summary.to_csv(
    OUT
    / "pairwise_distribution_summary.csv",
    index=False,
)


same_actor = np.concatenate(
    [
        mm,
        ss,
    ]
)

different_actor = ms


pair_labels = np.concatenate(
    [
        np.ones(
            len(
                same_actor
            )
        ),

        np.zeros(
            len(
                different_actor
            )
        ),
    ]
)

pair_scores = np.concatenate(
    [
        same_actor,
        different_actor,
    ]
)


same_vs_different_auc = float(
    roc_auc_score(
        pair_labels,
        pair_scores,
    )
)


#
# Centroid diagnostics.
#
centroid_rows = [
    {
        "comparison":
            "M centroid vs S centroid",

        "cosine_similarity":
            float(
                np.dot(
                    centroid_M,
                    centroid_S,
                )
            ),
    },

    {
        "comparison":
            "M neutral vs M positive centroid",

        "cosine_similarity":
            float(
                np.dot(
                    centroid_M_neutral,
                    centroid_M_positive,
                )
            ),
    },

    {
        "comparison":
            "S neutral vs S positive centroid",

        "cosine_similarity":
            float(
                np.dot(
                    centroid_S_neutral,
                    centroid_S_positive,
                )
            ),
    },
]


centroid_df = pd.DataFrame(
    centroid_rows
)

centroid_df.to_csv(
    OUT
    / "centroid_diagnostics.csv",
    index=False,
)


#
# Held-out classification against train centroids.
#
heldout_rows = []


for row in embedding_rows:

    if (
        row[
            "evaluation_role"
        ]
        != "heldout"
    ):
        continue


    emb = row[
        "embedding"
    ]

    sim_M = float(
        np.dot(
            emb,
            centroid_M,
        )
    )

    sim_S = float(
        np.dot(
            emb,
            centroid_S,
        )
    )


    actor = row[
        "actor"
    ]


    if actor == "M":
        correct_sim = sim_M
        wrong_sim = sim_S

    elif actor == "S":
        correct_sim = sim_S
        wrong_sim = sim_M

    else:
        raise RuntimeError(
            actor
        )


    predicted = (
        "M"
        if sim_M
        >= sim_S
        else "S"
    )


    heldout_rows.append(
        {
            "split": (
                row[
                    "split"
                ]
            ),

            "basename": (
                row[
                    "basename"
                ]
            ),

            "actor": (
                actor
            ),

            "conditioning_label": (
                row[
                    "conditioning_label"
                ]
            ),

            "source_emotion": (
                row[
                    "source_emotion"
                ]
            ),

            "sim_M": sim_M,
            "sim_S": sim_S,

            "correct_similarity": (
                correct_sim
            ),

            "wrong_similarity": (
                wrong_sim
            ),

            "correct_minus_wrong": (
                correct_sim
                - wrong_sim
            ),

            "predicted_actor": (
                predicted
            ),

            "correct_prediction": (
                predicted
                == actor
            ),
        }
    )


heldout_df = pd.DataFrame(
    heldout_rows
)


heldout_df.to_csv(
    OUT
    / "heldout_actor_scores.csv",
    index=False,
)


#
# Aggregate held-out results.
#
aggregate_rows = []


for group_name, x in (
    (
        "val",
        heldout_df[
            heldout_df[
                "split"
            ]
            == "val"
        ],
    ),

    (
        "test",
        heldout_df[
            heldout_df[
                "split"
            ]
            == "test"
        ],
    ),

    (
        "val+test",
        heldout_df,
    ),
):

    y_true = (
        x[
            "actor"
        ]
        == "M"
    ).astype(
        int
    )


    score_M = (
        x[
            "sim_M"
        ]
        -
        x[
            "sim_S"
        ]
    )


    y_pred = (
        x[
            "predicted_actor"
        ]
        == "M"
    ).astype(
        int
    )


    accuracy = float(
        x[
            "correct_prediction"
        ].mean()
    )


    balanced_accuracy = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )


    auc = float(
        roc_auc_score(
            y_true,
            score_M,
        )
    )


    margin = (
        x[
            "correct_minus_wrong"
        ].to_numpy(
            dtype=float
        )
    )


    accuracy_ci = bootstrap_ci(
        x[
            "correct_prediction"
        ].astype(
            float
        ).to_numpy(),
        statistic="mean",
        seed=(
            BOOTSTRAP_SEED
            + len(
                aggregate_rows
            )
        ),
    )


    margin_ci = bootstrap_ci(
        margin,
        statistic="median",
        seed=(
            BOOTSTRAP_SEED
            + 100
            + len(
                aggregate_rows
            )
        ),
    )


    aggregate_rows.append(
        {
            "group": (
                group_name
            ),

            "n": int(
                len(x)
            ),

            "accuracy": (
                accuracy
            ),

            "accuracy_ci95_low": (
                accuracy_ci[0]
            ),

            "accuracy_ci95_high": (
                accuracy_ci[1]
            ),

            "balanced_accuracy": (
                balanced_accuracy
            ),

            "roc_auc_M_vs_S": (
                auc
            ),

            "median_correct_similarity":
                float(
                    x[
                        "correct_similarity"
                    ].median()
                ),

            "median_wrong_similarity":
                float(
                    x[
                        "wrong_similarity"
                    ].median()
                ),

            "median_margin":
                float(
                    np.median(
                        margin
                    )
                ),

            "margin_ci95_low":
                margin_ci[0],

            "margin_ci95_high":
                margin_ci[1],

            "positive_margin_fraction":
                float(
                    np.mean(
                        margin > 0
                    )
                ),
        }
    )


aggregate_df = pd.DataFrame(
    aggregate_rows
)


aggregate_df.to_csv(
    OUT
    / "heldout_calibration_summary.csv",
    index=False,
)


#
# Actor / conditioning breakdown.
#
breakdown_rows = []


for keys, x in heldout_df.groupby(
    [
        "split",
        "actor",
        "conditioning_label",
    ]
):

    split, actor, label = keys

    breakdown_rows.append(
        {
            "split": split,
            "actor": actor,
            "conditioning_label": (
                label
            ),

            "n": int(
                len(x)
            ),

            "accuracy": float(
                x[
                    "correct_prediction"
                ].mean()
            ),

            "median_correct_similarity":
                float(
                    x[
                        "correct_similarity"
                    ].median()
                ),

            "median_wrong_similarity":
                float(
                    x[
                        "wrong_similarity"
                    ].median()
                ),

            "median_margin":
                float(
                    x[
                        "correct_minus_wrong"
                    ].median()
                ),
        }
    )


breakdown_df = pd.DataFrame(
    breakdown_rows
)


breakdown_df.to_csv(
    OUT
    / "heldout_breakdown.csv",
    index=False,
)


#
# Human-readable report.
#
print()
print("=" * 78)
print(
    "PAIRWISE CALIBRATION"
)
print("=" * 78)

print(
    pairwise_summary.to_string(
        index=False
    )
)

print()
print(
    "same-vs-different actor "
    "ROC-AUC:",
    f"{same_vs_different_auc:.6f}",
)


print()
print("=" * 78)
print(
    "CENTROID DIAGNOSTICS"
)
print("=" * 78)

print(
    centroid_df.to_string(
        index=False
    )
)


print()
print("=" * 78)
print(
    "HELD-OUT CENTROID CLASSIFICATION"
)
print("=" * 78)

print(
    aggregate_df.to_string(
        index=False
    )
)


print()
print("=" * 78)
print(
    "HELD-OUT BREAKDOWN"
)
print("=" * 78)

print(
    breakdown_df.to_string(
        index=False
    )
)


summary = {
    "version": 1,

    "backend_contract": (
        str(
            BACKEND
        )
    ),

    "backend_contract_sha256": (
        EXPECTED_BACKEND_SHA
    ),

    "enrollment": {
        "split": "train",

        "per_actor_per_condition": (
            ENROLL_PER_ACTOR_LABEL
        ),

        "total": int(
            len(
                enrollment
            )
        ),

        "duration_min_sec": (
            MIN_ENROLL_SEC
        ),

        "duration_max_sec": (
            MAX_ENROLL_SEC
        ),

        "selection": (
            "deterministic SHA256 ranking"
        ),

        "selection_seed": (
            SELECTION_SEED
        ),
    },

    "heldout": {
        "splits": [
            "val",
            "test",
        ],

        "total": int(
            len(
                heldout
            )
        ),
    },

    "pairwise_same_vs_different_auc":
        same_vs_different_auc,

    "centroid_M_vs_S_cosine":
        float(
            np.dot(
                centroid_M,
                centroid_S,
            )
        ),

    "new_embeddings_this_run":
        int(
            new_embeddings
        ),

    "bootstrap": {
        "samples": (
            BOOTSTRAP_SAMPLES
        ),

        "seed": (
            BOOTSTRAP_SEED
        ),
    },
}


summary_path = (
    OUT
    / "summary.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


checksum_targets = [
    OUT
    / "enrollment_selection.csv",

    OUT
    / "heldout_selection.csv",

    OUT
    / "actor_centroids.npz",

    OUT
    / "pairwise_distribution_summary.csv",

    OUT
    / "centroid_diagnostics.csv",

    OUT
    / "heldout_actor_scores.csv",

    OUT
    / "heldout_calibration_summary.csv",

    OUT
    / "heldout_breakdown.csv",

    summary_path,
]


with (
    OUT
    / "SHA256SUMS.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    for path in checksum_targets:
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


print()
print("=" * 78)
print(
    "DIALOGS ACTOR WAVLM "
    "CALIBRATION V1: PASS"
)
print("=" * 78)

print(
    "new_embeddings_this_run:",
    new_embeddings,
)

print(
    "output:",
    OUT,
)
