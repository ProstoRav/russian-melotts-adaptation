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
import soundfile as sf
import torch

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

EVAL_V2 = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_eval_v2"
)

OBJECTIVE_MANIFEST = (
    EVAL_V2
    / "objective_manifest.jsonl"
)

DEFENSE_MANIFEST = (
    EVAL_V2
    / "defense_manifest.jsonl"
)

YOCPT_TRAIN = (
    PROJECT
    / "artifacts/frozen/"
      "spk298_melotts_no_bert_stress_v1/"
      "train.list"
)

YOCPT_VAL = (
    PROJECT
    / "artifacts/frozen/"
      "spk298_melotts_no_bert_stress_v1/"
      "val.list"
)

DIALOGS_AUDIT = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_actual_speaker_audit_v1/"
      "selected_rows.csv"
)

DIALOGS_CALIBRATION = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_actor_wavlm_calibration_v1"
)

ACTOR_CENTROIDS = (
    DIALOGS_CALIBRATION
    / "actor_centroids.npz"
)

ACTOR_CALIBRATION_TEST = (
    DIALOGS_CALIBRATION
    / "heldout_actor_scores.csv"
)

BACKEND_CONTRACT = (
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
      "speaker_identity_eval_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE = (
    OUT
    / "embedding_cache.jsonl"
)


YOCPT_EPOCHS = [
    1,
    10,
    20,
    27,
    30,
]

YOCPT_CHOSEN = 27

DIALOGS_EPOCHS = [
    1,
    10,
    20,
    30,
    38,
    40,
]

DIALOGS_CHOSEN = 38

DEFENSE_EPOCHS = [
    1,
    10,
    20,
    30,
    38,
]


YOCPT_ENROLL_N = 200
YOCPT_ENROLL_MIN_SEC = 1.5
YOCPT_ENROLL_MAX_SEC = 15.0
YOCPT_SELECTION_SEED = 2982026

ANALYSIS_SR = 16000
TRIM_TOP_DB = 40
WINDOW_SEC = 8.0
HOP_SEC = 4.0

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 29841


# ============================================================
# Utilities
# ============================================================

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

        f.flush()

        os.fsync(
            f.fileno()
        )


def normalize_vector(
    vector,
):
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = float(
        np.linalg.norm(
            vector
        )
    )

    if norm <= 0:
        raise RuntimeError(
            "zero embedding"
        )

    return (
        vector
        / norm
    )


def cosine(
    a,
    b,
):
    return float(
        np.dot(
            a,
            b,
        )
    )


def bootstrap_ci(
    values,
    statistic,
    seed,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    if not len(values):
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAP_SAMPLES,
            len(values),
        ),
    )

    samples = values[
        indices
    ]

    if statistic == "median":
        stats = np.median(
            samples,
            axis=1,
        )

    elif statistic == "mean":
        stats = np.mean(
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


def summary_stats(
    values,
    seed,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    ci_low, ci_high = (
        bootstrap_ci(
            values,
            statistic="median",
            seed=seed,
        )
    )

    return {
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

        "median_ci95_low": (
            ci_low
        ),

        "median_ci95_high": (
            ci_high
        ),
    }


# ============================================================
# Backend contract
# ============================================================

if (
    sha256(
        BACKEND_CONTRACT
    )
    != EXPECTED_BACKEND_SHA
):
    raise RuntimeError(
        "speaker backend contract "
        "SHA mismatch"
    )


backend = json.loads(
    BACKEND_CONTRACT.read_text(
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
            f"missing backend file: "
            f"{required}"
        )


# ============================================================
# Load backend strictly locally
# ============================================================

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


print("=" * 78)
print(
    "SPEAKER / VOICE IDENTITY EVAL V1"
)
print("=" * 78)

print(
    "backend:",
    backend[
        "backend"
    ][
        "model_id"
    ],
)

print(
    "revision:",
    backend[
        "backend"
    ][
        "revision"
    ],
)

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


# ============================================================
# Embedding extraction
# ============================================================

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

    last = (
        len(y)
        - max_len
    )

    if (
        not starts
        or starts[-1] != last
    ):
        starts.append(
            last
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
def embed_raw(
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


cache = {
    row[
        "wav_sha256"
    ]: row
    for row in (
        load_jsonl(
            CACHE
        )
        if CACHE.is_file()
        else []
    )
}


new_embeddings = 0


def get_embedding(
    path: Path,
):
    global new_embeddings

    if not path.is_file():
        raise RuntimeError(
            f"missing WAV: {path}"
        )

    digest = sha256(
        path
    )

    if digest in cache:

        row = cache[
            digest
        ]

        vector = normalize_vector(
            row[
                "embedding"
            ]
        )

        return (
            vector,
            "CACHE",
        )

    result = embed_raw(
        path
    )

    row = {
        "wav_sha256": (
            digest
        ),

        "wav": str(
            path
        ),

        **result,
    }

    append_jsonl(
        CACHE,
        row,
    )

    cache[
        digest
    ] = row

    new_embeddings += 1

    return (
        normalize_vector(
            result[
                "embedding"
            ]
        ),
        "EMBED",
    )


# ============================================================
# YO-CPT enrollment selection
# ============================================================

train_rows = []


with YOCPT_TRAIN.open(
    "r",
    encoding="utf-8",
) as f:

    for index, line in enumerate(
        f,
        1,
    ):

        line = line.rstrip(
            "\n"
        )

        if not line:
            continue

        fields = line.split(
            "|"
        )

        if len(fields) != 7:
            raise RuntimeError(
                f"train row {index}: "
                f"NF={len(fields)}"
            )

        path = Path(
            fields[0]
        )

        if not path.is_file():
            raise RuntimeError(
                f"missing train WAV: "
                f"{path}"
            )

        info = sf.info(
            str(
                path
            )
        )

        duration = (
            info.frames
            / info.samplerate
        )

        if (
            YOCPT_ENROLL_MIN_SEC
            <= duration
            <= YOCPT_ENROLL_MAX_SEC
        ):

            rank = hashlib.sha256(
                (
                    f"{YOCPT_SELECTION_SEED}|"
                    f"{path}"
                ).encode(
                    "utf-8"
                )
            ).hexdigest()

            train_rows.append(
                {
                    "wav": str(
                        path
                    ),

                    "duration_sec": (
                        duration
                    ),

                    "rank": rank,
                }
            )


if (
    len(train_rows)
    < YOCPT_ENROLL_N
):
    raise RuntimeError(
        "not enough eligible "
        "YO-CPT enrollment WAVs"
    )


enrollment = (
    pd.DataFrame(
        train_rows
    )
    .sort_values(
        "rank"
    )
    .head(
        YOCPT_ENROLL_N
    )
    .drop(
        columns=[
            "rank"
        ]
    )
    .reset_index(
        drop=True
    )
)


enrollment.to_csv(
    OUT
    / "yocpt_enrollment_selection.csv",
    index=False,
)


print()
print("=" * 78)
print(
    "YO-CPT ENROLLMENT"
)
print("=" * 78)

print(
    "eligible:",
    len(
        train_rows
    ),
)

print(
    "selected:",
    len(
        enrollment
    ),
)


enroll_vectors = []


for idx, row in enrollment.iterrows():

    vector, status = (
        get_embedding(
            Path(
                row[
                    "wav"
                ]
            )
        )
    )

    enroll_vectors.append(
        vector
    )

    print(
        f"[YO-ENROLL "
        f"{idx + 1:03d}/"
        f"{len(enrollment):03d}] "
        f"{status}",
        flush=True,
    )


yocpt_centroid = normalize_vector(
    np.mean(
        np.stack(
            enroll_vectors,
            axis=0,
        ),
        axis=0,
    )
)


np.save(
    OUT
    / "yocpt_spk298_centroid.npy",
    yocpt_centroid,
)


# ============================================================
# Objective manifests
# ============================================================

objective = load_jsonl(
    OBJECTIVE_MANIFEST
)

if len(objective) != 1300:
    raise RuntimeError(
        f"objective rows="
        f"{len(objective)}, "
        "expected 1300"
    )


yocpt_synth = [
    row
    for row in objective
    if row[
        "experiment"
    ] == "yocpt"
]


dialogs_synth = [
    row
    for row in objective
    if row[
        "experiment"
    ] == "dialogs"
]


if len(yocpt_synth) != 970:
    raise RuntimeError(
        f"YO synth="
        f"{len(yocpt_synth)}"
    )


if len(dialogs_synth) != 330:
    raise RuntimeError(
        f"Dialogs synth="
        f"{len(dialogs_synth)}"
    )


# ============================================================
# YO-CPT natural validation references
# ============================================================

yocpt_holdout = load_jsonl(
    EVAL_V2
    / "objective/yocpt/"
      "holdout.jsonl"
)


if len(yocpt_holdout) != 194:
    raise RuntimeError(
        f"YO holdout="
        f"{len(yocpt_holdout)}"
    )


yocpt_ref_vectors = {}
yocpt_ref_rows = []


for idx, row in enumerate(
    yocpt_holdout,
    1,
):

    path = Path(
        row[
            "reference_local"
        ]
    )

    vector, status = (
        get_embedding(
            path
        )
    )

    row_id = row[
        "row_id"
    ]

    yocpt_ref_vectors[
        row_id
    ] = vector

    similarity = cosine(
        vector,
        yocpt_centroid,
    )

    yocpt_ref_rows.append(
        {
            "row_id": (
                row_id
            ),

            "wav": str(
                path
            ),

            "similarity_to_enrollment_centroid":
                similarity,
        }
    )

    print(
        f"[YO-REF "
        f"{idx:03d}/194] "
        f"{status} "
        f"{row_id}",
        flush=True,
    )


yocpt_ref_df = pd.DataFrame(
    yocpt_ref_rows
)


yocpt_ref_df.to_csv(
    OUT
    / "yocpt_natural_reference_scores.csv",
    index=False,
)


# ============================================================
# YO-CPT synthetic speaker preservation
# ============================================================

yocpt_score_rows = []


for idx, row in enumerate(
    yocpt_synth,
    1,
):

    epoch = int(
        row[
            "epoch"
        ]
    )

    if epoch not in (
        YOCPT_EPOCHS
    ):
        raise RuntimeError(
            f"unexpected YO epoch "
            f"{epoch}"
        )

    row_id = row[
        "row_id"
    ]

    path = Path(
        row[
            "wav"
        ]
    )

    vector, status = (
        get_embedding(
            path
        )
    )

    ref_vector = (
        yocpt_ref_vectors[
            row_id
        ]
    )

    yocpt_score_rows.append(
        {
            "epoch": (
                epoch
            ),

            "checkpoint_role": (
                row[
                    "checkpoint_role"
                ]
            ),

            "row_id": (
                row_id
            ),

            "wav": str(
                path
            ),

            "similarity_to_enrollment_centroid":
                cosine(
                    vector,
                    yocpt_centroid,
                ),

            "similarity_to_matched_natural":
                cosine(
                    vector,
                    ref_vector,
                ),
        }
    )

    print(
        f"[YO-SYN "
        f"{idx:03d}/970] "
        f"{status} "
        f"E{epoch:02d} "
        f"{row_id}",
        flush=True,
    )


yocpt_scores = pd.DataFrame(
    yocpt_score_rows
)


yocpt_scores.to_csv(
    OUT
    / "yocpt_synthetic_scores.csv",
    index=False,
)


#
# Natural-reference baseline + trajectory.
#
yocpt_summary_rows = []


reference_stats = summary_stats(
    yocpt_ref_df[
        "similarity_to_enrollment_centroid"
    ],
    BOOTSTRAP_SEED,
)


yocpt_summary_rows.append(
    {
        "checkpoint": (
            "NATURAL_REFERENCE"
        ),

        "epoch": (
            np.nan
        ),

        "metric": (
            "similarity_to_"
            "enrollment_centroid"
        ),

        **reference_stats,
    }
)


for epoch_idx, epoch in enumerate(
    YOCPT_EPOCHS
):

    x = yocpt_scores[
        yocpt_scores[
            "epoch"
        ]
        == epoch
    ]

    for metric_idx, metric in enumerate(
        [
            "similarity_to_enrollment_centroid",
            "similarity_to_matched_natural",
        ]
    ):

        stats = summary_stats(
            x[
                metric
            ],
            (
                BOOTSTRAP_SEED
                + 10
                + epoch_idx * 10
                + metric_idx
            ),
        )

        yocpt_summary_rows.append(
            {
                "checkpoint": (
                    "CHOSEN_CANDIDATE"
                    if epoch
                    == YOCPT_CHOSEN
                    else "EPOCH"
                ),

                "epoch": (
                    epoch
                ),

                "metric": (
                    metric
                ),

                **stats,
            }
        )


yocpt_summary = pd.DataFrame(
    yocpt_summary_rows
)


yocpt_summary.to_csv(
    OUT
    / "yocpt_speaker_similarity_summary.csv",
    index=False,
)


# ============================================================
# Dialogs actor centroids
# ============================================================

centroid_data = np.load(
    ACTOR_CENTROIDS
)


centroid_M = normalize_vector(
    centroid_data[
        "M"
    ]
)

centroid_S = normalize_vector(
    centroid_data[
        "S"
    ]
)


dialogs_audit = pd.read_csv(
    DIALOGS_AUDIT
)


dialogs_test_audit = (
    dialogs_audit[
        dialogs_audit[
            "split"
        ]
        == "test"
    ]
    .copy()
)


audit_by_basename = {
    row[
        "basename"
    ]: row
    for _, row in (
        dialogs_test_audit
        .iterrows()
    )
}


dialogs_holdout = load_jsonl(
    EVAL_V2
    / "objective/dialogs/"
      "holdout.jsonl"
)


if len(dialogs_holdout) != 55:
    raise RuntimeError(
        f"Dialogs holdout="
        f"{len(dialogs_holdout)}"
    )


dialogs_row_meta = {}


for row in dialogs_holdout:

    #
    # `reference_local` is the canonical evaluation copy
    # named R0001.wav ... R0055.wav.
    #
    # `wav` is the original Dialogs source WAV and therefore
    # carries the basename used by the frozen actor audit.
    #
    source_wav = Path(
        row[
            "wav"
        ]
    )

    basename = (
        source_wav.name
    )

    if not source_wav.is_file():
        raise RuntimeError(
            f"Dialogs source WAV missing: "
            f"{source_wav}"
        )

    if basename not in (
        audit_by_basename
    ):
        raise RuntimeError(
            f"Dialogs audit missing "
            f"{basename}"
        )

    audit_row = (
        audit_by_basename[
            basename
        ]
    )

    #
    # Cross-check that the holdout conditioning label
    # and actor-audit conditioning label agree.
    #
    if (
        str(
            row[
                "speaker"
            ]
        )
        !=
        str(
            audit_row[
                "artificial_label"
            ]
        )
    ):
        raise RuntimeError(
            f"Dialogs conditioning-label mismatch: "
            f"{row['row_id']} "
            f"{basename}: "
            f"holdout={row['speaker']} "
            f"audit={audit_row['artificial_label']}"
        )

    dialogs_row_meta[
        row[
            "row_id"
        ]
    ] = {
        "basename": basename,

        "source_wav": str(
            source_wav
        ),

        "source_actor": (
            audit_row[
                "source_speaker_id"
            ]
        ),

        "conditioning_label": (
            audit_row[
                "artificial_label"
            ]
        ),

        "source_emotion": (
            audit_row[
                "source_emotion"
            ]
        ),
    }


# ============================================================
# Dialogs objective actor-centroid affinity
# ============================================================

dialogs_score_rows = []


for idx, row in enumerate(
    dialogs_synth,
    1,
):

    epoch = int(
        row[
            "epoch"
        ]
    )

    if epoch not in (
        DIALOGS_EPOCHS
    ):
        raise RuntimeError(
            f"unexpected Dialogs "
            f"epoch {epoch}"
        )

    row_id = row[
        "row_id"
    ]

    meta = dialogs_row_meta[
        row_id
    ]

    vector, status = (
        get_embedding(
            Path(
                row[
                    "wav"
                ]
            )
        )
    )

    sim_M = cosine(
        vector,
        centroid_M,
    )

    sim_S = cosine(
        vector,
        centroid_S,
    )

    source_actor = (
        meta[
            "source_actor"
        ]
    )

    if source_actor == "M":
        source_margin = (
            sim_M - sim_S
        )

    elif source_actor == "S":
        source_margin = (
            sim_S - sim_M
        )

    else:
        raise RuntimeError(
            source_actor
        )

    predicted_actor = (
        "M"
        if sim_M >= sim_S
        else "S"
    )

    dialogs_score_rows.append(
        {
            "epoch": epoch,

            "checkpoint_role": (
                row[
                    "checkpoint_role"
                ]
            ),

            "row_id": row_id,

            "basename": (
                meta[
                    "basename"
                ]
            ),

            "conditioning_label": (
                meta[
                    "conditioning_label"
                ]
            ),

            "source_actor": (
                source_actor
            ),

            "source_emotion": (
                meta[
                    "source_emotion"
                ]
            ),

            "sim_M": sim_M,
            "sim_S": sim_S,

            "M_minus_S": (
                sim_M
                - sim_S
            ),

            "source_actor_affinity_margin":
                source_margin,

            "nearest_actor_centroid":
                predicted_actor,

            "nearest_matches_source_actor":
                (
                    predicted_actor
                    == source_actor
                ),
        }
    )

    print(
        f"[DLG-SYN "
        f"{idx:03d}/330] "
        f"{status} "
        f"E{epoch:02d} "
        f"{row_id}",
        flush=True,
    )


dialogs_scores = pd.DataFrame(
    dialogs_score_rows
)


dialogs_scores.to_csv(
    OUT
    / "dialogs_actor_affinity_scores.csv",
    index=False,
)


#
# Per epoch + conditioning label.
#
dialogs_summary_rows = []


for (
    epoch,
    label
), x in dialogs_scores.groupby(
    [
        "epoch",
        "conditioning_label",
    ]
):

    for metric_idx, metric in enumerate(
        [
            "sim_M",
            "sim_S",
            "M_minus_S",
            "source_actor_affinity_margin",
        ]
    ):

        stats = summary_stats(
            x[
                metric
            ],
            (
                BOOTSTRAP_SEED
                + 1000
                + int(
                    epoch
                ) * 10
                + metric_idx
            ),
        )

        dialogs_summary_rows.append(
            {
                "epoch": int(
                    epoch
                ),

                "checkpoint": (
                    "CHOSEN_CANDIDATE"
                    if int(
                        epoch
                    )
                    == DIALOGS_CHOSEN
                    else "EPOCH"
                ),

                "conditioning_label": (
                    label
                ),

                "metric": (
                    metric
                ),

                **stats,
            }
        )


    dialogs_summary_rows.append(
        {
            "epoch": int(
                epoch
            ),

            "checkpoint": (
                "CHOSEN_CANDIDATE"
                if int(
                    epoch
                )
                == DIALOGS_CHOSEN
                else "EPOCH"
            ),

            "conditioning_label": (
                label
            ),

            "metric": (
                "nearest_matches_"
                "source_actor_fraction"
            ),

            "n": int(
                len(x)
            ),

            "mean": float(
                x[
                    "nearest_matches_source_actor"
                ].mean()
            ),

            "median": float(
                x[
                    "nearest_matches_source_actor"
                ].median()
            ),

            "std": float(
                x[
                    "nearest_matches_source_actor"
                ].astype(
                    float
                ).std(
                    ddof=0
                )
            ),

            "p10": float(
                np.percentile(
                    x[
                        "nearest_matches_source_actor"
                    ].astype(
                        float
                    ),
                    10,
                )
            ),

            "q1": float(
                np.percentile(
                    x[
                        "nearest_matches_source_actor"
                    ].astype(
                        float
                    ),
                    25,
                )
            ),

            "q3": float(
                np.percentile(
                    x[
                        "nearest_matches_source_actor"
                    ].astype(
                        float
                    ),
                    75,
                )
            ),

            "p90": float(
                np.percentile(
                    x[
                        "nearest_matches_source_actor"
                    ].astype(
                        float
                    ),
                    90,
                )
            ),

            "median_ci95_low": (
                np.nan
            ),

            "median_ci95_high": (
                np.nan
            ),
        }
    )


dialogs_summary = pd.DataFrame(
    dialogs_summary_rows
)


dialogs_summary.to_csv(
    OUT
    / "dialogs_actor_affinity_summary.csv",
    index=False,
)


# ============================================================
# Natural test calibration reference
# ============================================================

natural_calibration = pd.read_csv(
    ACTOR_CALIBRATION_TEST
)


natural_test = natural_calibration[
    natural_calibration[
        "split"
    ]
    == "test"
].copy()


natural_test.to_csv(
    OUT
    / "dialogs_natural_test_actor_baseline.csv",
    index=False,
)


# ============================================================
# Controlled S01-S09 SID0 vs SID1 voice consistency
# ============================================================

defense = load_jsonl(
    DEFENSE_MANIFEST
)


if len(defense) != 90:
    raise RuntimeError(
        f"defense rows="
        f"{len(defense)}"
    )


defense_vectors = {}


for idx, row in enumerate(
    defense,
    1,
):

    epoch = int(
        row[
            "epoch"
        ]
    )

    if epoch not in (
        DEFENSE_EPOCHS
    ):
        raise RuntimeError(
            f"unexpected defense "
            f"epoch {epoch}"
        )

    vector, status = (
        get_embedding(
            Path(
                row[
                    "wav"
                ]
            )
        )
    )

    key = (
        epoch,
        row[
            "prompt_id"
        ],
        int(
            row[
                "speaker_id"
            ]
        ),
    )

    if key in defense_vectors:
        raise RuntimeError(
            f"duplicate defense key "
            f"{key}"
        )

    defense_vectors[
        key
    ] = vector

    print(
        f"[DEFENSE "
        f"{idx:02d}/90] "
        f"{status} "
        f"E{epoch:02d} "
        f"{row['prompt_id']} "
        f"SID{int(row['speaker_id'])}",
        flush=True,
    )


defense_pair_rows = []


for epoch in DEFENSE_EPOCHS:

    for prompt_num in range(
        1,
        10,
    ):

        prompt_id = (
            f"S{prompt_num:02d}"
        )

        neutral = (
            defense_vectors[
                (
                    epoch,
                    prompt_id,
                    0,
                )
            ]
        )

        positive = (
            defense_vectors[
                (
                    epoch,
                    prompt_id,
                    1,
                )
            ]
        )

        neutral_M = cosine(
            neutral,
            centroid_M,
        )

        neutral_S = cosine(
            neutral,
            centroid_S,
        )

        positive_M = cosine(
            positive,
            centroid_M,
        )

        positive_S = cosine(
            positive,
            centroid_S,
        )

        defense_pair_rows.append(
            {
                "epoch": (
                    epoch
                ),

                "checkpoint": (
                    "CHOSEN_CANDIDATE"
                    if epoch
                    == DIALOGS_CHOSEN
                    else "EPOCH"
                ),

                "prompt_id": (
                    prompt_id
                ),

                "sid0_sid1_cosine":
                    cosine(
                        neutral,
                        positive,
                    ),

                "sid0_sim_M": (
                    neutral_M
                ),

                "sid0_sim_S": (
                    neutral_S
                ),

                "sid0_M_minus_S": (
                    neutral_M
                    - neutral_S
                ),

                "sid1_sim_M": (
                    positive_M
                ),

                "sid1_sim_S": (
                    positive_S
                ),

                "sid1_M_minus_S": (
                    positive_M
                    - positive_S
                ),
            }
        )


defense_pairs = pd.DataFrame(
    defense_pair_rows
)


defense_pairs.to_csv(
    OUT
    / "dialogs_defense_voice_consistency.csv",
    index=False,
)


defense_summary_rows = []


for epoch_idx, epoch in enumerate(
    DEFENSE_EPOCHS
):

    x = defense_pairs[
        defense_pairs[
            "epoch"
        ]
        == epoch
    ]

    for metric_idx, metric in enumerate(
        [
            "sid0_sid1_cosine",
            "sid0_M_minus_S",
            "sid1_M_minus_S",
        ]
    ):

        stats = summary_stats(
            x[
                metric
            ],
            (
                BOOTSTRAP_SEED
                + 2000
                + epoch_idx * 10
                + metric_idx
            ),
        )

        defense_summary_rows.append(
            {
                "epoch": (
                    epoch
                ),

                "checkpoint": (
                    "CHOSEN_CANDIDATE"
                    if epoch
                    == DIALOGS_CHOSEN
                    else "EPOCH"
                ),

                "metric": (
                    metric
                ),

                **stats,
            }
        )


defense_summary = pd.DataFrame(
    defense_summary_rows
)


defense_summary.to_csv(
    OUT
    / "dialogs_defense_voice_consistency_summary.csv",
    index=False,
)


# ============================================================
# Human-readable report
# ============================================================

print()
print("=" * 78)
print(
    "YO-CPT NATURAL SPEAKER BASELINE"
)
print("=" * 78)

print(
    "Natural validation -> "
    "train enrollment centroid"
)

print(
    pd.DataFrame(
        [
            reference_stats
        ]
    ).to_string(
        index=False
    )
)


print()
print("=" * 78)
print(
    "YO-CPT SPEAKER PRESERVATION"
)
print("=" * 78)


for epoch in YOCPT_EPOCHS:

    x = yocpt_summary[
        yocpt_summary[
            "epoch"
        ]
        == epoch
    ]

    centroid_row = x[
        x[
            "metric"
        ]
        ==
        "similarity_to_enrollment_centroid"
    ].iloc[0]

    matched_row = x[
        x[
            "metric"
        ]
        ==
        "similarity_to_matched_natural"
    ].iloc[0]

    suffix = (
        "  CHOSEN_CANDIDATE"
        if epoch
        == YOCPT_CHOSEN
        else ""
    )

    print(
        f"E{epoch:02d}"
        f"{suffix}"
        f"  centroid="
        f"{centroid_row['median']:.6f} "
        f"CI95=["
        f"{centroid_row['median_ci95_low']:.6f}, "
        f"{centroid_row['median_ci95_high']:.6f}] "
        f"matched_ref="
        f"{matched_row['median']:.6f} "
        f"CI95=["
        f"{matched_row['median_ci95_low']:.6f}, "
        f"{matched_row['median_ci95_high']:.6f}]"
    )


print()
print("=" * 78)
print(
    "DIALOGS NATURAL TEST ACTOR BASELINE"
)
print("=" * 78)

print(
    "n:",
    len(
        natural_test
    ),
)

print(
    "accuracy:",
    float(
        natural_test[
            "correct_prediction"
        ].mean()
    ),
)

print(
    "median correct similarity:",
    float(
        natural_test[
            "correct_similarity"
        ].median()
    ),
)

print(
    "median wrong similarity:",
    float(
        natural_test[
            "wrong_similarity"
        ].median()
    ),
)

print(
    "median margin:",
    float(
        natural_test[
            "correct_minus_wrong"
        ].median()
    ),
)


print()
print("=" * 78)
print(
    "DIALOGS SYNTHETIC ACTOR-CENTROID AFFINITY"
)
print("=" * 78)


for epoch in DIALOGS_EPOCHS:

    print()
    print(
        f"E{epoch:02d}"
        + (
            "  CHOSEN_CANDIDATE"
            if epoch
            == DIALOGS_CHOSEN
            else ""
        )
    )

    x = dialogs_scores[
        dialogs_scores[
            "epoch"
        ]
        == epoch
    ]

    for label in (
        "dialogs_female_neutral",
        "dialogs_female_positive",
    ):

        y = x[
            x[
                "conditioning_label"
            ]
            == label
        ]

        short = (
            "neutral"
            if label.endswith(
                "neutral"
            )
            else "positive"
        )

        print(
            f"  {short:8s} "
            f"sim_M="
            f"{y['sim_M'].median():.6f} "
            f"sim_S="
            f"{y['sim_S'].median():.6f} "
            f"M-S="
            f"{y['M_minus_S'].median():+.6f} "
            f"source-affinity-margin="
            f"{y['source_actor_affinity_margin'].median():+.6f} "
            f"source-match="
            f"{y['nearest_matches_source_actor'].mean():.3f}"
        )


print()
print("=" * 78)
print(
    "DIALOGS CONTROLLED SID0/SID1 "
    "VOICE CONSISTENCY"
)
print("=" * 78)


for epoch in DEFENSE_EPOCHS:

    x = defense_summary[
        (
            defense_summary[
                "epoch"
            ]
            == epoch
        )
        &
        (
            defense_summary[
                "metric"
            ]
            ==
            "sid0_sid1_cosine"
        )
    ].iloc[0]

    print(
        f"E{epoch:02d}"
        + (
            "  CHOSEN_CANDIDATE"
            if epoch
            == DIALOGS_CHOSEN
            else ""
        )
        + (
            f"  cosine="
            f"{x['median']:.6f} "
            f"CI95=["
            f"{x['median_ci95_low']:.6f}, "
            f"{x['median_ci95_high']:.6f}] "
            f"P10-P90=["
            f"{x['p10']:.6f}, "
            f"{x['p90']:.6f}]"
        )
    )


# ============================================================
# Provenance
# ============================================================

summary = {
    "version": 1,

    "backend_contract_sha256": (
        EXPECTED_BACKEND_SHA
    ),

    "backend_snapshot": (
        str(
            snapshot
        )
    ),

    "embedding_dim": 512,

    "preprocessing": (
        backend[
            "preprocessing"
        ]
    ),

    "yocpt": {
        "interpretation": (
            "speaker preservation"
        ),

        "enrollment_split": (
            "train"
        ),

        "enrollment_n": (
            YOCPT_ENROLL_N
        ),

        "evaluation_split": (
            "validation holdout"
        ),

        "evaluation_n": 194,

        "epochs": (
            YOCPT_EPOCHS
        ),

        "chosen_candidate": (
            YOCPT_CHOSEN
        ),
    },

    "dialogs": {
        "interpretation": (
            "actor-centroid affinity / "
            "synthetic voice consistency; "
            "not speaker preservation"
        ),

        "actor_centroids": (
            str(
                ACTOR_CENTROIDS
            )
        ),

        "objective_test_n": 55,

        "epochs": (
            DIALOGS_EPOCHS
        ),

        "chosen_candidate": (
            DIALOGS_CHOSEN
        ),

        "controlled_prompts": 9,

        "controlled_epochs": (
            DEFENSE_EPOCHS
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
    },

    "new_embeddings_this_run": (
        new_embeddings
    ),
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
    / "yocpt_enrollment_selection.csv",

    OUT
    / "yocpt_spk298_centroid.npy",

    OUT
    / "yocpt_natural_reference_scores.csv",

    OUT
    / "yocpt_synthetic_scores.csv",

    OUT
    / "yocpt_speaker_similarity_summary.csv",

    OUT
    / "dialogs_actor_affinity_scores.csv",

    OUT
    / "dialogs_actor_affinity_summary.csv",

    OUT
    / "dialogs_natural_test_actor_baseline.csv",

    OUT
    / "dialogs_defense_voice_consistency.csv",

    OUT
    / "dialogs_defense_voice_consistency_summary.csv",

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
    "SPEAKER / VOICE IDENTITY EVAL V1: PASS"
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
