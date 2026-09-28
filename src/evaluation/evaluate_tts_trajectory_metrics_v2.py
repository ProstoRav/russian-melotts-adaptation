#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


PROJECT = Path(
    "${PROJECT_ROOT}"
)

INPUT = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_eval_v2"
)

OLD_METRICS = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_metrics_v1"
)

OUT = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_metrics_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


ASR_CACHE = (
    OUT
    / "asr_cache.jsonl"
)

ACOUSTIC_CACHE = (
    OUT
    / "acoustic_cache.jsonl"
)


WHISPER_ID = (
    "openai/whisper-small"
)

WHISPER_REVISION = (
    "973afd24965f72e36ca33b3055d56a652f456b4d"
)

WHISPER_SNAPSHOT = Path(
    "/home/nuc/.cache/huggingface/hub/"
    "models--openai--whisper-small/"
    "snapshots/"
    "973afd24965f72e36ca33b3055d56a652f456b4d"
)


EXPECTED = {
    "objective_synth": 1300,
    "reference": 249,
    "inventory": 1549,
}


MAIN_EPOCHS = {
    "dialogs": [
        1,
        10,
        20,
        30,
        38,
    ],

    "yocpt": [
        1,
        10,
        20,
        27,
    ],
}


WINNER = {
    "dialogs": 38,
    "yocpt": 27,
}


POST_WINNER = {
    "dialogs": 40,
    "yocpt": 30,
}


# ============================================================
# Generic utilities
# ============================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []

    if not path.is_file():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def append_jsonl(
    path: Path,
    obj,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()

        os.fsync(
            f.fileno()
        )


# ============================================================
# Text normalization
#
# Keep the same strict normalization semantics as metrics_v1.
# This preserves comparability.
#
# Numeral orthography is NOT silently converted here.
# e.g. "14" vs "четырнадцать" remains an ASR orthographic
# mismatch and can later be audited separately.
# ============================================================

def normalize_text(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(
            ch
        )
    )

    text = (
        text
        .lower()
        .replace(
            "ё",
            "е",
        )
    )

    text = re.sub(
        r"[^0-9a-zа-я]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalized_chars(
    text: str,
):
    return list(
        normalize_text(
            text
        ).replace(
            " ",
            "",
        )
    )


def normalized_words(
    text: str,
):
    norm = normalize_text(
        text
    )

    if not norm:
        return []

    return norm.split()


# ============================================================
# Deterministic Levenshtein distance
# ============================================================

def levenshtein_distance(
    ref,
    hyp,
):
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return m

    if m == 0:
        return n

    previous = list(
        range(
            m + 1
        )
    )

    for i in range(
        1,
        n + 1,
    ):

        current = [
            i
        ] + [
            0
        ] * m

        r = ref[
            i - 1
        ]

        for j in range(
            1,
            m + 1,
        ):

            cost = (
                0
                if r == hyp[
                    j - 1
                ]
                else 1
            )

            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1]
                + cost,
            )

        previous = current

    return previous[m]


def text_error_metrics(
    reference: str,
    hypothesis: str,
):
    rw = normalized_words(
        reference
    )

    hw = normalized_words(
        hypothesis
    )

    rc = normalized_chars(
        reference
    )

    hc = normalized_chars(
        hypothesis
    )

    word_edits = (
        levenshtein_distance(
            rw,
            hw,
        )
    )

    char_edits = (
        levenshtein_distance(
            rc,
            hc,
        )
    )

    return {
        "word_edits": (
            word_edits
        ),

        "word_ref_units": (
            len(rw)
        ),

        "char_edits": (
            char_edits
        ),

        "char_ref_units": (
            len(rc)
        ),

        "wer": (
            word_edits
            / len(rw)
            if rw
            else float("nan")
        ),

        "cer": (
            char_edits
            / len(rc)
            if rc
            else float("nan")
        ),
    }


# ============================================================
# Load v2 objective inventory
# ============================================================

manifest_path = (
    INPUT
    / "objective_manifest.jsonl"
)

manifest = load_jsonl(
    manifest_path
)

if len(manifest) != EXPECTED[
    "objective_synth"
]:
    raise RuntimeError(
        "objective manifest count "
        f"{len(manifest)} != 1300"
    )


inventory = []


for experiment in (
    "dialogs",
    "yocpt",
):

    holdout_path = (
        INPUT
        / "objective"
        / experiment
        / "holdout.jsonl"
    )

    holdout = load_jsonl(
        holdout_path
    )

    expected_ref = (
        55
        if experiment
        == "dialogs"
        else 194
    )

    if len(
        holdout
    ) != expected_ref:
        raise RuntimeError(
            f"{experiment}: holdout "
            f"{len(holdout)} != "
            f"{expected_ref}"
        )

    for row in holdout:

        wav = Path(
            row[
                "reference_local"
            ]
        )

        if not wav.is_file():
            raise RuntimeError(
                f"missing reference: "
                f"{wav}"
            )

        inventory.append(
            {
                "experiment": (
                    experiment
                ),

                "holdout_kind": (
                    "test"
                    if experiment
                    == "dialogs"
                    else "validation"
                ),

                "row_id": (
                    row[
                        "row_id"
                    ]
                ),

                "speaker": (
                    row[
                        "speaker"
                    ]
                ),

                "checkpoint_role": (
                    "REFERENCE"
                ),

                "epoch": None,
                "step": None,

                "text": (
                    row[
                        "text"
                    ]
                ),

                "wav": str(wav),

                "reference_wav": (
                    str(wav)
                ),
            }
        )


reference_lookup = {
    (
        r["experiment"],
        r["row_id"],
    ): r["wav"]
    for r in inventory
}


for row in manifest:

    key = (
        row[
            "experiment"
        ],
        row[
            "row_id"
        ],
    )

    reference = (
        reference_lookup.get(
            key
        )
    )

    if reference is None:
        raise RuntimeError(
            f"reference missing for "
            f"{key}"
        )

    inventory.append(
        {
            "experiment": (
                row[
                    "experiment"
                ]
            ),

            "holdout_kind": (
                row[
                    "holdout_kind"
                ]
            ),

            "row_id": (
                row[
                    "row_id"
                ]
            ),

            "speaker": (
                row[
                    "speaker"
                ]
            ),

            "checkpoint_role": (
                row[
                    "checkpoint_role"
                ]
            ),

            "epoch": int(
                row[
                    "epoch"
                ]
            ),

            "step": int(
                row[
                    "step"
                ]
            ),

            "text": (
                row[
                    "text"
                ]
            ),

            "wav": (
                row[
                    "wav"
                ]
            ),

            "reference_wav": (
                reference
            ),
        }
    )


if len(
    inventory
) != EXPECTED[
    "inventory"
]:
    raise RuntimeError(
        "inventory count "
        f"{len(inventory)} "
        "!= 1549"
    )


#
# SHA each input once.
#
for item in inventory:

    wav = Path(
        item["wav"]
    )

    if not wav.is_file():
        raise RuntimeError(
            f"missing WAV: {wav}"
        )

    item[
        "wav_sha256"
    ] = sha256(
        wav
    )


needed_shas = {
    item[
        "wav_sha256"
    ]
    for item in inventory
}


print("=" * 78)
print(
    "TTS TRAJECTORY METRICS V2"
)
print("=" * 78)

print(
    "inventory_audio:",
    len(inventory),
)

print(
    "reference_audio:",
    EXPECTED[
        "reference"
    ],
)

print(
    "synthetic_audio:",
    EXPECTED[
        "objective_synth"
    ],
)

print(
    "unique_wav_sha:",
    len(
        needed_shas
    ),
)


# ============================================================
# Seed V2 caches from V1 caches, by audio SHA
# ============================================================

def cache_to_dict(
    path: Path,
):
    result = {}

    for row in load_jsonl(
        path
    ):
        #
        # Backward compatibility with
        # tts_trajectory_metrics_v1 acoustic cache.
        #
        # V1:
        #   f0_p10_p90_range_hz
        #
        # V2 canonical name:
        #   f0_range_hz
        #
        # Do not mutate the old cache artifact itself.
        #
        if (
            "f0_range_hz" not in row
            and
            "f0_p10_p90_range_hz" in row
        ):
            row["f0_range_hz"] = (
                row[
                    "f0_p10_p90_range_hz"
                ]
            )

        digest = row.get(
            "wav_sha256"
        )

        if digest:
            result[
                digest
            ] = row

    return result


old_asr = cache_to_dict(
    OLD_METRICS
    / "asr_cache.jsonl"
)

old_acoustic = cache_to_dict(
    OLD_METRICS
    / "acoustic_cache.jsonl"
)


asr_by_sha = cache_to_dict(
    ASR_CACHE
)

acoustic_by_sha = (
    cache_to_dict(
        ACOUSTIC_CACHE
    )
)


seeded_asr = 0
seeded_acoustic = 0


for digest in sorted(
    needed_shas
):

    if (
        digest
        not in asr_by_sha
        and digest
        in old_asr
    ):
        row = dict(
            old_asr[
                digest
            ]
        )

        row[
            "cache_origin"
        ] = (
            "tts_trajectory_metrics_v1"
        )

        append_jsonl(
            ASR_CACHE,
            row,
        )

        asr_by_sha[
            digest
        ] = row

        seeded_asr += 1


    if (
        digest
        not in acoustic_by_sha
        and digest
        in old_acoustic
    ):
        row = dict(
            old_acoustic[
                digest
            ]
        )

        row[
            "cache_origin"
        ] = (
            "tts_trajectory_metrics_v1"
        )

        append_jsonl(
            ACOUSTIC_CACHE,
            row,
        )

        acoustic_by_sha[
            digest
        ] = row

        seeded_acoustic += 1


asr_hits = len(
    needed_shas
    & set(
        asr_by_sha
    )
)

acoustic_hits = len(
    needed_shas
    & set(
        acoustic_by_sha
    )
)


print()
print(
    "seeded ASR from v1:",
    seeded_asr,
)

print(
    "seeded acoustic from v1:",
    seeded_acoustic,
)

print(
    "ASR cache hits in v2 inventory:",
    asr_hits,
)

print(
    "ASR missing:",
    len(
        needed_shas
    ) - asr_hits,
)

print(
    "acoustic cache hits:",
    acoustic_hits,
)

print(
    "acoustic missing:",
    len(
        needed_shas
    ) - acoustic_hits,
)


# ============================================================
# Whisper load only if actually needed
# ============================================================

missing_asr = (
    needed_shas
    - set(
        asr_by_sha
    )
)


processor = None
model = None
forced_decoder_ids = None


if missing_asr:

    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "normalizer.json",
        "pytorch_model.bin",
    ]

    missing_files = [
        name
        for name in required
        if not (
            WHISPER_SNAPSHOT
            / name
        ).is_file()
    ]

    if missing_files:
        raise RuntimeError(
            "Whisper snapshot "
            "incomplete: "
            + repr(
                missing_files
            )
        )

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    MODEL_DTYPE = (
        torch.float16
        if DEVICE
        == "cuda"
        else torch.float32
    )

    print()
    print(
        "===== WHISPER ====="
    )

    print(
        "model:",
        WHISPER_ID,
    )

    print(
        "revision:",
        WHISPER_REVISION,
    )

    print(
        "snapshot:",
        WHISPER_SNAPSHOT,
    )

    print(
        "device:",
        DEVICE,
    )

    print(
        "dtype:",
        MODEL_DTYPE,
    )


    processor = (
        WhisperProcessor
        .from_pretrained(
            str(
                WHISPER_SNAPSHOT
            ),
            local_files_only=True,
        )
    )


    model = (
        WhisperForConditionalGeneration
        .from_pretrained(
            str(
                WHISPER_SNAPSHOT
            ),
            local_files_only=True,
            torch_dtype=(
                MODEL_DTYPE
            ),
        )
    )

    model.to(
        DEVICE
    )

    model.eval()


    forced_decoder_ids = (
        processor
        .get_decoder_prompt_ids(
            language="russian",
            task="transcribe",
        )
    )

    print(
        "pinned Whisper load: PASS"
    )

else:
    DEVICE = (
        "cache-only"
    )

    MODEL_DTYPE = (
        "cache-only"
    )


# ============================================================
# ASR
# ============================================================

def whisper_chunk(
    y: np.ndarray,
):
    inputs = processor(
        y,
        sampling_rate=16000,
        return_tensors="pt",
    )

    features = (
        inputs
        .input_features
        .to(
            device=DEVICE,
            dtype=MODEL_DTYPE,
        )
    )

    with torch.inference_mode():

        ids = model.generate(
            features,
            forced_decoder_ids=(
                forced_decoder_ids
            ),
            max_new_tokens=448,
        )

    text = (
        processor
        .batch_decode(
            ids,
            skip_special_tokens=True,
        )[0]
    )

    return text.strip()


def transcribe(
    path: Path,
):
    y, sr = librosa.load(
        str(path),
        sr=16000,
        mono=True,
    )

    if sr != 16000:
        raise RuntimeError(
            "Whisper SR contract failed"
        )

    chunk_samples = (
        25
        * sr
    )

    outputs = []

    for start in range(
        0,
        len(y),
        chunk_samples,
    ):

        chunk = y[
            start:
            start
            + chunk_samples
        ]

        if len(chunk) < 400:
            continue

        outputs.append(
            whisper_chunk(
                chunk
            )
        )

    return (
        " ".join(
            outputs
        ).strip(),
        len(outputs),
    )


# ============================================================
# Acoustic metrics
# ============================================================

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
            f"empty audio: {path}"
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

    if len(finite):

        f0_mean = float(
            np.mean(
                finite
            )
        )

        f0_median = float(
            np.median(
                finite
            )
        )

        f0_std = float(
            np.std(
                finite
            )
        )

        p10 = float(
            np.percentile(
                finite,
                10,
            )
        )

        p90 = float(
            np.percentile(
                finite,
                90,
            )
        )

        f0_range = (
            p90
            - p10
        )

    else:
        f0_mean = float("nan")
        f0_median = float("nan")
        f0_std = float("nan")
        p10 = float("nan")
        p90 = float("nan")
        f0_range = float("nan")


    voiced_fraction = float(
        len(finite)
        / max(
            len(f0),
            1,
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
        "duration_sec": (
            duration
        ),

        "f0_mean_hz": (
            f0_mean
        ),

        "f0_median_hz": (
            f0_median
        ),

        "f0_std_hz": (
            f0_std
        ),

        "f0_p10_hz": (
            p10
        ),

        "f0_p90_hz": (
            p90
        ),

        "f0_range_hz": (
            f0_range
        ),

        "voiced_fraction": (
            voiced_fraction
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


# ============================================================
# Evaluate all v2 inventory
# ============================================================

records = []

new_asr = 0
new_acoustic = 0

total = len(
    inventory
)


for idx, item in enumerate(
    inventory,
    1,
):

    digest = item[
        "wav_sha256"
    ]

    wav = Path(
        item["wav"]
    )


    if digest in asr_by_sha:

        asr = asr_by_sha[
            digest
        ]

        asr_status = (
            "CACHE"
        )

    else:
        text, chunks = (
            transcribe(
                wav
            )
        )

        asr = {
            "wav_sha256": digest,
            "wav": str(wav),
            "asr_text": text,
            "chunks": chunks,
            "model": WHISPER_ID,
            "revision": (
                WHISPER_REVISION
            ),
            "language": "ru",
            "task": (
                "transcribe"
            ),
        }

        append_jsonl(
            ASR_CACHE,
            asr,
        )

        asr_by_sha[
            digest
        ] = asr

        new_asr += 1

        asr_status = "ASR"


    if digest in acoustic_by_sha:

        acoustic = (
            acoustic_by_sha[
                digest
            ]
        )

        acoustic_status = (
            "CACHE"
        )

    else:
        acoustic = {
            "wav_sha256": (
                digest
            ),
            "wav": str(wav),
            **acoustic_metrics(
                wav
            ),
        }

        append_jsonl(
            ACOUSTIC_CACHE,
            acoustic,
        )

        acoustic_by_sha[
            digest
        ] = acoustic

        new_acoustic += 1

        acoustic_status = (
            "ACOUSTIC"
        )


    errors = (
        text_error_metrics(
            item["text"],
            asr[
                "asr_text"
            ],
        )
    )


    records.append(
        {
            **item,

            **errors,

            "reference_text_norm": (
                normalize_text(
                    item[
                        "text"
                    ]
                )
            ),

            "asr_text": (
                asr[
                    "asr_text"
                ]
            ),

            "asr_text_norm": (
                normalize_text(
                    asr[
                        "asr_text"
                    ]
                )
            ),

            "asr_chunks": (
                asr[
                    "chunks"
                ]
            ),

            "duration_sec": (
                acoustic[
                    "duration_sec"
                ]
            ),

            "f0_mean_hz": (
                acoustic[
                    "f0_mean_hz"
                ]
            ),

            "f0_median_hz": (
                acoustic[
                    "f0_median_hz"
                ]
            ),

            "f0_std_hz": (
                acoustic[
                    "f0_std_hz"
                ]
            ),

            "f0_range_hz": (
                acoustic[
                    "f0_range_hz"
                ]
            ),

            "voiced_fraction": (
                acoustic[
                    "voiced_fraction"
                ]
            ),

            "rms_db": (
                acoustic[
                    "rms_db"
                ]
            ),
        }
    )


    print(
        f"[{idx:04d}/{total:04d}] "
        f"{asr_status:5s} "
        f"{acoustic_status:8s} "
        f"{item['experiment']:7s} "
        f"{item['checkpoint_role']:11s} "
        f"{item['row_id']} "
        f"CER={errors['cer']:.4f} "
        f"WER={errors['wer']:.4f}",
        flush=True,
    )


df = pd.DataFrame(
    records
)


# ============================================================
# Reference-paired duration/F0 errors
# ============================================================

refs = (
    df[
        df[
            "checkpoint_role"
        ]
        == "REFERENCE"
    ][
        [
            "experiment",
            "row_id",
            "duration_sec",
            "f0_median_hz",
            "f0_range_hz",
            "voiced_fraction",
            "rms_db",
        ]
    ]
    .rename(
        columns={
            "duration_sec":
                "reference_duration_sec",

            "f0_median_hz":
                "reference_f0_median_hz",

            "f0_range_hz":
                "reference_f0_range_hz",

            "voiced_fraction":
                "reference_voiced_fraction",

            "rms_db":
                "reference_rms_db",
        }
    )
)


df = df.merge(
    refs,
    on=[
        "experiment",
        "row_id",
    ],
    how="left",
)


df[
    "duration_ratio"
] = (
    df[
        "duration_sec"
    ]
    /
    df[
        "reference_duration_sec"
    ]
)


df[
    "duration_abs_pct_error"
] = (
    np.abs(
        df[
            "duration_ratio"
        ]
        - 1.0
    )
    * 100.0
)


df[
    "f0_median_abs_pct_error"
] = (
    np.abs(
        df[
            "f0_median_hz"
        ]
        -
        df[
            "reference_f0_median_hz"
        ]
    )
    /
    df[
        "reference_f0_median_hz"
    ]
    * 100.0
)


ref_mask = (
    df[
        "checkpoint_role"
    ]
    == "REFERENCE"
)

df.loc[
    ref_mask,
    "duration_abs_pct_error"
] = 0.0

df.loc[
    ref_mask,
    "f0_median_abs_pct_error"
] = 0.0


# ============================================================
# Persist per-audio
# ============================================================

df.to_csv(
    OUT
    / "per_audio_metrics.csv",
    index=False,
)

df.to_csv(
    OUT
    / "per_audio_metrics.tsv",
    index=False,
    sep="\t",
)


# ============================================================
# Corpus aggregation
# ============================================================

aggregate_rows = []


group_cols = [
    "experiment",
    "holdout_kind",
    "checkpoint_role",
    "epoch",
]


for keys, g in df.groupby(
    group_cols,
    dropna=False,
):

    (
        experiment,
        holdout_kind,
        checkpoint_role,
        epoch,
    ) = keys


    word_edits = int(
        g[
            "word_edits"
        ].sum()
    )

    word_ref_units = int(
        g[
            "word_ref_units"
        ].sum()
    )

    char_edits = int(
        g[
            "char_edits"
        ].sum()
    )

    char_ref_units = int(
        g[
            "char_ref_units"
        ].sum()
    )


    aggregate_rows.append(
        {
            "experiment": (
                experiment
            ),

            "holdout_kind": (
                holdout_kind
            ),

            "checkpoint_role": (
                checkpoint_role
            ),

            "epoch": epoch,

            "n": int(
                len(g)
            ),

            #
            # Main report metrics.
            #
            "corpus_wer": (
                word_edits
                / word_ref_units
            ),

            "corpus_cer": (
                char_edits
                / char_ref_units
            ),

            "word_edits": (
                word_edits
            ),

            "word_ref_units": (
                word_ref_units
            ),

            "char_edits": (
                char_edits
            ),

            "char_ref_units": (
                char_ref_units
            ),

            #
            # Keep macro metrics for backwards
            # compatibility with metrics_v1.
            #
            "macro_wer_mean": float(
                g[
                    "wer"
                ].mean()
            ),

            "macro_wer_median": float(
                g[
                    "wer"
                ].median()
            ),

            "macro_cer_mean": float(
                g[
                    "cer"
                ].mean()
            ),

            "macro_cer_median": float(
                g[
                    "cer"
                ].median()
            ),

            "duration_mape_mean": float(
                g[
                    "duration_abs_pct_error"
                ].mean()
            ),

            "duration_ratio_mean": float(
                g[
                    "duration_ratio"
                ].mean()
            ),

            "f0_median_hz": float(
                g[
                    "f0_median_hz"
                ].median()
            ),

            "f0_range_hz": float(
                g[
                    "f0_range_hz"
                ].median()
            ),

            "f0_mape_mean": float(
                g[
                    "f0_median_abs_pct_error"
                ].mean()
            ),

            "voiced_fraction_mean": float(
                g[
                    "voiced_fraction"
                ].mean()
            ),

            "rms_db_mean": float(
                g[
                    "rms_db"
                ].mean()
            ),
        }
    )


aggregate = pd.DataFrame(
    aggregate_rows
)


aggregate[
    "epoch_sort"
] = (
    aggregate[
        "epoch"
    ].fillna(
        -1
    )
)


aggregate = aggregate.sort_values(
    [
        "experiment",
        "epoch_sort",
    ]
)


aggregate.to_csv(
    OUT
    / "aggregate_overall.csv",
    index=False,
)


# ============================================================
# Per-speaker aggregate
# ============================================================

speaker_rows = []


for keys, g in df.groupby(
    [
        "experiment",
        "checkpoint_role",
        "epoch",
        "speaker",
    ],
    dropna=False,
):

    (
        experiment,
        checkpoint_role,
        epoch,
        speaker,
    ) = keys


    speaker_rows.append(
        {
            "experiment": (
                experiment
            ),

            "checkpoint_role": (
                checkpoint_role
            ),

            "epoch": epoch,

            "speaker": speaker,

            "n": len(g),

            "corpus_wer": (
                g[
                    "word_edits"
                ].sum()
                /
                g[
                    "word_ref_units"
                ].sum()
            ),

            "corpus_cer": (
                g[
                    "char_edits"
                ].sum()
                /
                g[
                    "char_ref_units"
                ].sum()
            ),

            "macro_wer_mean": (
                g[
                    "wer"
                ].mean()
            ),

            "macro_cer_mean": (
                g[
                    "cer"
                ].mean()
            ),

            "duration_mape_mean": (
                g[
                    "duration_abs_pct_error"
                ].mean()
            ),

            "f0_median_hz": (
                g[
                    "f0_median_hz"
                ].median()
            ),

            "f0_range_hz": (
                g[
                    "f0_range_hz"
                ].median()
            ),

            "voiced_fraction_mean": (
                g[
                    "voiced_fraction"
                ].mean()
            ),

            "rms_db_mean": (
                g[
                    "rms_db"
                ].mean()
            ),
        }
    )


pd.DataFrame(
    speaker_rows
).to_csv(
    OUT
    / "aggregate_by_speaker.csv",
    index=False,
)


# ============================================================
# First -> winner and winner -> post-winner
# ============================================================

first_winner_rows = []
post_rows = []


def get_agg(
    experiment,
    epoch,
):
    x = aggregate[
        (
            aggregate[
                "experiment"
            ]
            == experiment
        )
        &
        (
            aggregate[
                "epoch"
            ]
            == epoch
        )
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"aggregate lookup "
            f"{experiment} E{epoch}: "
            f"{len(x)} rows"
        )

    return x.iloc[0]


for experiment in (
    "dialogs",
    "yocpt",
):

    first = get_agg(
        experiment,
        1,
    )

    winner = get_agg(
        experiment,
        WINNER[
            experiment
        ],
    )

    post = get_agg(
        experiment,
        POST_WINNER[
            experiment
        ],
    )


    first_winner_rows.append(
        {
            "experiment": (
                experiment
            ),

            "first_epoch": 1,

            "winner_epoch": (
                WINNER[
                    experiment
                ]
            ),

            "first_corpus_cer": (
                first[
                    "corpus_cer"
                ]
            ),

            "winner_corpus_cer": (
                winner[
                    "corpus_cer"
                ]
            ),

            "cer_relative_reduction_pct": (
                (
                    1.0
                    -
                    winner[
                        "corpus_cer"
                    ]
                    /
                    first[
                        "corpus_cer"
                    ]
                )
                * 100.0
            ),

            "first_corpus_wer": (
                first[
                    "corpus_wer"
                ]
            ),

            "winner_corpus_wer": (
                winner[
                    "corpus_wer"
                ]
            ),

            "wer_relative_reduction_pct": (
                (
                    1.0
                    -
                    winner[
                        "corpus_wer"
                    ]
                    /
                    first[
                        "corpus_wer"
                    ]
                )
                * 100.0
            ),

            "first_duration_mape": (
                first[
                    "duration_mape_mean"
                ]
            ),

            "winner_duration_mape": (
                winner[
                    "duration_mape_mean"
                ]
            ),
        }
    )


    post_rows.append(
        {
            "experiment": (
                experiment
            ),

            "winner_epoch": (
                WINNER[
                    experiment
                ]
            ),

            "post_epoch": (
                POST_WINNER[
                    experiment
                ]
            ),

            "winner_corpus_cer": (
                winner[
                    "corpus_cer"
                ]
            ),

            "post_corpus_cer": (
                post[
                    "corpus_cer"
                ]
            ),

            "cer_delta_pp": (
                (
                    post[
                        "corpus_cer"
                    ]
                    -
                    winner[
                        "corpus_cer"
                    ]
                )
                * 100.0
            ),

            "cer_relative_change_pct": (
                (
                    post[
                        "corpus_cer"
                    ]
                    /
                    winner[
                        "corpus_cer"
                    ]
                    - 1.0
                )
                * 100.0
            ),

            "winner_corpus_wer": (
                winner[
                    "corpus_wer"
                ]
            ),

            "post_corpus_wer": (
                post[
                    "corpus_wer"
                ]
            ),

            "wer_delta_pp": (
                (
                    post[
                        "corpus_wer"
                    ]
                    -
                    winner[
                        "corpus_wer"
                    ]
                )
                * 100.0
            ),

            "wer_relative_change_pct": (
                (
                    post[
                        "corpus_wer"
                    ]
                    /
                    winner[
                        "corpus_wer"
                    ]
                    - 1.0
                )
                * 100.0
            ),

            "winner_duration_mape": (
                winner[
                    "duration_mape_mean"
                ]
            ),

            "post_duration_mape": (
                post[
                    "duration_mape_mean"
                ]
            ),

            "duration_mape_delta_pp": (
                post[
                    "duration_mape_mean"
                ]
                -
                winner[
                    "duration_mape_mean"
                ]
            ),

            "winner_f0_median_hz": (
                winner[
                    "f0_median_hz"
                ]
            ),

            "post_f0_median_hz": (
                post[
                    "f0_median_hz"
                ]
            ),

            "winner_f0_range_hz": (
                winner[
                    "f0_range_hz"
                ]
            ),

            "post_f0_range_hz": (
                post[
                    "f0_range_hz"
                ]
            ),
        }
    )


first_winner = pd.DataFrame(
    first_winner_rows
)

post_sanity = pd.DataFrame(
    post_rows
)


first_winner.to_csv(
    OUT
    / "first_to_winner.csv",
    index=False,
)


post_sanity.to_csv(
    OUT
    / "post_winner_sanity.csv",
    index=False,
)


# ============================================================
# Human-readable output
# ============================================================

print()
print("=" * 78)
print(
    "OVERALL METRICS V2"
)
print("=" * 78)


for experiment in (
    "dialogs",
    "yocpt",
):

    print()
    print(
        experiment.upper()
    )

    x = aggregate[
        aggregate[
            "experiment"
        ]
        == experiment
    ].copy()

    x = x.sort_values(
        "epoch_sort"
    )


    for _, row in x.iterrows():

        if pd.isna(
            row[
                "epoch"
            ]
        ):
            epoch_label = (
                "REF"
            )
        else:
            epoch_label = (
                f"E{int(row['epoch'])}"
            )

        print(
            f"{row['checkpoint_role']:11s} "
            f"{epoch_label:4s} "
            f"n={int(row['n']):3d} "
            f"cCER={row['corpus_cer']:.4f} "
            f"cWER={row['corpus_wer']:.4f} "
            f"mCER={row['macro_cer_mean']:.4f} "
            f"mWER={row['macro_wer_mean']:.4f} "
            f"durMAPE={row['duration_mape_mean']:.2f}% "
            f"F0med={row['f0_median_hz']:.2f}Hz "
            f"F0range={row['f0_range_hz']:.2f}Hz"
        )


print()
print("=" * 78)
print(
    "FIRST -> WINNER"
)
print("=" * 78)

print(
    first_winner.to_string(
        index=False
    )
)


print()
print("=" * 78)
print(
    "POST-WINNER SANITY"
)
print("=" * 78)

print(
    post_sanity.to_string(
        index=False
    )
)


# ============================================================
# Plots
# ============================================================

PLOTS = (
    OUT
    / "plots"
)

PLOTS.mkdir(
    parents=True,
    exist_ok=True,
)


for experiment in (
    "dialogs",
    "yocpt",
):

    main_epochs = (
        MAIN_EPOCHS[
            experiment
        ]
    )

    x = aggregate[
        (
            aggregate[
                "experiment"
            ]
            == experiment
        )
        &
        (
            aggregate[
                "epoch"
            ].isin(
                main_epochs
            )
        )
    ].copy()

    x = x.sort_values(
        "epoch"
    )


    ref = aggregate[
        (
            aggregate[
                "experiment"
            ]
            == experiment
        )
        &
        (
            aggregate[
                "checkpoint_role"
            ]
            == "REFERENCE"
        )
    ].iloc[0]


    #
    # Main CER/WER trajectory.
    #
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x[
            "epoch"
        ],
        x[
            "corpus_cer"
        ]
        * 100.0,
        marker="o",
        label="Corpus CER",
    )

    ax.plot(
        x[
            "epoch"
        ],
        x[
            "corpus_wer"
        ]
        * 100.0,
        marker="o",
        label="Corpus WER",
    )

    ax.axhline(
        ref[
            "corpus_cer"
        ]
        * 100.0,
        linestyle="--",
        label=(
            "Natural-reference CER"
        ),
    )

    ax.axhline(
        ref[
            "corpus_wer"
        ]
        * 100.0,
        linestyle=":",
        label=(
            "Natural-reference WER"
        ),
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Error, %"
    )

    ax.set_title(
        f"{experiment.upper()}: "
        "main intelligibility trajectory"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOTS
        / (
            f"{experiment}_"
            "main_corpus_cer_wer.png"
        ),
        dpi=180,
    )

    plt.close(fig)


    #
    # Main duration trajectory.
    #
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x[
            "epoch"
        ],
        x[
            "duration_mape_mean"
        ],
        marker="o",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Duration MAPE, %"
    )

    ax.set_title(
        f"{experiment.upper()}: "
        "main duration trajectory"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        PLOTS
        / (
            f"{experiment}_"
            "main_duration_mape.png"
        ),
        dpi=180,
    )

    plt.close(fig)


    #
    # Winner vs post-winner only.
    #
    winner = get_agg(
        experiment,
        WINNER[
            experiment
        ],
    )

    post = get_agg(
        experiment,
        POST_WINNER[
            experiment
        ],
    )

    epochs = [
        WINNER[
            experiment
        ],
        POST_WINNER[
            experiment
        ],
    ]


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        epochs,
        [
            winner[
                "corpus_cer"
            ]
            * 100.0,
            post[
                "corpus_cer"
            ]
            * 100.0,
        ],
        marker="o",
        label="Corpus CER",
    )

    ax.plot(
        epochs,
        [
            winner[
                "corpus_wer"
            ]
            * 100.0,
            post[
                "corpus_wer"
            ]
            * 100.0,
        ],
        marker="o",
        label="Corpus WER",
    )

    ax.set_xticks(
        epochs
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Error, %"
    )

    ax.set_title(
        f"{experiment.upper()}: "
        "winner vs post-winner sanity"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PLOTS
        / (
            f"{experiment}_"
            "post_winner_sanity.png"
        ),
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# Provenance
# ============================================================

summary = {
    "version": 2,

    "input_eval": (
        str(INPUT)
    ),

    "input_manifest_sha256": (
        sha256(
            manifest_path
        )
    ),

    "inventory_audio": (
        len(inventory)
    ),

    "reference_audio": 249,

    "synthetic_audio": 1300,

    "new_asr_this_run": (
        new_asr
    ),

    "new_acoustic_this_run": (
        new_acoustic
    ),

    "seeded_asr_from_v1": (
        seeded_asr
    ),

    "seeded_acoustic_from_v1": (
        seeded_acoustic
    ),

    "whisper": {
        "model": (
            WHISPER_ID
        ),

        "revision": (
            WHISPER_REVISION
        ),

        "snapshot": (
            str(
                WHISPER_SNAPSHOT
            )
        ),

        "language": "russian",
        "task": "transcribe",
    },

    "normalization": {
        "unicode": "NFKD",
        "combining_marks_removed": True,
        "lowercase": True,
        "yo_to_e": True,
        "punctuation_removed": True,
        "cer_spaces_removed": True,
        "numeral_verbalization": False,
    },

    "metrics": {
        "corpus_wer": (
            "sum word Levenshtein edits "
            "/ sum reference words"
        ),

        "corpus_cer": (
            "sum character Levenshtein edits "
            "/ sum reference characters"
        ),

        "macro_metrics_retained": True,

        "duration_mape": (
            "mean absolute percentage error "
            "vs matched natural reference"
        ),

        "f0_method": (
            "librosa.pyin"
        ),

        "f0_analysis_sr": 16000,
        "f0_min_hz": 60,
        "f0_max_hz": 600,
    },

    "main_epochs": (
        MAIN_EPOCHS
    ),

    "winner": (
        WINNER
    ),

    "post_winner": (
        POST_WINNER
    ),

    "post_winner_excluded_from_main_trajectory": True,
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
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
    "TTS TRAJECTORY METRICS V2: PASS"
)
print("=" * 78)

print(
    "new_asr_this_run:",
    new_asr,
)

print(
    "new_acoustic_this_run:",
    new_acoustic,
)

print(
    "output:",
    OUT,
)

print(
    "plots:",
    PLOTS,
)
