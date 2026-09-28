#!${VENV_ROOT}/bin/python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import huggingface_hub
import librosa
import numpy as np
import pandas as pd
import torch
import transformers

from huggingface_hub import snapshot_download
from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMForXVector,
)


PROJECT = Path(
    "${PROJECT_ROOT}"
)

MODEL_ID = (
    "microsoft/wavlm-base-plus-sv"
)

MODEL_REVISION = (
    "1bfd64eca136543feb28c5ffaf05381c6af33121"
)

OUT = (
    PROJECT
    / "artifacts/frozen/"
      "speaker_embedding_backend_"
      "wavlm_base_plus_sv_v1"
)

AUDIT = (
    PROJECT
    / "artifacts/eval/"
      "dialogs_actual_speaker_audit_v1/"
      "selected_rows.csv"
)

YOCPT_HOLDOUT = (
    PROJECT
    / "artifacts/eval/"
      "tts_trajectory_eval_v2/"
      "objective/yocpt/"
      "holdout.jsonl"
)


if OUT.exists():
    raise RuntimeError(
        f"frozen artifact already exists: {OUT}"
    )

OUT.mkdir(
    parents=True,
    exist_ok=False,
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


print("=" * 78)
print("SPEAKER EMBEDDING BACKEND PREP V1")
print("=" * 78)

print(
    "model:",
    MODEL_ID,
)

print(
    "revision:",
    MODEL_REVISION,
)

print(
    "torch:",
    torch.__version__,
)

print(
    "transformers:",
    transformers.__version__,
)

print(
    "huggingface_hub:",
    huggingface_hub.__version__,
)

print(
    "librosa:",
    librosa.__version__,
)


#
# Download only the exact files needed.
#
snapshot = Path(
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        ],
    )
)


if (
    snapshot.name
    != MODEL_REVISION
):
    raise RuntimeError(
        "resolved snapshot revision mismatch: "
        f"{snapshot.name}"
    )


required_files = [
    snapshot / "config.json",
    snapshot / "preprocessor_config.json",
    snapshot / "pytorch_model.bin",
]


for path in required_files:

    if not path.is_file():
        raise RuntimeError(
            f"missing model file: {path}"
        )


print()
print(
    "snapshot:",
    snapshot,
)

print()
print("===== MODEL FILES =====")

model_file_hashes = {}

for path in required_files:

    digest = sha256(
        path
    )

    model_file_hashes[
        path.name
    ] = digest

    print(
        path.name,
        path.stat().st_size,
        digest,
    )


#
# Load strictly from the pinned local snapshot.
#
feature_extractor = (
    Wav2Vec2FeatureExtractor
    .from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
)


model = (
    WavLMForXVector
    .from_pretrained(
        str(snapshot),
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
print("===== RUNTIME =====")

print(
    "device:",
    device,
)

if (
    device
    == "cuda"
):
    print(
        "device_name:",
        torch.cuda.get_device_name(
            0
        ),
    )

print(
    "model_dtype:",
    next(
        model.parameters()
    ).dtype,
)


ANALYSIS_SR = 16000
TRIM_TOP_DB = 40
WINDOW_SEC = 8.0
HOP_SEC = 4.0


def load_audio(
    path: Path,
):
    y, sr = librosa.load(
        str(path),
        sr=ANALYSIS_SR,
        mono=True,
    )

    if sr != ANALYSIS_SR:
        raise RuntimeError(
            "resample contract failed"
        )

    if not len(y):
        raise RuntimeError(
            f"empty audio: {path}"
        )

    trimmed, _ = (
        librosa.effects.trim(
            y,
            top_db=TRIM_TOP_DB,
        )
    )

    #
    # Do not accidentally replace a very quiet
    # utterance with an empty signal.
    #
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
def embed(
    path: Path,
):
    y = load_audio(
        path
    )

    windows = make_windows(
        y
    )

    embeddings = []

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
            key: value.to(
                device
            )
            for key, value
            in inputs.items()
        }

        output = model(
            **kwargs
        )

        emb = (
            output.embeddings[
                0
            ]
            .float()
        )

        emb = (
            torch.nn.functional
            .normalize(
                emb,
                dim=0,
            )
        )

        embeddings.append(
            emb.cpu()
        )

    stacked = torch.stack(
        embeddings,
        dim=0,
    )

    combined = torch.mean(
        stacked,
        dim=0,
    )

    combined = (
        torch.nn.functional
        .normalize(
            combined,
            dim=0,
        )
    )

    result = combined.numpy()

    return (
        result,
        len(windows),
        len(y)
        / ANALYSIS_SR,
    )


#
# Select smoke files.
#
with YOCPT_HOLDOUT.open(
    "r",
    encoding="utf-8",
) as f:

    first_line = next(
        line
        for line in f
        if line.strip()
    )

yocpt_row = json.loads(
    first_line
)

yocpt_ref = Path(
    yocpt_row[
        "reference_local"
    ]
)


audit = pd.read_csv(
    AUDIT
)

test = audit[
    audit[
        "split"
    ]
    == "test"
]


m_rows = (
    test[
        test[
            "source_speaker_id"
        ]
        == "M"
    ]
    .head(2)
)

s_rows = (
    test[
        test[
            "source_speaker_id"
        ]
        == "S"
    ]
    .head(2)
)


if (
    len(m_rows) != 2
    or len(s_rows) != 2
):
    raise RuntimeError(
        "need at least 2 M and 2 S test rows"
    )


m1 = Path(
    m_rows.iloc[0][
        "wav"
    ]
)

m2 = Path(
    m_rows.iloc[1][
        "wav"
    ]
)

s1 = Path(
    s_rows.iloc[0][
        "wav"
    ]
)

s2 = Path(
    s_rows.iloc[1][
        "wav"
    ]
)


for path in (
    yocpt_ref,
    m1,
    m2,
    s1,
    s2,
):

    if not path.is_file():
        raise RuntimeError(
            f"smoke WAV missing: {path}"
        )


print()
print("===== EMBEDDING SMOKE =====")


y1, y_windows, y_sec = embed(
    yocpt_ref
)

#
# Duplicate extraction is intentional:
# verifies inference reproducibility.
#
y2, _, _ = embed(
    yocpt_ref
)

m1e, m1w, m1sec = embed(
    m1
)

m2e, m2w, m2sec = embed(
    m2
)

s1e, s1w, s1sec = embed(
    s1
)

s2e, s2w, s2sec = embed(
    s2
)


same_file_cos = cosine(
    y1,
    y2,
)

m_m_cos = cosine(
    m1e,
    m2e,
)

s_s_cos = cosine(
    s1e,
    s2e,
)

m_s_cos = cosine(
    m1e,
    s1e,
)


print(
    "embedding_dim:",
    y1.shape[0],
)

print(
    "YO-CPT ref:",
    yocpt_ref,
)

print(
    "YO-CPT duration_sec:",
    round(
        y_sec,
        3,
    ),
)

print(
    "YO-CPT windows:",
    y_windows,
)

print(
    "YO-CPT norm:",
    float(
        np.linalg.norm(
            y1
        )
    ),
)

print(
    "same-file cosine:",
    f"{same_file_cos:.9f}",
)


print()
print(
    "M1:",
    m1.name,
    "sec=",
    round(
        m1sec,
        3,
    ),
    "windows=",
    m1w,
)

print(
    "M2:",
    m2.name,
    "sec=",
    round(
        m2sec,
        3,
    ),
    "windows=",
    m2w,
)

print(
    "S1:",
    s1.name,
    "sec=",
    round(
        s1sec,
        3,
    ),
    "windows=",
    s1w,
)

print(
    "S2:",
    s2.name,
    "sec=",
    round(
        s2sec,
        3,
    ),
    "windows=",
    s2w,
)


print()
print(
    "M1 vs M2 cosine:",
    f"{m_m_cos:.6f}",
)

print(
    "S1 vs S2 cosine:",
    f"{s_s_cos:.6f}",
)

print(
    "M1 vs S1 cosine:",
    f"{m_s_cos:.6f}",
)


if not (
    0.999
    <= np.linalg.norm(
        y1
    )
    <= 1.001
):
    raise RuntimeError(
        "embedding normalization failed"
    )


if same_file_cos < 0.99999:
    raise RuntimeError(
        "same-file reproducibility failed"
    )


contract = {
    "version": 1,

    "backend": {
        "model_id": MODEL_ID,
        "revision": (
            MODEL_REVISION
        ),
        "snapshot": (
            str(
                snapshot
            )
        ),
        "files_sha256": (
            model_file_hashes
        ),
    },

    "runtime": {
        "torch": (
            torch.__version__
        ),
        "transformers": (
            transformers.__version__
        ),
        "huggingface_hub": (
            huggingface_hub.__version__
        ),
        "librosa": (
            librosa.__version__
        ),
        "numpy": (
            np.__version__
        ),
        "device": (
            device
        ),
        "dtype": "float32",
    },

    "preprocessing": {
        "mono": True,
        "sampling_rate": (
            ANALYSIS_SR
        ),
        "leading_trailing_trim": (
            "librosa.effects.trim"
        ),
        "trim_top_db": (
            TRIM_TOP_DB
        ),
        "window_sec": (
            WINDOW_SEC
        ),
        "hop_sec": (
            HOP_SEC
        ),
        "window_embedding": (
            "L2 normalized"
        ),
        "utterance_embedding": (
            "mean of window embeddings, "
            "then L2 normalized"
        ),
    },

    "similarity": (
        "cosine similarity of "
        "L2-normalized embeddings"
    ),

    "smoke": {
        "same_file_cosine": (
            same_file_cos
        ),
        "M1_M2_cosine": (
            m_m_cos
        ),
        "S1_S2_cosine": (
            s_s_cos
        ),
        "M1_S1_cosine": (
            m_s_cos
        ),
    },
}


contract_path = (
    OUT
    / "contract.json"
)

contract_path.write_text(
    json.dumps(
        contract,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


(
    OUT
    / "SHA256SUMS.txt"
).write_text(
    (
        f"{sha256(contract_path)}  "
        "contract.json\n"
    ),
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
    "SPEAKER EMBEDDING BACKEND: PASS"
)
print("=" * 78)

print(
    "contract:",
    contract_path,
)

print(
    "contract_sha256:",
    sha256(
        contract_path
    ),
)
