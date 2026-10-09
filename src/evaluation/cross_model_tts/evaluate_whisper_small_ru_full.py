#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


dst = Path(sys.argv[1]).resolve()
snapshot = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

prompts_path = (
    dst / "prompts.benchmark.jsonl"
)

per_prompt_path = (
    output_dir / "per_prompt.jsonl"
)

summary_path = (
    output_dir / "system_summary.json"
)


SYSTEMS = [
    "melotts_yocpt_e27",
    "melotts_dialogs_e38_neutral",
    "melotts_dialogs_e38_positive",
    "dialogs_ru_vits2",
    "silero_v5_5_ru",
    "xtts_v2_spk298",
    "piper_ru_irina_medium",
]

WHISPER_ID = "openai/whisper-small"

WHISPER_REVISION = (
    "973afd24965f72e36ca33b3055d56a652f456b4d"
)

NORMALIZATION_VERSION = "ru_simple_v1"

LANGUAGE = "russian"
TASK = "transcribe"

CHUNK_SECONDS = 25
MAX_NEW_TOKENS = 448


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def normalize_text(text: str) -> str:
    # Exact historical ru_simple_v1.
    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.lower().replace(
        "ё",
        "е",
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


def normalize_cer(text: str) -> str:
    return normalize_text(
        text
    ).replace(
        " ",
        "",
    )


def safe_wer(ref, hyp):
    if not ref:
        return float("nan")

    return float(
        jiwer.wer(
            ref,
            hyp,
        )
    )


def safe_cer(ref, hyp):
    if not ref:
        return float("nan")

    return float(
        jiwer.cer(
            ref,
            hyp,
        )
    )


def find_wav(system_id, prompt_id):
    root = (
        dst /
        "renders" /
        system_id
    )

    matches = sorted(
        root.glob(
            f"{prompt_id}*.wav"
        )
    )

    assert len(matches) == 1, (
        system_id,
        prompt_id,
        matches,
    )

    return matches[0]


def scalar_stats(values):
    assert values

    return {
        "mean":
            statistics.mean(values),

        "median":
            statistics.median(values),

        "min":
            min(values),

        "max":
            max(values),

        "stdev":
            (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            ),
    }


# ------------------------------------------------------------
# Frozen prompts.
# ------------------------------------------------------------

prompt_rows = [
    json.loads(x)
    for x in prompts_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if x.strip()
]

assert len(prompt_rows) == 9

assert [
    x["id"]
    for x in prompt_rows
] == [
    f"S{i:02d}"
    for i in range(1, 10)
]

prompt_by_id = {
    x["id"]: x
    for x in prompt_rows
}

for row in prompt_rows:
    assert row["text_plain"]
    assert "\u0301" not in row["text_plain"]


# ------------------------------------------------------------
# Canonical 7 x 9 WAV inventory.
# ------------------------------------------------------------

matrix = []

for system_id in SYSTEMS:
    for n in range(1, 10):

        prompt_id = f"S{n:02d}"
        wav = find_wav(
            system_id,
            prompt_id,
        )

        info = sf.info(wav)

        assert info.channels == 1
        assert info.frames > 0

        matrix.append({
            "system_id":
                system_id,

            "prompt_id":
                prompt_id,

            "wav_path":
                wav,

            "wav_sha256":
                sha256(wav),

            "source_sample_rate_hz":
                info.samplerate,

            "source_duration_seconds":
                info.duration,

            "reference_raw":
                prompt_by_id[
                    prompt_id
                ]["text_plain"],
        })


assert len(matrix) == 63


# ------------------------------------------------------------
# Pinned Whisper snapshot audit.
# ------------------------------------------------------------

required_files = [
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

missing = [
    name
    for name in required_files
    if not (
        snapshot / name
    ).is_file()
]

assert not missing, missing


# ------------------------------------------------------------
# Exact historical runtime.
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MODEL_DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

print("model =", WHISPER_ID)
print("revision =", WHISPER_REVISION)
print("snapshot =", snapshot)
print("device =", DEVICE)
print("dtype =", MODEL_DTYPE)

if DEVICE == "cuda":
    print(
        "physical_device =",
        torch.cuda.get_device_name(0),
    )


processor = (
    WhisperProcessor.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
)

model = (
    WhisperForConditionalGeneration
    .from_pretrained(
        str(snapshot),
        local_files_only=True,
        torch_dtype=MODEL_DTYPE,
    )
)

model.to(DEVICE)
model.eval()


forced_decoder_ids = (
    processor.get_decoder_prompt_ids(
        language=LANGUAGE,
        task=TASK,
    )
)

print(
    "forced_decoder_ids =",
    forced_decoder_ids,
)

print(
    "local Whisper-small load: PASS"
)


def whisper_chunk(
    y: np.ndarray,
):
    inputs = processor(
        y,
        sampling_rate=16000,
        return_tensors="pt",
    )

    features = (
        inputs.input_features.to(
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
            max_new_tokens=(
                MAX_NEW_TOKENS
            ),
        )

    text = (
        processor.batch_decode(
            ids,
            skip_special_tokens=True,
        )[0]
    )

    return text.strip()


def transcribe(path: Path):
    # Exact historical 25-second deterministic chunks.
    y, sr = librosa.load(
        str(path),
        sr=16000,
        mono=True,
    )

    if sr != 16000:
        raise RuntimeError(
            "Whisper resample contract failed"
        )

    chunk_samples = (
        CHUNK_SECONDS
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
            start + chunk_samples
        ]

        if len(chunk) < 400:
            continue

        outputs.append(
            whisper_chunk(
                chunk
            )
        )

    return (
        " ".join(outputs).strip(),
        len(outputs),
        len(y),
    )


# ------------------------------------------------------------
# Warmup excluded.
# Use first canonical WAV computationally,
# then recompute it officially.
# ------------------------------------------------------------

print()
print("===== WARMUP =====")

warm_t0 = time.perf_counter()

warm_text, warm_chunks, _ = transcribe(
    matrix[0]["wav_path"]
)

if DEVICE == "cuda":
    torch.cuda.synchronize()

warm_seconds = (
    time.perf_counter()
    - warm_t0
)

print(
    "warmup_seconds =",
    warm_seconds,
)

print(
    "warmup_chunks =",
    warm_chunks,
)

print(
    "warmup_text =",
    repr(warm_text),
)


# ------------------------------------------------------------
# Resume support.
# ------------------------------------------------------------

completed = {}

if per_prompt_path.exists():
    for line in (
        per_prompt_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):
        if not line.strip():
            continue

        row = json.loads(line)

        key = (
            row["system_id"],
            row["prompt_id"],
        )

        completed[key] = row


print(
    "resume_completed =",
    len(completed),
)


# ------------------------------------------------------------
# Official ASR + WER/CER.
# ------------------------------------------------------------

for seq, item in enumerate(
    matrix,
    start=1,
):
    key = (
        item["system_id"],
        item["prompt_id"],
    )

    if key in completed:
        old = completed[key]

        assert (
            old["wav_sha256"]
            == item["wav_sha256"]
        )

        continue

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    hyp_raw, n_chunks, samples_16k = (
        transcribe(
            item["wav_path"]
        )
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    asr_seconds = (
        time.perf_counter()
        - t0
    )

    ref_raw = item["reference_raw"]

    ref_norm = normalize_text(
        ref_raw
    )

    hyp_norm = normalize_text(
        hyp_raw
    )

    ref_cer = normalize_cer(
        ref_raw
    )

    hyp_cer = normalize_cer(
        hyp_raw
    )

    wer = safe_wer(
        ref_norm,
        hyp_norm,
    )

    cer = safe_cer(
        ref_cer,
        hyp_cer,
    )

    assert not math.isnan(wer)
    assert not math.isnan(cer)

    row = {
        "metric_family":
            "Whisper-small ASR intelligibility",

        "system_id":
            item["system_id"],

        "prompt_id":
            item["prompt_id"],

        "wav_path":
            str(item["wav_path"]),

        "wav_sha256":
            item["wav_sha256"],

        "source_sample_rate_hz":
            item[
                "source_sample_rate_hz"
            ],

        "source_duration_seconds":
            item[
                "source_duration_seconds"
            ],

        "asr_sample_rate_hz":
            16000,

        "asr_samples":
            samples_16k,

        "whisper_chunks":
            n_chunks,

        "reference_field":
            "text_plain",

        "reference_raw":
            ref_raw,

        "hypothesis_raw":
            hyp_raw,

        "reference_normalized":
            ref_norm,

        "hypothesis_normalized":
            hyp_norm,

        "reference_cer_normalized":
            ref_cer,

        "hypothesis_cer_normalized":
            hyp_cer,

        "WER":
            wer,

        "CER":
            cer,

        "asr_seconds":
            asr_seconds,

        "normalization":
            NORMALIZATION_VERSION,

        "whisper_model":
            WHISPER_ID,

        "whisper_revision":
            WHISPER_REVISION,

        "language":
            LANGUAGE,

        "task":
            TASK,

        "max_new_tokens":
            MAX_NEW_TOKENS,

        "chunk_seconds":
            CHUNK_SECONDS,

        "device":
            DEVICE,

        "dtype":
            str(MODEL_DTYPE),

        "torch_version":
            torch.__version__,
    }

    with per_prompt_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

    completed[key] = row

    print(
        f"[{seq:02d}/63] "
        f"{item['system_id']} "
        f"{item['prompt_id']} "
        f"WER={wer:.4f} "
        f"CER={cer:.4f} "
        f"chunks={n_chunks} "
        f"time={asr_seconds:.3f}s",
        flush=True,
    )


# ------------------------------------------------------------
# Integrity.
# ------------------------------------------------------------

rows = list(
    completed.values()
)

assert len(rows) == 63

expected_keys = {
    (
        system_id,
        f"S{n:02d}",
    )
    for system_id in SYSTEMS
    for n in range(1, 10)
}

actual_keys = {
    (
        x["system_id"],
        x["prompt_id"],
    )
    for x in rows
}

assert actual_keys == expected_keys


system_order = {
    x: i
    for i, x in enumerate(
        SYSTEMS
    )
}

rows = sorted(
    rows,
    key=lambda x: (
        system_order[
            x["system_id"]
        ],
        x["prompt_id"],
    ),
)

# Canonical rewrite after resume.
with per_prompt_path.open(
    "w",
    encoding="utf-8",
) as f:
    for row in rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
        )


# ------------------------------------------------------------
# System summaries.
#
# Report:
# 1. micro/corpus WER and CER from summed edit counts;
# 2. mean/median per-prompt values.
# ------------------------------------------------------------

summary = {
    "metric_family":
        "Whisper-small ASR intelligibility",

    "whisper_model":
        WHISPER_ID,

    "whisper_revision":
        WHISPER_REVISION,

    "snapshot":
        str(snapshot),

    "normalization":
        NORMALIZATION_VERSION,

    "reference_field":
        "text_plain",

    "language":
        LANGUAGE,

    "task":
        TASK,

    "chunk_seconds":
        CHUNK_SECONDS,

    "max_new_tokens":
        MAX_NEW_TOKENS,

    "device":
        DEVICE,

    "dtype":
        str(MODEL_DTYPE),

    "torch_version":
        torch.__version__,

    "n_systems":
        7,

    "n_prompts_per_system":
        9,

    "n_wavs":
        63,

    "systems":
        {},
}


for system_id in SYSTEMS:
    selected = [
        x
        for x in rows
        if x["system_id"] == system_id
    ]

    assert len(selected) == 9

    wers = [
        x["WER"]
        for x in selected
    ]

    cers = [
        x["CER"]
        for x in selected
    ]

    # Sum edit counts per utterance rather than joining
    # utterances, so edits cannot cross prompt boundaries.
    w_sub = 0
    w_del = 0
    w_ins = 0
    w_hits = 0

    c_sub = 0
    c_del = 0
    c_ins = 0
    c_hits = 0

    for row in selected:
        w = jiwer.process_words(
            row[
                "reference_normalized"
            ],
            row[
                "hypothesis_normalized"
            ],
        )

        w_sub += w.substitutions
        w_del += w.deletions
        w_ins += w.insertions
        w_hits += w.hits

        c = jiwer.process_characters(
            row[
                "reference_cer_normalized"
            ],
            row[
                "hypothesis_cer_normalized"
            ],
        )

        c_sub += c.substitutions
        c_del += c.deletions
        c_ins += c.insertions
        c_hits += c.hits

    word_ref_total = (
        w_hits
        + w_sub
        + w_del
    )

    char_ref_total = (
        c_hits
        + c_sub
        + c_del
    )

    assert word_ref_total > 0
    assert char_ref_total > 0

    micro_wer = (
        w_sub
        + w_del
        + w_ins
    ) / word_ref_total

    micro_cer = (
        c_sub
        + c_del
        + c_ins
    ) / char_ref_total

    summary["systems"][
        system_id
    ] = {
        "n_prompts":
            9,

        "micro_WER":
            micro_wer,

        "micro_CER":
            micro_cer,

        "WER_per_prompt":
            scalar_stats(wers),

        "CER_per_prompt":
            scalar_stats(cers),

        "word_edit_counts": {
            "hits":
                w_hits,

            "substitutions":
                w_sub,

            "deletions":
                w_del,

            "insertions":
                w_ins,

            "reference_words":
                word_ref_total,
        },

        "char_edit_counts": {
            "hits":
                c_hits,

            "substitutions":
                c_sub,

            "deletions":
                c_del,

            "insertions":
                c_ins,

            "reference_characters":
                char_ref_total,
        },
    }


summary[
    "per_prompt_sha256"
] = sha256(
    per_prompt_path
)


summary_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print()
print("=" * 78)
print("FINAL SYSTEM WER / CER")
print("=" * 78)

for system_id in SYSTEMS:
    s = summary[
        "systems"
    ][system_id]

    print(
        f"{system_id:36s} "
        f"micro_WER={s['micro_WER']:.6f} "
        f"micro_CER={s['micro_CER']:.6f} "
        f"WER_mean={s['WER_per_prompt']['mean']:.6f} "
        f"CER_mean={s['CER_per_prompt']['mean']:.6f}"
    )


print()
print(
    "per_prompt =",
    per_prompt_path,
)

print(
    "summary =",
    summary_path,
)

print()
print(
    "WHISPER-SMALL RU FULL "
    "WER/CER EVALUATION: PASS"
)
