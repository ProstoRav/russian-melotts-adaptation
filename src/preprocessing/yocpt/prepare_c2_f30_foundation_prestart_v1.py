from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


PROJECT = Path.home() / "projects/ru_melotts_research"
MELO = Path.home() / "projects/MeloTTS"

BASE_CONFIG = (
    PROJECT
    / "artifacts/pretrain/runB_transfer_v1/config.json"
)

DATA = (
    PROJECT
    / "artifacts/c2_yocpt_c2_f30_melotts_no_bert_v1"
)

TRAIN = DATA / "train.list"
VAL = DATA / "val.list"

OUT = (
    PROJECT
    / "artifacts/prestart/c2_f30_foundation_v1"
)

EXPERIMENT = (
    PROJECT
    / "experiments/c2_f30_foundation_v1"
)

CONFIG = OUT / "config.json"
PRETRAIN_G = OUT / "pretrain_G.pth"
REPORT = OUT / "transfer_report.json"

ENGLISH_V3 = (
    Path.home()
    / "checkpoints/MeloTTS-English-v3/checkpoint.pth"
)

EXPECTED_ENGLISH_SHA256 = (
    "959433dc1c6df618922560b4b7bc8c7a"
    "f0a4b7ceaea267480c5d9ae8a3cfe536"
)

EXPECTED_TRAIN = 12257
EXPECTED_VAL = 432

SPEAKER = "c2_f30_generic_ru"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


if OUT.exists():
    raise RuntimeError(
        f"refusing to overwrite prestart artifact: {OUT}"
    )


for path in (
    BASE_CONFIG,
    TRAIN,
    VAL,
    ENGLISH_V3,
):
    if not path.is_file():
        raise RuntimeError(
            f"missing required file: {path}"
        )


source_sha = sha256_file(
    ENGLISH_V3
)

if source_sha != EXPECTED_ENGLISH_SHA256:
    raise RuntimeError(
        "English-v3 SHA256 mismatch:\n"
        f"actual   {source_sha}\n"
        f"expected {EXPECTED_ENGLISH_SHA256}"
    )


train_rows = TRAIN.read_text(
    encoding="utf-8"
).splitlines()

val_rows = VAL.read_text(
    encoding="utf-8"
).splitlines()

if len(train_rows) != EXPECTED_TRAIN:
    raise RuntimeError(
        f"train rows={len(train_rows)}"
    )

if len(val_rows) != EXPECTED_VAL:
    raise RuntimeError(
        f"val rows={len(val_rows)}"
    )


#
# Verify local MeloTTS frontend/config vocabulary remains
# identical to the known-good frozen Run B configuration.
#
sys.path.insert(
    0,
    str(MELO),
)

from melo.text.symbols import (  # noqa: E402
    symbols,
    num_tones,
    num_languages,
    language_id_map,
)


cfg = json.loads(
    BASE_CONFIG.read_text(
        encoding="utf-8"
    )
)


if cfg["symbols"] != list(symbols):
    raise RuntimeError(
        "current MeloTTS symbols differ from "
        "known-good Run B config"
    )

if int(cfg["num_tones"]) != int(num_tones):
    raise RuntimeError(
        "num_tones mismatch"
    )

if int(cfg["num_languages"]) != int(num_languages):
    raise RuntimeError(
        "num_languages mismatch"
    )

if "RU" not in language_id_map:
    raise RuntimeError(
        "RU missing from language_id_map"
    )

ru_language_id = int(
    language_id_map["RU"]
)

if ru_language_id >= int(
    cfg["num_languages"]
):
    raise RuntimeError(
        "RU language ID outside embedding table"
    )


#
# ---------------------------------------------------------
# Keep all known-good optimization/model settings.
# Patch only C2-F data/conditioning/runtime identity.
# ---------------------------------------------------------
#
train = cfg["train"]
data = cfg["data"]


train.update({
    "seed": 52,

    "batch_size": 2,

    "learning_rate": 0.0003,

    "fp16_run": False,

    "skip_optimizer": False,

    "log_interval": 100,

    "eval_interval": 1000,
})


data.update({
    "training_files":
        str(TRAIN.resolve()),

    "validation_files":
        str(VAL.resolve()),

    #
    # Deliberately preserve English-v3 model geometry.
    # Physical YO-CPT WAVs remain untouched at 24 kHz;
    # MeloTTS loader resamples them on demand.
    #
    "sampling_rate":
        44100,

    "n_speakers":
        1,

    "spk2id": {
        SPEAKER: 0,
    },

    "cleaned_text":
        True,

    "disable_bert":
        True,

    "min_text_len":
        1,

    "max_text_len":
        320,
})


cfg["model_dir"] = str(
    EXPERIMENT.resolve()
)

cfg["pretrain_G"] = str(
    PRETRAIN_G.resolve()
)

cfg["pretrain_D"] = None
cfg["pretrain_dur"] = None
cfg["port"] = None


cfg["project_notes"] = {
    "experiment":
        "C2-F30",

    "stage":
        "Russian foundation",

    "foundation_start":
        "official MeloTTS English-v3 partial transfer",

    "source_checkpoint":
        str(ENGLISH_V3.resolve()),

    "source_checkpoint_sha256":
        source_sha,

    "language":
        "RU",

    "ru_language_id":
        ru_language_id,

    "frontend":
        "espeak-ng IPA",

    "bert_mode":
        "disabled/zero features",

    "precision":
        "FP32",

    "conditioning":
        "single synthetic foundation speaker",

    "speaker":
        SPEAKER,

    "physical_corpus":
        "YO-CPT C2-F30; physically multi-speaker",

    "physical_wav_sampling_rate":
        24000,

    "model_sampling_rate":
        44100,

    "sampling_rate_policy":
        (
            "Preserve English-v3 44.1 kHz model/hop/"
            "upsampler geometry; 24 kHz source WAVs are "
            "resampled on demand by the loader and are "
            "not modified on disk."
        ),

    "spec_cache_policy":
        (
            "Permanent full F30 .spec.pt cache forbidden; "
            "must be disabled before training."
        ),

    "train_rows":
        EXPECTED_TRAIN,

    "val_rows":
        EXPECTED_VAL,
}


OUT.mkdir(
    parents=True,
    exist_ok=False,
)

CONFIG.write_text(
    json.dumps(
        cfg,
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== C2-F30 FOUNDATION PRESTART CONFIG V1 ====="
)

print(
    "train rows:",
    len(train_rows),
)

print(
    "val rows:",
    len(val_rows),
)

print(
    "batch_size:",
    train["batch_size"],
)

print(
    "learning_rate:",
    train["learning_rate"],
)

print(
    "fp16_run:",
    train["fp16_run"],
)

print(
    "disable_bert:",
    data["disable_bert"],
)

print(
    "n_speakers:",
    data["n_speakers"],
)

print(
    "spk2id:",
    data["spk2id"],
)

print(
    "physical WAV rate:",
    24000,
)

print(
    "model sampling rate:",
    data["sampling_rate"],
)

print(
    "RU language id:",
    ru_language_id,
)

print()
print(
    "English-v3 SHA256:",
    source_sha,
)

print(
    "config:",
    CONFIG,
)

print(
    "pretrain target:",
    PRETRAIN_G,
)

print(
    "transfer report:",
    REPORT,
)

print()
print(
    "C2-F30 FOUNDATION PRESTART CONFIG V1: PASS"
)
