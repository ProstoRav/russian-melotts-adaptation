import gc
import hashlib
import inspect
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import utmosv2


dst = Path(sys.argv[1]).resolve()
weights_dir = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

raw_path = output_dir / "fold_scores.jsonl"
final_path = output_dir / "per_prompt.jsonl"
summary_path = output_dir / "system_summary.json"


SYSTEMS = [
    "melotts_yocpt_e27",
    "melotts_dialogs_e38_neutral",
    "melotts_dialogs_e38_positive",
    "dialogs_ru_vits2",
    "silero_v5_5_ru",
    "xtts_v2_spk298",
    "piper_ru_irina_medium",
]

FOLDS = [0, 1, 2, 3, 4]

NUM_REPETITIONS = 5
PREDICT_DATASET = "sarulab"
DEVICE = "cuda:0"

PROMPT_SEED_BASE = 62000


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def find_wav(system_id, prompt_id):
    root = dst / "renders" / system_id

    matches = sorted(
        root.glob(f"{prompt_id}*.wav")
    )

    assert len(matches) == 1, (
        system_id,
        prompt_id,
        matches,
    )

    return matches[0]


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values),
    }


assert torch.cuda.is_available()

print(
    "GPU =",
    torch.cuda.get_device_name(0),
)

print(
    "torch =",
    torch.__version__,
)

torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.use_deterministic_algorithms(
    True,
    warn_only=True,
)


# ------------------------------------------------------------
# Frozen 7 x 9 canonical input matrix.
# ------------------------------------------------------------

matrix = []

for system_id in SYSTEMS:
    for prompt_num in range(1, 10):

        prompt_id = f"S{prompt_num:02d}"

        wav = find_wav(
            system_id,
            prompt_id,
        )

        matrix.append({
            "system_id":
                system_id,

            "prompt_id":
                prompt_id,

            "prompt_num":
                prompt_num,

            "wav_path":
                wav,

            "wav_sha256":
                sha256(wav),
        })

assert len(matrix) == 63

print("input_wavs =", len(matrix))
print("systems =", len(SYSTEMS))
print("folds =", FOLDS)
print(
    "num_repetitions_per_fold =",
    NUM_REPETITIONS,
)
print(
    "effective_predictions_per_wav =",
    len(FOLDS) * NUM_REPETITIONS,
)


# ------------------------------------------------------------
# Resume existing completed fold rows.
# ------------------------------------------------------------

completed = {}

if raw_path.exists():

    for line in raw_path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        row = json.loads(line)

        key = (
            row["fold"],
            row["system_id"],
            row["prompt_id"],
        )

        completed[key] = row


print(
    "resume_completed =",
    len(completed),
)


# ------------------------------------------------------------
# One resident model per fold.
# ------------------------------------------------------------

for fold in FOLDS:

    ckpt = (
        weights_dir /
        f"fold{fold}_s42_best_model.pth"
    )

    assert ckpt.is_file(), ckpt

    ckpt_sha = sha256(ckpt)

    missing = [
        x
        for x in matrix
        if (
            fold,
            x["system_id"],
            x["prompt_id"],
        ) not in completed
    ]

    if not missing:
        print()
        print(
            f"FOLD {fold}: "
            "already complete"
        )
        continue

    print()
    print("=" * 76)
    print(
        f"LOAD FOLD {fold} | "
        f"remaining={len(missing)}"
    )
    print("=" * 76)

    seed_all(63000 + fold)

    torch.cuda.empty_cache()

    t0 = time.perf_counter()

    model = utmosv2.create_model(
        config="fusion_stage3",
        pretrained=True,
        fold=fold,
        seed=42,
        checkpoint_path=str(ckpt),
        device=DEVICE,
    )

    torch.cuda.synchronize()

    load_sec = (
        time.perf_counter() - t0
    )

    sig = inspect.signature(
        model.predict
    )

    print(
        "checkpoint_sha256 =",
        ckpt_sha,
    )

    print(
        "model_load_seconds =",
        load_sec,
    )

    print(
        "predict_signature =",
        sig,
    )


    for seq, item in enumerate(
        missing,
        start=1,
    ):

        system_id = item["system_id"]
        prompt_id = item["prompt_id"]
        prompt_num = item["prompt_num"]
        wav = item["wav_path"]

        # Same semantic prompt gets the same
        # stochastic TTA seed in every system
        # and every fold.
        eval_seed = (
            PROMPT_SEED_BASE
            + prompt_num
        )

        seed_all(eval_seed)

        kwargs = {
            "input_path":
                str(wav),

            "predict_dataset":
                PREDICT_DATASET,

            "device":
                DEVICE,

            "num_workers":
                0,

            "num_repetitions":
                NUM_REPETITIONS,

            "verbose":
                False,
        }

        # Keep compatibility with exact
        # pinned API.
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
        }

        torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.inference_mode():
            value = model.predict(
                **kwargs
            )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - t0
        )

        score = float(
            np.asarray(
                value
            ).reshape(-1)[0]
        )

        assert np.isfinite(score)

        row = {
            "metric":
                "UTMOSv2_predicted_MOS",

            "system_id":
                system_id,

            "prompt_id":
                prompt_id,

            "fold":
                fold,

            "fold_training_seed":
                42,

            "evaluation_seed":
                eval_seed,

            "num_repetitions":
                NUM_REPETITIONS,

            "predict_dataset":
                PREDICT_DATASET,

            "score":
                score,

            "prediction_seconds":
                elapsed,

            "wav_path":
                str(wav),

            "wav_sha256":
                item["wav_sha256"],

            "checkpoint_path":
                str(ckpt),

            "checkpoint_sha256":
                ckpt_sha,

            "device":
                DEVICE,

            "torch_version":
                torch.__version__,

            "gpu":
                torch.cuda.get_device_name(0),
        }

        # Flush one result at a time,
        # making the run resumable.
        with raw_path.open(
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

        completed[
            (
                fold,
                system_id,
                prompt_id,
            )
        ] = row

        print(
            f"fold={fold} "
            f"[{seq:02d}/"
            f"{len(missing):02d}] "
            f"{system_id} "
            f"{prompt_id} "
            f"MOS={score:.6f} "
            f"time={elapsed:.3f}s",
            flush=True,
        )


    del model

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ------------------------------------------------------------
# Integrity.
# 5 folds x 7 systems x 9 prompts = 315.
# ------------------------------------------------------------

rows = list(
    completed.values()
)

assert len(rows) == 315, len(rows)

for fold in FOLDS:

    n = sum(
        x["fold"] == fold
        for x in rows
    )

    assert n == 63, (
        fold,
        n,
    )


# ------------------------------------------------------------
# Final per-WAV score =
# arithmetic mean over 5 fold predictions.
#
# Each fold prediction internally averages
# 5 stochastic inference repetitions.
# ------------------------------------------------------------

final_rows = []


for system_id in SYSTEMS:

    for prompt_num in range(1, 10):

        prompt_id = (
            f"S{prompt_num:02d}"
        )

        selected = sorted(
            [
                x
                for x in rows
                if (
                    x["system_id"]
                    == system_id
                    and
                    x["prompt_id"]
                    == prompt_id
                )
            ],
            key=lambda x: x["fold"],
        )

        assert len(selected) == 5

        assert [
            x["fold"]
            for x in selected
        ] == FOLDS

        fold_scores = [
            x["score"]
            for x in selected
        ]

        final_rows.append({
            "metric":
                "UTMOSv2_predicted_MOS",

            "system_id":
                system_id,

            "prompt_id":
                prompt_id,

            "score":
                statistics.mean(
                    fold_scores
                ),

            "fold_scores":
                fold_scores,

            "fold_stdev":
                statistics.stdev(
                    fold_scores
                ),

            "num_folds":
                5,

            "num_repetitions_per_fold":
                5,

            "effective_predictions":
                25,

            "predict_dataset":
                PREDICT_DATASET,

            "evaluation_seed":
                PROMPT_SEED_BASE
                + prompt_num,

            "wav_path":
                selected[0][
                    "wav_path"
                ],

            "wav_sha256":
                selected[0][
                    "wav_sha256"
                ],
        })


assert len(final_rows) == 63


with final_path.open(
    "w",
    encoding="utf-8",
) as f:

    for row in final_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
        )


# ------------------------------------------------------------
# System summaries across the 9 semantic prompts.
# ------------------------------------------------------------

summary = {
    "metric":
        "UTMOSv2_predicted_MOS",

    "interpretation":
        (
            "automatic MOS prediction; "
            "not human MOS"
        ),

    "config":
        "fusion_stage3",

    "folds":
        FOLDS,

    "num_repetitions_per_fold":
        NUM_REPETITIONS,

    "effective_predictions_per_wav":
        25,

    "predict_dataset":
        PREDICT_DATASET,

    "device":
        DEVICE,

    "gpu":
        torch.cuda.get_device_name(0),

    "torch_version":
        torch.__version__,

    "systems":
        {},
}


for system_id in SYSTEMS:

    values = [
        x["score"]
        for x in final_rows
        if (
            x["system_id"]
            == system_id
        )
    ]

    assert len(values) == 9

    summary["systems"][
        system_id
    ] = {
        "n_prompts":
            9,

        **stats(values),
    }


summary[
    "fold_scores_sha256"
] = sha256(raw_path)

summary[
    "per_prompt_sha256"
] = sha256(final_path)


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
print("=" * 76)
print("FINAL SYSTEM UTMOS")
print("=" * 76)

for system_id in SYSTEMS:

    s = summary[
        "systems"
    ][system_id]

    print(
        f"{system_id:36s} "
        f"mean={s['mean']:.6f} "
        f"median={s['median']:.6f} "
        f"sd={s['stdev']:.6f}"
    )


print()
print(
    "fold_scores =",
    raw_path,
)

print(
    "per_prompt =",
    final_path,
)

print(
    "summary =",
    summary_path,
)

print()
print(
    "UTMOSV2 GPU FULL 5x5 "
    "EVALUATION: PASS"
)
