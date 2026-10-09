import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import distillmos


dst = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()

output_dir.mkdir(
    parents=True,
    exist_ok=True,
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

DEVICE = "cuda"


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


def stats(values):
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
            statistics.stdev(values),
    }


assert torch.cuda.is_available()
assert "+rocm" in torch.__version__

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

print(
    "GPU =",
    torch.cuda.get_device_name(0),
)

print(
    "torch =",
    torch.__version__,
)


# ------------------------------------------------------------
# Frozen canonical 7 x 9 matrix.
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

        # No silent downmixing:
        # canonical benchmark should be mono.
        assert info.channels == 1, (
            wav,
            info.channels,
        )

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

            "source_sr":
                info.samplerate,

            "source_frames":
                info.frames,

            "source_duration_seconds":
                info.duration,
        })


assert len(matrix) == 63

print("input_wavs =", len(matrix))


# ------------------------------------------------------------
# Model.
# ------------------------------------------------------------

model = (
    distillmos
    .ConvTransformerSQAModel()
)

model.eval()
model = model.to(DEVICE)

print(
    "model_params =",
    sum(
        p.numel()
        for p in model.parameters()
    ),
)


# ------------------------------------------------------------
# Decoder + deterministic resampling.
# ------------------------------------------------------------

resamplers = {}


def load16k(path):
    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )

    assert audio.shape[1] == 1
    assert audio.shape[0] > 0
    assert np.isfinite(audio).all()

    audio = audio[:, 0]

    x = torch.from_numpy(
        np.ascontiguousarray(audio)
    ).unsqueeze(0)

    if sr != 16000:

        if sr not in resamplers:
            resamplers[sr] = (
                torchaudio
                .transforms
                .Resample(
                    orig_freq=sr,
                    new_freq=16000,
                )
            )

        x = resamplers[sr](x)

    assert x.ndim == 2
    assert x.shape[0] == 1
    assert x.shape[-1] > 0
    assert torch.isfinite(x).all()

    return x


# ------------------------------------------------------------
# Warmup excluded from official rows.
# Use first canonical file only as computational warmup;
# its official score is recomputed afterward.
# ------------------------------------------------------------

warm_x = load16k(
    matrix[0]["wav_path"]
).to(DEVICE)

torch.cuda.synchronize()
warm_t0 = time.perf_counter()

with torch.inference_mode():
    warm_score = model(warm_x)

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
    "warmup_score =",
    float(
        warm_score
        .reshape(-1)[0]
        .detach()
        .cpu()
    ),
)

del warm_x, warm_score


# ------------------------------------------------------------
# Resume.
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
# Official 63 evaluations.
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
        continue

    x = load16k(
        item["wav_path"]
    )

    samples_16k = int(
        x.shape[-1]
    )

    x = x.to(
        DEVICE,
        non_blocking=False,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        pred = model(x)

    torch.cuda.synchronize()

    prediction_seconds = (
        time.perf_counter()
        - t0
    )

    score = float(
        pred
        .reshape(-1)[0]
        .detach()
        .cpu()
    )

    assert np.isfinite(score)

    # Distill-MOS output contract.
    assert 1.0 <= score <= 5.0, (
        key,
        score,
    )

    row = {
        "metric":
            "DistillMOS_predicted_MOS",

        "system_id":
            item["system_id"],

        "prompt_id":
            item["prompt_id"],

        "score":
            score,

        "wav_path":
            str(item["wav_path"]),

        "wav_sha256":
            item["wav_sha256"],

        "source_sample_rate_hz":
            item["source_sr"],

        "source_frames":
            item["source_frames"],

        "source_duration_seconds":
            item[
                "source_duration_seconds"
            ],

        "evaluation_sample_rate_hz":
            16000,

        "samples_16k":
            samples_16k,

        "duration_16k_seconds":
            samples_16k / 16000.0,

        "prediction_seconds":
            prediction_seconds,

        "device":
            DEVICE,

        "gpu":
            torch.cuda.get_device_name(0),

        "torch_version":
            torch.__version__,

        "model_variant":
            7,
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

    completed[key] = row

    print(
        f"[{seq:02d}/63] "
        f"{item['system_id']} "
        f"{item['prompt_id']} "
        f"MOS={score:.6f} "
        f"time={prediction_seconds:.4f}s",
        flush=True,
    )


# ------------------------------------------------------------
# Integrity.
# ------------------------------------------------------------

rows = list(
    completed.values()
)

assert len(rows) == 63, len(rows)

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

assert (
    actual_keys
    == expected_keys
)


# ------------------------------------------------------------
# Deterministic ordering.
# ------------------------------------------------------------

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

# Rewrite once into canonical ordering.
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
# ------------------------------------------------------------

summary = {
    "metric":
        "DistillMOS_predicted_MOS",

    "interpretation":
        (
            "automatic MOS prediction; "
            "not human MOS"
        ),

    "model":
        "Microsoft Distill-MOS",

    "model_variant":
        7,

    "evaluation_sample_rate_hz":
        16000,

    "wav_decoder":
        "soundfile",

    "resampler":
        "torchaudio.transforms.Resample",

    "device":
        DEVICE,

    "gpu":
        torch.cuda.get_device_name(0),

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

    values = [
        x["score"]
        for x in rows
        if (
            x["system_id"]
            == system_id
        )
    ]

    assert len(values) == 9

    summary[
        "systems"
    ][system_id] = {
        "n_prompts": 9,
        **stats(values),
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
print("=" * 76)
print("FINAL SYSTEM DISTILL-MOS")
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
    "per_prompt =",
    per_prompt_path,
)

print(
    "summary =",
    summary_path,
)

print()
print(
    "DISTILL-MOS GPU FULL "
    "EVALUATION: PASS"
)
