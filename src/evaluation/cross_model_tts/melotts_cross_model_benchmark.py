import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


melo_root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
checkpoint_path = Path(sys.argv[3]).resolve()

system_id = sys.argv[4]
sid = int(sys.argv[5])

prompts_path = Path(sys.argv[6]).resolve()
out_dir = Path(sys.argv[7]).resolve()
perf_dir = Path(sys.argv[8]).resolve()

input_field = sys.argv[9]
protocol_id = sys.argv[10]

assert input_field in {
    "text_stressed",
    "text_plain",
}

sys.path.insert(0, str(melo_root))

from melo.api import TTS


SEED_BASE = 52000
WARM_SEED = 51999

SDP_RATIO = 0.0
NOISE_SCALE = 0.667
NOISE_SCALE_W = 0.8
SPEED = 1.0
STYLE_ID = 0


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values),
    }


torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


rows = [
    json.loads(line)
    for line in prompts_path.read_text(
        encoding="utf-8"
    ).splitlines()
]

assert [x["id"] for x in rows] == [
    f"S{i:02d}" for i in range(1, 10)
]


if input_field == "text_stressed":
    for row in rows:
        assert "\u0301" in row[input_field], row["id"]


print("===== MODEL LOAD =====")
print("system_id =", system_id)
print("protocol_id =", protocol_id)
print("input_field =", input_field)
print("sid =", sid)
print("checkpoint =", checkpoint_path)
print("config =", config_path)

seed_all(52000)

t0 = time.perf_counter()

tts = TTS(
    language="RU",
    device="cpu",
    use_hf=False,
    config_path=str(config_path),
    ckpt_path=str(checkpoint_path),
)

load_seconds = time.perf_counter() - t0

assert str(tts.device) == "cpu"

sr = int(tts.hps.data.sampling_rate)
n_speakers = int(tts.hps.data.n_speakers)

assert 0 <= sid < n_speakers

print("load_seconds =", load_seconds)
print("device =", tts.device)
print("sampling_rate =", sr)
print("n_speakers =", n_speakers)
print("spk2id =", dict(tts.hps.data.spk2id))
print("torch_threads =", torch.get_num_threads())
print(
    "torch_interop_threads =",
    torch.get_num_interop_threads(),
)


# ------------------------------------------------------------
# Generic warm-up. Excluded from reported performance.
# ------------------------------------------------------------

warm_text = (
    "Это отдельная техническая фраза для прогрева модели "
    "перед измерением производительности."
)

seed_all(WARM_SEED)

print()
print("===== GENERIC WARM-UP (EXCLUDED) =====")

t0 = time.perf_counter()

warm = tts.tts_to_file(
    warm_text,
    sid,
    output_path=None,
    sdp_ratio=SDP_RATIO,
    noise_scale=NOISE_SCALE,
    noise_scale_w=NOISE_SCALE_W,
    speed=SPEED,
    quiet=True,
    style_id=STYLE_ID,
)

warm_wall = time.perf_counter() - t0

warm = np.asarray(
    warm,
    dtype=np.float32,
).reshape(-1)

assert len(warm) > 0
assert np.isfinite(warm).all()

warm_audio = len(warm) / sr

print("warm_wall_seconds =", warm_wall)
print("warm_audio_seconds =", warm_audio)
print("warm_RTF =", warm_wall / warm_audio)


# ------------------------------------------------------------
# Official S01-S09.
# ------------------------------------------------------------

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

perf_dir.mkdir(
    parents=True,
    exist_ok=True,
)

results = []

print()
print("===== OFFICIAL BENCHMARK =====")

for index, row in enumerate(rows, start=1):
    pid = row["id"]
    seed = SEED_BASE + int(pid[1:])

    input_text = row[input_field]
    asr_reference = row["text_plain"]

    seed_all(seed)

    start = time.perf_counter()

    wav = tts.tts_to_file(
        input_text,
        sid,
        output_path=None,
        sdp_ratio=SDP_RATIO,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W,
        speed=SPEED,
        quiet=True,
        style_id=STYLE_ID,
    )

    wall = time.perf_counter() - start

    wav = np.asarray(
        wav,
        dtype=np.float32,
    ).reshape(-1)

    assert len(wav) > 0
    assert np.isfinite(wav).all()

    audio_seconds = len(wav) / sr
    rtf = wall / audio_seconds

    wav_path = (
        out_dir /
        f"{pid}_{system_id}.wav"
    )

    # Disk write intentionally outside timing.
    sf.write(
        wav_path,
        wav,
        sr,
        subtype="PCM_16",
    )

    item = {
        "system_id": system_id,
        "protocol_id": protocol_id,
        "prompt_id": pid,
        "category": row["category"],
        "sequence_index": index,
        "seed": seed,

        "sid": sid,
        "style_id": STYLE_ID,

        "input_field": input_field,
        "input_text": input_text,
        "input_text_sha256":
            sha256_text(input_text),

        "canonical_pronunciation_target":
            row["text_stressed"],

        "asr_reference_text":
            asr_reference,
        "asr_reference_sha256":
            sha256_text(asr_reference),

        "sdp_ratio": SDP_RATIO,
        "noise_scale": NOISE_SCALE,
        "noise_scale_w": NOISE_SCALE_W,
        "speed": SPEED,

        "sampling_rate_hz": sr,
        "samples": int(len(wav)),
        "audio_duration_seconds":
            audio_seconds,

        "wall_inference_seconds": wall,
        "RTF": rtf,

        "delivery_mode": "full_waveform",
        "full_waveform_availability_seconds":
            wall,

        "cpu_threads": 2,

        "model_load_in_timing": False,
        "warmup_in_timing": False,
        "disk_write_in_timing": False,

        "wav_path": str(wav_path),
        "wav_sha256":
            sha256_file(wav_path),
    }

    results.append(item)

    print(
        f"[{index:02d}/09] {pid} "
        f"seed={seed} "
        f"wall={wall:.6f}s "
        f"audio={audio_seconds:.6f}s "
        f"RTF={rtf:.6f}"
    )


per_prompt_path = (
    perf_dir / "per_prompt.jsonl"
)

with per_prompt_path.open(
    "w",
    encoding="utf-8",
) as f:
    for item in results:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
        )


rtfs = [x["RTF"] for x in results]

walls = [
    x["full_waveform_availability_seconds"]
    for x in results
]

durations = [
    x["audio_duration_seconds"]
    for x in results
]


summary = {
    "system_id": system_id,
    "protocol_id": protocol_id,
    "input_field": input_field,

    "n_prompts": 9,

    "device": "CPU",
    "cpu_threads": 2,

    "sid": sid,
    "style_id": STYLE_ID,

    "model_load_excluded": True,
    "warmup_excluded": True,
    "disk_write_excluded": True,

    "RTF": stats(rtfs),

    "full_waveform_availability_seconds":
        stats(walls),

    "audio_duration_seconds": {
        **stats(durations),
        "total": sum(durations),
    },

    "checkpoint_sha256":
        sha256_file(checkpoint_path),

    "config_sha256":
        sha256_file(config_path),

    "per_prompt_sha256":
        sha256_file(per_prompt_path),
}


summary_path = (
    perf_dir / "summary.json"
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
print(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )
)

print()
print(
    "MELOTTS BENCHMARK:",
    system_id,
    protocol_id,
    "PASS",
)
