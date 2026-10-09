import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


model_dir = Path(sys.argv[1]).resolve()
prompts_path = Path(sys.argv[2]).resolve()
gpt_path = Path(sys.argv[3]).resolve()
speaker_path = Path(sys.argv[4]).resolve()
mode = sys.argv[5]
out_dir = Path(sys.argv[6]).resolve()
metrics_path = Path(sys.argv[7]).resolve()

assert mode in {"nonstream", "stream"}

SEED_BASE = 52000
WARM_SEED = 51999


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


rows = [
    json.loads(x)
    for x in prompts_path.read_text(
        encoding="utf-8"
    ).splitlines()
]

assert [r["id"] for r in rows] == [
    f"S{i:02d}" for i in range(1, 10)
]


config = XttsConfig()
config.load_json(str(model_dir / "config.json"))

print("===== CONFIG =====")
print("mode =", mode)
print("temperature =", config.temperature)
print("length_penalty =", config.length_penalty)
print("repetition_penalty =", config.repetition_penalty)
print("top_k =", config.top_k)
print("top_p =", config.top_p)
print(
    "output_sample_rate =",
    config.model_args.output_sample_rate,
)


print()
print("===== MODEL LOAD =====")

t0 = time.perf_counter()

model = Xtts(config)

model.load_checkpoint(
    config,
    checkpoint_dir=str(model_dir),
    eval=True,
    use_deepspeed=False,
)

model.cpu()
model.eval()

load_seconds = time.perf_counter() - t0

print("load_seconds =", load_seconds)
print("device =", model.device)

assert str(model.device) == "cpu"


gpt_cond_latent = torch.from_numpy(
    np.load(gpt_path)
).to(
    device="cpu",
    dtype=torch.float32,
)

speaker_embedding = torch.from_numpy(
    np.load(speaker_path)
).to(
    device="cpu",
    dtype=torch.float32,
)

assert tuple(gpt_cond_latent.shape) == (1, 32, 1024)
assert tuple(speaker_embedding.shape) == (1, 512, 1)

sr = int(config.model_args.output_sample_rate)
assert sr == 24000


kwargs = dict(
    temperature=config.temperature,
    length_penalty=config.length_penalty,
    repetition_penalty=config.repetition_penalty,
    top_k=config.top_k,
    top_p=config.top_p,
    do_sample=True,
    speed=1.0,
    enable_text_splitting=False,
)


# ------------------------------------------------------------
# Mode-specific generic warm-up.
# Excluded from reported performance.
# ------------------------------------------------------------

warm_text = (
    "Это отдельная техническая фраза для прогрева модели "
    "перед измерением производительности."
)

seed_all(WARM_SEED)

print()
print("===== GENERIC WARM-UP (EXCLUDED) =====")

warm_start = time.perf_counter()

with torch.inference_mode():
    if mode == "nonstream":
        warm_result = model.inference(
            warm_text,
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            **kwargs,
        )

        warm_wav = np.asarray(
            warm_result["wav"],
            dtype=np.float32,
        ).reshape(-1)

    else:
        warm_chunks = []

        for chunk in model.inference_stream(
            warm_text,
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=20,
            overlap_wav_len=1024,
            **kwargs,
        ):
            warm_chunks.append(
                chunk.detach()
                .cpu()
                .float()
                .reshape(-1)
                .numpy()
            )

        assert warm_chunks
        warm_wav = np.concatenate(warm_chunks)

warm_wall = time.perf_counter() - warm_start
warm_audio_sec = len(warm_wav) / sr

print("warm_wall_seconds =", warm_wall)
print("warm_audio_seconds =", warm_audio_sec)
print("warm_RTF =", warm_wall / warm_audio_sec)


# ------------------------------------------------------------
# Official S01-S09.
# ------------------------------------------------------------

out_dir.mkdir(parents=True, exist_ok=True)
metrics_path.parent.mkdir(parents=True, exist_ok=True)

results = []

print()
print("===== OFFICIAL BENCHMARK =====")

for sequence_index, row in enumerate(rows, start=1):
    pid = row["id"]
    prompt_num = int(pid[1:])
    seed = SEED_BASE + prompt_num

    seed_all(seed)

    wav_path = out_dir / f"{pid}_xtts.wav"

    if mode == "nonstream":
        start = time.perf_counter()

        with torch.inference_mode():
            result = model.inference(
                row["text_plain"],
                "ru",
                gpt_cond_latent,
                speaker_embedding,
                **kwargs,
            )

        wall = time.perf_counter() - start

        wav = np.asarray(
            result["wav"],
            dtype=np.float32,
        ).reshape(-1)

        assert len(wav) > 0
        assert np.isfinite(wav).all()

        audio_sec = len(wav) / sr
        rtf = wall / audio_sec

        # Disk write outside timing.
        sf.write(
            wav_path,
            wav,
            sr,
            subtype="PCM_16",
        )

        item = {
            "system_id": "xtts_v2_spk298",
            "mode": "nonstream",
            "prompt_id": pid,
            "category": row["category"],
            "sequence_index": sequence_index,
            "seed": seed,

            "input_protocol": "primary_plain",
            "input_text": row["text_plain"],

            "sampling_rate_hz": sr,
            "samples": int(len(wav)),
            "audio_duration_seconds": audio_sec,

            "wall_inference_seconds": wall,
            "RTF": rtf,

            "delivery_mode": "full_waveform",
            "full_waveform_availability_seconds": wall,

            "model_load_in_timing": False,
            "conditioning_in_timing": False,
            "warmup_in_timing": False,
            "disk_write_in_timing": False,

            "cpu_threads": 2,

            "wav_path": str(wav_path),
            "wav_sha256": sha256(wav_path),
        }

        print(
            f"[{sequence_index:02d}/09] {pid} "
            f"seed={seed} "
            f"wall={wall:.6f}s "
            f"audio={audio_sec:.6f}s "
            f"RTF={rtf:.6f}"
        )

    else:
        start = time.perf_counter()

        chunks = []
        ttfa = None

        with torch.inference_mode():
            generator = model.inference_stream(
                row["text_plain"],
                "ru",
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=20,
                overlap_wav_len=1024,
                **kwargs,
            )

            for chunk in generator:
                now = time.perf_counter()

                if ttfa is None:
                    ttfa = now - start

                chunk_np = (
                    chunk.detach()
                    .cpu()
                    .float()
                    .reshape(-1)
                    .numpy()
                )

                chunks.append(chunk_np)

        wall = time.perf_counter() - start

        assert chunks
        assert ttfa is not None

        # Concatenation and disk write are outside timing.
        wav = np.concatenate(chunks)

        assert len(wav) > 0
        assert np.isfinite(wav).all()

        audio_sec = len(wav) / sr
        rtf = wall / audio_sec

        first_chunk_samples = len(chunks[0])
        first_chunk_audio_sec = first_chunk_samples / sr

        sf.write(
            wav_path,
            wav,
            sr,
            subtype="PCM_16",
        )

        item = {
            "system_id": "xtts_v2_spk298",
            "mode": "stream",
            "prompt_id": pid,
            "category": row["category"],
            "sequence_index": sequence_index,
            "seed": seed,

            "input_protocol": "primary_plain",
            "input_text": row["text_plain"],

            "sampling_rate_hz": sr,
            "samples": int(len(wav)),
            "audio_duration_seconds": audio_sec,

            "chunk_count": len(chunks),
            "first_chunk_samples": first_chunk_samples,
            "first_chunk_audio_seconds": first_chunk_audio_sec,

            "TTFA_seconds": ttfa,
            "stream_wall_seconds": wall,
            "stream_RTF": rtf,

            "delivery_mode": "native_xtts_streaming",
            "stream_chunk_size_tokens": 20,
            "overlap_wav_len_samples": 1024,

            "model_load_in_timing": False,
            "conditioning_in_timing": False,
            "warmup_in_timing": False,
            "disk_write_in_timing": False,

            "cpu_threads": 2,

            "wav_path": str(wav_path),
            "wav_sha256": sha256(wav_path),
        }

        print(
            f"[{sequence_index:02d}/09] {pid} "
            f"seed={seed} "
            f"chunks={len(chunks)} "
            f"TTFA={ttfa:.6f}s "
            f"wall={wall:.6f}s "
            f"audio={audio_sec:.6f}s "
            f"RTF={rtf:.6f}"
        )

    results.append(item)


with metrics_path.open("w", encoding="utf-8") as f:
    for item in results:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
            ) + "\n"
        )

assert len(results) == 9

print()
print("XTTS FULL BENCHMARK:", mode, "PASS")
print("metrics =", metrics_path)
