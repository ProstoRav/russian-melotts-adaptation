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
out_nonstream = Path(sys.argv[5]).resolve()
out_stream = Path(sys.argv[6]).resolve()


torch.set_num_threads(2)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


rows = [
    json.loads(x)
    for x in prompts_path.read_text(
        encoding="utf-8"
    ).splitlines()
]

row = next(x for x in rows if x["id"] == "S05")


config = XttsConfig()
config.load_json(str(model_dir / "config.json"))

print("===== INFERENCE CONFIG =====")
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


# ------------------------------------------------------------
# Generic warm-up, excluded.
# ------------------------------------------------------------

warm_text = (
    "Это отдельная техническая фраза для прогрева модели "
    "перед измерением производительности."
)

seed_all(51999)

print()
print("===== GENERIC NON-STREAM WARM-UP =====")

t0 = time.perf_counter()

with torch.inference_mode():
    warm = model.inference(
        warm_text,
        "ru",
        gpt_cond_latent,
        speaker_embedding,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
    )

warm_wall = time.perf_counter() - t0
warm_audio = np.asarray(warm["wav"], dtype=np.float32)

print("warm_wall_seconds =", warm_wall)
print(
    "warm_audio_seconds =",
    len(warm_audio) / sr,
)
print(
    "warm_RTF =",
    warm_wall / (len(warm_audio) / sr),
)


# ------------------------------------------------------------
# S05 non-streaming.
# ------------------------------------------------------------

seed_all(52005)

print()
print("===== S05 NON-STREAMING =====")
print("text =", row["text_plain"])

t0 = time.perf_counter()

with torch.inference_mode():
    out = model.inference(
        row["text_plain"],
        "ru",
        gpt_cond_latent,
        speaker_embedding,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
    )

nonstream_wall = time.perf_counter() - t0

nonstream_wav = np.asarray(
    out["wav"],
    dtype=np.float32,
).squeeze()

assert nonstream_wav.ndim == 1
assert len(nonstream_wav) > 0
assert np.isfinite(nonstream_wav).all()

out_nonstream.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sf.write(
    out_nonstream,
    nonstream_wav,
    sr,
    subtype="PCM_16",
)

nonstream_audio_sec = (
    len(nonstream_wav) / sr
)

print("wall_seconds =", nonstream_wall)
print("audio_seconds =", nonstream_audio_sec)
print(
    "RTF =",
    nonstream_wall / nonstream_audio_sec,
)
print("wav_sha256 =", sha256(out_nonstream))


# ------------------------------------------------------------
# S05 native streaming.
#
# Reset seed so this is an independent S05 realization under
# exactly the same RNG seed, not continuation of non-stream RNG.
# ------------------------------------------------------------

seed_all(52005)

print()
print("===== S05 NATIVE STREAMING =====")

start = time.perf_counter()

stream_chunks = []
ttfa = None

with torch.inference_mode():
    generator = model.inference_stream(
        row["text_plain"],
        "ru",
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=20,
        overlap_wav_len=1024,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
    )

    for i, chunk in enumerate(generator, start=1):
        now = time.perf_counter()

        if ttfa is None:
            ttfa = now - start

        chunk = (
            chunk.detach()
            .cpu()
            .float()
            .reshape(-1)
            .numpy()
        )

        stream_chunks.append(chunk)

        print(
            f"chunk={i} "
            f"samples={len(chunk)} "
            f"audio_sec={len(chunk)/sr:.6f}"
        )

stream_wall = time.perf_counter() - start

assert stream_chunks
assert ttfa is not None

stream_wav = np.concatenate(stream_chunks)

assert len(stream_wav) > 0
assert np.isfinite(stream_wav).all()

sf.write(
    out_stream,
    stream_wav,
    sr,
    subtype="PCM_16",
)

stream_audio_sec = len(stream_wav) / sr

print()
print("chunks =", len(stream_chunks))
print("TTFA_seconds =", ttfa)
print("stream_wall_seconds =", stream_wall)
print("stream_audio_seconds =", stream_audio_sec)
print(
    "stream_RTF =",
    stream_wall / stream_audio_sec,
)
print("wav_sha256 =", sha256(out_stream))

print()
print("XTTS S05 SMOKE: PASS")
