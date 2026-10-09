import hashlib
import json
import random
import sys
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
seed = int(sys.argv[6])
output = Path(sys.argv[7]).resolve()

assert mode in {"nonstream", "stream"}


def seed_all(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


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

row = next(x for x in rows if x["id"] == "S05")

config = XttsConfig()
config.load_json(str(model_dir / "config.json"))

model = Xtts(config)

model.load_checkpoint(
    config,
    checkpoint_dir=str(model_dir),
    eval=True,
    use_deepspeed=False,
)

model.cpu()
model.eval()

gpt_cond_latent = torch.from_numpy(
    np.load(gpt_path)
).float()

speaker_embedding = torch.from_numpy(
    np.load(speaker_path)
).float()

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

warm_text = (
    "Это отдельная техническая фраза для прогрева модели "
    "перед измерением производительности."
)

# ------------------------------------------------------------
# Mode-specific generic warm-up.
# ------------------------------------------------------------

seed_all(51999)

with torch.inference_mode():
    if mode == "nonstream":
        model.inference(
            warm_text,
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            **kwargs,
        )

    else:
        for _ in model.inference_stream(
            warm_text,
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=20,
            overlap_wav_len=1024,
            **kwargs,
        ):
            pass


# ------------------------------------------------------------
# S05.
# ------------------------------------------------------------

seed_all(seed)

with torch.inference_mode():
    if mode == "nonstream":
        result = model.inference(
            row["text_plain"],
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            **kwargs,
        )

        wav = np.asarray(
            result["wav"],
            dtype=np.float32,
        ).reshape(-1)

    else:
        chunks = []

        for chunk in model.inference_stream(
            row["text_plain"],
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=20,
            overlap_wav_len=1024,
            **kwargs,
        ):
            chunks.append(
                chunk.detach()
                .cpu()
                .float()
                .reshape(-1)
                .numpy()
            )

        assert chunks
        wav = np.concatenate(chunks)


assert wav.ndim == 1
assert len(wav) > 0
assert np.isfinite(wav).all()

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sf.write(
    output,
    wav,
    sr,
    subtype="PCM_16",
)

print("mode =", mode)
print("seed =", seed)
print("samples =", len(wav))
print("audio_seconds =", len(wav) / sr)
print("sha256 =", sha256(output))
print("output =", output)
