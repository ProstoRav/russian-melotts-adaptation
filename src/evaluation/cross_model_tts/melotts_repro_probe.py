import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


melo_root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
checkpoint_path = Path(sys.argv[3]).resolve()
sid = int(sys.argv[4])
prompts_path = Path(sys.argv[5]).resolve()
output_path = Path(sys.argv[6]).resolve()

sys.path.insert(0, str(melo_root))

from melo.api import TTS


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

row = next(x for x in rows if x["id"] == "S05")

seed_all(52000)

tts = TTS(
    language="RU",
    device="cpu",
    use_hf=False,
    config_path=str(config_path),
    ckpt_path=str(checkpoint_path),
)

sr = int(tts.hps.data.sampling_rate)
assert sr == 44100


# Same generic warm-up as official benchmark.
seed_all(51999)

tts.tts_to_file(
    "Это отдельная техническая фраза для прогрева модели "
    "перед измерением производительности.",
    sid,
    output_path=None,
    sdp_ratio=0.0,
    noise_scale=0.667,
    noise_scale_w=0.8,
    speed=1.0,
    quiet=True,
    style_id=0,
)


# S05 canonical stressed.
seed_all(52005)

wav = tts.tts_to_file(
    row["text_stressed"],
    sid,
    output_path=None,
    sdp_ratio=0.0,
    noise_scale=0.667,
    noise_scale_w=0.8,
    speed=1.0,
    quiet=True,
    style_id=0,
)

wav = np.asarray(
    wav,
    dtype=np.float32,
).reshape(-1)

assert len(wav) > 0
assert np.isfinite(wav).all()

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sf.write(
    output_path,
    wav,
    sr,
    subtype="PCM_16",
)

print("sid =", sid)
print("samples =", len(wav))
print("seconds =", len(wav) / sr)
print("sha256 =", sha256(output_path))
