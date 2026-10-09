import hashlib
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from piper import PiperVoice, SynthesisConfig
from piper.config import PiperConfig


voice_path = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
prompts_path = Path(sys.argv[3]).resolve()
seed = int(sys.argv[4])
output = Path(sys.argv[5]).resolve()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Important: seed BEFORE session construction.
random.seed(seed)
np.random.seed(seed)
ort.set_seed(seed)

with config_path.open("r", encoding="utf-8") as f:
    config_dict = json.load(f)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(voice_path),
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)

voice = PiperVoice(
    session=session,
    config=PiperConfig.from_dict(config_dict),
)

syn_config = SynthesisConfig(
    length_scale=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=True,
)

rows = [
    json.loads(x)
    for x in prompts_path.read_text(
        encoding="utf-8"
    ).splitlines()
]

row = next(x for x in rows if x["id"] == "S01")


# ------------------------------------------------------------
# Same generic warm-up that will be used by final benchmark.
# This deliberately advances the model RNG before S01.
# ------------------------------------------------------------

warm_text = (
    "Это отдельная техническая фраза для прогрева модели "
    "перед измерением производительности."
)

for _ in voice.synthesize(
    warm_text,
    syn_config=syn_config,
):
    pass


# ------------------------------------------------------------
# S01
# ------------------------------------------------------------

chunks = list(
    voice.synthesize(
        row["text_plain"],
        syn_config=syn_config,
    )
)

assert chunks

sr = chunks[0].sample_rate
sample_width = chunks[0].sample_width
channels = chunks[0].sample_channels

output.parent.mkdir(parents=True, exist_ok=True)

with wave.open(str(output), "wb") as wf:
    wf.setframerate(sr)
    wf.setsampwidth(sample_width)
    wf.setnchannels(channels)

    for chunk in chunks:
        wf.writeframes(chunk.audio_int16_bytes)

samples = sum(
    len(c.audio_int16_array)
    for c in chunks
)

print("seed =", seed)
print("samples =", samples)
print("seconds =", samples / sr)
print("sha256 =", sha256(output))
print("output =", output)
