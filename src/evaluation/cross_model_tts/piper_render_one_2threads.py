import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import onnxruntime as ort

from piper import PiperVoice, SynthesisConfig
from piper.config import PiperConfig


voice_path = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
prompts_path = Path(sys.argv[3]).resolve()
prompt_id = sys.argv[4]
output = Path(sys.argv[5]).resolve()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


rows = [
    json.loads(x)
    for x in prompts_path.read_text(
        encoding="utf-8"
    ).splitlines()
]

row = next(
    x for x in rows
    if x["id"] == prompt_id
)


# ------------------------------------------------------------
# Explicit ONNX Runtime thread contract.
# ------------------------------------------------------------

with config_path.open("r", encoding="utf-8") as f:
    config_dict = json.load(f)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2
session_options.inter_op_num_threads = 1
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(voice_path),
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)

voice = PiperVoice(
    session=session,
    config=PiperConfig.from_dict(config_dict),
)

actual_opts = voice.session.get_session_options()

assert voice.session.get_providers() == [
    "CPUExecutionProvider"
]
assert actual_opts.intra_op_num_threads == 2
assert actual_opts.inter_op_num_threads == 1

print("providers =", voice.session.get_providers())
print(
    "intra_op_num_threads =",
    actual_opts.intra_op_num_threads,
)
print(
    "inter_op_num_threads =",
    actual_opts.inter_op_num_threads,
)
print("execution_mode =", actual_opts.execution_mode)


syn_config = SynthesisConfig(
    length_scale=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=True,
)


# ------------------------------------------------------------
# Render.
# ------------------------------------------------------------

start = time.perf_counter()

chunks = []
ttfa = None

for chunk in voice.synthesize(
    row["text_plain"],
    syn_config=syn_config,
):
    now = time.perf_counter()

    if ttfa is None:
        ttfa = now - start

    chunks.append(chunk)

total_wall = time.perf_counter() - start

assert chunks
assert ttfa is not None

sr = chunks[0].sample_rate
sample_width = chunks[0].sample_width
channels = chunks[0].sample_channels

assert sr == 22050
assert sample_width == 2
assert channels == 1

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with wave.open(str(output), "wb") as wf:
    wf.setframerate(sr)
    wf.setsampwidth(sample_width)
    wf.setnchannels(channels)

    for chunk in chunks:
        wf.writeframes(
            chunk.audio_int16_bytes
        )

samples = sum(
    len(chunk.audio_int16_array)
    for chunk in chunks
)

audio_seconds = samples / float(sr)
rtf = total_wall / audio_seconds

print("prompt =", prompt_id)
print("chunks =", len(chunks))
print("samples =", samples)
print("audio_seconds =", audio_seconds)
print("TTFA_sentence_chunk_seconds =", ttfa)
print("total_wall_seconds =", total_wall)
print("RTF =", rtf)
print("sha256 =", sha256(output))
print("output =", output)
