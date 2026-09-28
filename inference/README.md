# Inference utilities

These scripts reproduce the same high-level MeloTTS inference path used by the
frozen trajectory evaluation:

```python
TTS(
    language="RU",
    device="cpu",
    use_hf=False,
    config_path=...,
    ckpt_path=...,
)
```

with:

- `sdp_ratio = 0`
- `noise_scale = 0.667`
- `noise_scale_w = 0.8`
- `speed = 1.0`
- `style_id = 0`

The private checkpoint bundle is expected to contain `CHECKPOINTS.json`,
the two frozen configs and the 11 defense generator checkpoints.

## Prerequisites

1. Clone the defense code repository.
2. Reconstruct the pinned MeloTTS source state using `vendor_melotts/overlay/`.
3. Obtain the private checkpoint bundle.
4. Install the Python/runtime dependencies used by MeloTTS.
5. For closest reproduction, use the pinned project eSpeak NG build.

Example reconstruction of MeloTTS:

```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
git checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a
cp -a /path/to/russian-melotts-adaptation/vendor_melotts/overlay/. .
```

## Single utterance — YO-CPT

```bash
python inference/synthesize.py \
  --checkpoint-root ~/exports/russian-melotts-defense-checkpoints-v1 \
  --melo-root ~/projects/MeloTTS \
  --espeak-build ~/projects/ru_melotts_research/tools/espeak-ng-ru-stress-ab2eab9/build \
  --stage yocpt \
  --epoch 27 \
  --text "Проверочная фраза для синтеза речи." \
  --output yocpt_e27.wav
```

YO-CPT has one target speaker, so the script reads the single SID from the frozen
config automatically.

## Single utterance — Dialogs neutral

```bash
python inference/synthesize.py \
  --checkpoint-root ~/exports/russian-melotts-defense-checkpoints-v1 \
  --melo-root ~/projects/MeloTTS \
  --espeak-build ~/projects/ru_melotts_research/tools/espeak-ng-ru-stress-ab2eab9/build \
  --stage dialogs \
  --epoch 38 \
  --style neutral \
  --text "Проверочная фраза для синтеза речи." \
  --output dialogs_e38_neutral.wav
```

## Single utterance — Dialogs positive

```bash
python inference/synthesize.py \
  --checkpoint-root ~/exports/russian-melotts-defense-checkpoints-v1 \
  --melo-root ~/projects/MeloTTS \
  --espeak-build ~/projects/ru_melotts_research/tools/espeak-ng-ru-stress-ab2eab9/build \
  --stage dialogs \
  --epoch 38 \
  --style positive \
  --text "Проверочная фраза для синтеза речи." \
  --output dialogs_e38_positive.wav
```

`neutral` and `positive` are mapped to the actual SID values from the frozen
Dialogs config. An explicit `--sid` can be supplied if needed.

## Compare every defense epoch

YO-CPT:

```bash
python inference/render_defense_trajectory.py \
  --checkpoint-root ~/exports/russian-melotts-defense-checkpoints-v1 \
  --melo-root ~/projects/MeloTTS \
  --espeak-build ~/projects/ru_melotts_research/tools/espeak-ng-ru-stress-ab2eab9/build \
  --stage yocpt \
  --text "Одна и та же фраза используется для сравнения эпох." \
  --output-dir trajectory_yocpt
```

Dialogs positive:

```bash
python inference/render_defense_trajectory.py \
  --checkpoint-root ~/exports/russian-melotts-defense-checkpoints-v1 \
  --melo-root ~/projects/MeloTTS \
  --espeak-build ~/projects/ru_melotts_research/tools/espeak-ng-ru-stress-ab2eab9/build \
  --stage dialogs \
  --style positive \
  --text "Одна и та же фраза используется для сравнения эпох." \
  --output-dir trajectory_dialogs_positive
```

The trajectory utility uses exactly the same seed for every epoch, making the
comparison easier to interpret.

## Exact controlled SID0/SID1 comparison

For a controlled Dialogs pair, run the same text and same seed once with
`--style neutral` and once with `--style positive`.

For the original frozen defense experiment, prompt-specific seeds were
`52000 + prompt_number`.

## Device

The final objective/defense trajectory evaluation used CPU inference for a fixed
deterministic evaluation domain. Therefore these utilities default to:

```text
--device cpu
```

A user may explicitly request another MeloTTS-supported device, but those outputs
should not be described as byte-identical reproductions of the frozen CPU
evaluation without validating them separately.

## Integrity

`synthesize.py` checks the requested checkpoint against the SHA256 recorded in
`CHECKPOINTS.json` before loading it.
