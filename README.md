# Russian MeloTTS adaptation — defense-ready code snapshot

Minimal reproducible code snapshot prepared for diploma defense.

## Implemented pipeline

`English-v3 MeloTTS -> Russian YO-CPT spk_298 -> chosen E27 -> Dialogs neutral/positive -> chosen E38`

The repository contains only the code and compact evidence needed to inspect the methodology:

- final preprocessing/training scripts;
- deterministic evaluation scripts;
- final frozen production configs;
- final result CSVs and main plots;
- a patch describing local MeloTTS source changes;
- exact evaluator/model revisions and checkpoint hashes.

## Deliberately not included

- source datasets or parquet shards;
- WAV/audio files;
- model checkpoints/weights;
- Hugging Face caches;
- training logs;
- failed/obsolete experiments;
- old script versions;
- machine-specific runtime caches.

## Final results

### YO-CPT: E1 -> chosen E27
- WER: 63.67% -> 24.69% (relative reduction 61.22%)
- CER: 32.88% -> 11.82% (relative reduction 64.05%)
- E27 similarity to spk_298 enrollment centroid: 0.98841
- natural validation median against the same centroid: 0.98867

### Dialogs: E1 -> chosen E38
- WER: 29.35% -> 12.44% (relative reduction 57.61%)
- CER: 11.52% -> 4.06% (relative reduction 64.78%)
- E38 positive-neutral duration difference: +1.97%
- E38 positive-neutral RMS difference: -0.80 dB
- E38 SID0<->SID1 external voice cosine: 0.99187

## Layout

```text
src/preprocessing/
src/training/
src/evaluation/
src/reporting/
configs/
results/final_evaluation_package_v1/
environment/
DATA.md
REPRODUCIBILITY.md
CODE_MAP.md
```

## Reproduction scope

Large external data and weights are intentionally excluded. With the datasets and selected checkpoints supplied locally, the final evaluation is rebuilt from the scripts under `src/evaluation/` and `src/reporting/`.

See `CODE_MAP.md`, `DATA.md` and `REPRODUCIBILITY.md`.

## Important terminology

- YO-CPT uses a **validation holdout**, not a test set.
- Dialogs uses train / validation / test.
- Dialogs SID0 and SID1 are style-conditioning labels, not Masha/Sveta identities.
- CER/WER are ASR-dependent and are interpreted together with acoustic, speaker and listening-QC evidence.
- Formal MOS was not performed.

## Licensing

Third-party code, models and datasets keep their original licenses. This repository does not redistribute their weights or audio. Choose a license for original project code separately before making the repository public.

## MeloTTS source adaptation

The project is based on the official MeloTTS repository:

`https://github.com/myshell-ai/MeloTTS.git`

Pinned upstream commit:

`209145371cff8fc3bd60d7be902ea69cbdb7965a`

The final local MeloTTS source changes are distributed as a small source overlay:

`vendor_melotts/overlay/`

To reconstruct the source state used by the project:

```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
git checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a
cp -a /path/to/ru-melotts-defense/vendor_melotts/overlay/. .
```

The overlay contains only the 11 modified upstream source files and 3 new
Russian-frontend files used by the final project.

The final production configuration is BERT-off. `russian_bert.py` is nevertheless
included because it is part of the implemented Russian frontend/import structure.

Some scripts and frozen configs intentionally retain the machine-specific paths used
during the actual project run. Adjust those paths before running the code on another
machine; they are preserved here to keep the defense snapshot faithful to the
executed source/configuration state.

## MeloTTS source adaptation

The project is based on the official MeloTTS repository:

`https://github.com/myshell-ai/MeloTTS.git`

Pinned upstream commit:

`209145371cff8fc3bd60d7be902ea69cbdb7965a`

The final local MeloTTS source changes are distributed as a small source overlay:

`vendor_melotts/overlay/`

To reconstruct the source state used by the project:

```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
git checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a
cp -a /path/to/ru-melotts-defense/vendor_melotts/overlay/. .
```

The overlay contains only the 11 modified upstream source files and 3 new
Russian-frontend files used by the final project.

The final production configuration is BERT-off. `russian_bert.py` is nevertheless
included because it is part of the implemented Russian frontend/import structure.

Some scripts and frozen configs intentionally retain the machine-specific paths used
during the actual project run. Adjust those paths before running the code on another
machine; they are preserved here to keep the defense snapshot faithful to the
executed source/configuration state.
