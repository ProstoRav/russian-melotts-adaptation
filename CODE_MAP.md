# Code map

This repository contains the final code path used for the diploma project.
Historical experiments, failed runs, benchmark probes and superseded script
versions are intentionally excluded.

## Preprocessing — YO-CPT / spk_298

| File | Purpose |
|---|---|
| `src/preprocessing/yocpt/select_yocpt_c2_f30_manifest_v1.py` | Selects the final YO-CPT candidate manifest. |
| `src/preprocessing/yocpt/build_yocpt_f30_clean_pool_v1.py` | Builds the clean candidate pool. |
| `src/preprocessing/yocpt/audit_yocpt_c2_f30_waveform_qc_v1.py` | Physical waveform quality audit. |
| `src/preprocessing/yocpt/audit_yocpt_c2_f30_physical_duration_v1.py` | Duration-level physical QC. |
| `src/preprocessing/yocpt/summarize_c2_f30_physical_qc_gate_v1.py` | Summarizes the final physical QC gate. |
| `src/preprocessing/yocpt/freeze_c2_f30_physical_v1.py` | Freezes the accepted physical dataset. |
| `src/preprocessing/yocpt/freeze_c2_f30_train_val_split_v1.py` | Freezes the train/validation split. |
| `src/preprocessing/yocpt/freeze_c2_f30_melotts_raw_v1.py` | Creates the raw MeloTTS representation. |
| `src/preprocessing/yocpt/freeze_c2_f30_melotts_no_bert_v1.py` | Creates the final BERT-off MeloTTS representation. |
| `src/preprocessing/yocpt/freeze_c2_f30_speaker_conditioning_v1.py` | Freezes speaker-conditioning metadata. |
| `src/preprocessing/yocpt/prepare_c2_f30_foundation_prestart_v1.py` | Prepares the final Stage-1 training state. |

## Preprocessing — Dialogs

| File | Purpose |
|---|---|
| `src/preprocessing/dialogs_np_source_stress_g2p_v1.py` | Stress/G2P processing for the selected neutral/positive subset. |
| `src/preprocessing/dialogs/build_dialogs_np_melotts_lists_v1.py` | Builds the final MeloTTS train/validation/test lists. |
| `src/preprocessing/dialogs/build_dialogs_np_e27_prestart_v1.py` | Initializes the Dialogs stage from chosen YO-CPT E27. |

## Training

| File | Purpose |
|---|---|
| `src/training/melotts_train_variable_bucket_v1.py` | Final variable-bucket Dialogs training entrypoint. |

## Evaluation

| File | Purpose |
|---|---|
| `src/evaluation/render_tts_trajectory_eval_v2.py` | Deterministic synthesis of the final checkpoint trajectories. |
| `src/evaluation/evaluate_tts_trajectory_metrics_v2.py` | ASR and acoustic trajectory evaluation. |
| `src/evaluation/evaluate_dialogs_matched_style_defense_v1.py` | Controlled neutral/positive matched-text evaluation. |
| `src/evaluation/audit_dialogs_actual_speakers_v1.py` | Dialogs actor-distribution audit. |
| `src/evaluation/prepare_speaker_embedding_backend_wavlm_v1.py` | Pins and validates the WavLM speaker backend. |
| `src/evaluation/calibrate_dialogs_actor_wavlm_v1.py` | Calibrates the actor-centroid diagnostic. |
| `src/evaluation/evaluate_speaker_identity_v1.py` | YO-CPT speaker preservation and Dialogs voice consistency. |

## Reporting

| File | Purpose |
|---|---|
| `src/reporting/build_tts_publication_boxplots_v3.py` | Final CER/WER/acoustic publication plots. |
| `src/reporting/build_speaker_identity_publication_plots_v1.py` | Final speaker/voice identity plots. |
| `src/reporting/build_final_evaluation_package_v1.py` | Builds the frozen final evaluation package and bootstrap summaries. |

## Frozen configs

- `configs/yocpt_e27/config.frozen.json`
- `configs/dialogs_e38/config.frozen.json`

## MeloTTS source adaptation

Upstream repository:

`https://github.com/myshell-ai/MeloTTS.git`

Pinned upstream commit:

`209145371cff8fc3bd60d7be902ea69cbdb7965a`

The final MeloTTS source state is represented by:

`vendor_melotts/overlay/`

Modified upstream files:

- `melo/api.py`
- `melo/commons.py`
- `melo/data_utils.py`
- `melo/models.py`
- `melo/monotonic_align/__init__.py`
- `melo/split_utils.py`
- `melo/text/__init__.py`
- `melo/text/cleaner.py`
- `melo/text/symbols.py`
- `melo/train.py`
- `melo/utils.py`

New Russian frontend files:

- `melo/RU_symbol.txt`
- `melo/text/russian.py`
- `melo/text/russian_bert.py`

The final production configuration is BERT-off. `russian_bert.py` remains included
because it is part of the implemented Russian frontend/import structure.

## Machine-specific paths

Some final scripts/configs retain the exact local path defaults used during the
project (for example `/home/nuc/...`). They are preserved intentionally because this
repository is a defense snapshot of the actual executed code. Adapt these paths to
your local environment before running the pipeline on another machine.

## Final result package

`results/final_evaluation_package_v1/`

contains the compact final result tables and eight primary defense plots.

Datasets, checkpoints, generated WAV files, caches and historical experiment logs
are intentionally not distributed in this repository.
