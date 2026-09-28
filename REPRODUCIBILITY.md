# Reproducibility

## Chosen checkpoints

### YO-CPT
- E27 / step 56645
- G SHA256: `7d03e90f9b94ecca28ab62246078ef3431708ba5392d129d7ddde5a2420cf2cc`

### Dialogs
- E38 / step 25155
- G SHA256: `256aba28c09e7fa8f29035ec838da9f2733e0e632bd6b128cf5dd425a4307e1d`

## Evaluation backends

### Whisper-small
Revision: `973afd24965f72e36ca33b3055d56a652f456b4d`

### WavLM-Base-Plus-SV
Revision: `1bfd64eca136543feb28c5ffaf05381c6af33121`

## Russian frontend

- eSpeak NG: `1.52.0.1`
- pinned eSpeak SHA256: `60874d253f54c88df7a838e76be8fd97a3109d097feb6c1b9eeed1cda3151f28`

## Evaluation conventions

- controlled neutral/positive tests use the same text and synthesis seed;
- CER/WER use one fixed ASR revision and normalization;
- final corpus comparisons include paired bootstrap confidence intervals;
- F0/RMS/duration are supporting acoustic evidence, not proof of perceived emotion;
- formal MOS was not performed.
