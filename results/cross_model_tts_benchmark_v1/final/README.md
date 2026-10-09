# Cross-model Russian TTS benchmark v1 — final

Status: **FROZEN_FINAL_RESULTS**

## Headline comparison

| System | WER ↓ | CER ↓ | UTMOSv2 ↑ | Distill-MOS ↑ | Median RTF CPU×2 ↓ | Full-wave latency, s ↓ | Native stream TTFA, s ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Silero v5_5_ru | 13.77% | 12.77% | 2.692 | 4.523 | 0.026 | 0.199 | — |
| Piper irina-medium | 14.37% | 13.39% | 3.396 | 4.412 | 0.040 | 0.360 | — |
| XTTS-v2 spk298 | 14.97% | 14.11% | 2.295 | 4.274 | 1.934 | 16.714 | 1.635 |
| dialogs-ru-vits2 | 17.37% | 14.21% | 2.462 | 4.432 | 0.273 | 2.032 | — |
| MeloTTS E38 positive | 18.56% | 13.90% | 2.470 | 4.564 | 0.380 | 3.025 | — |
| MeloTTS E38 neutral | 25.15% | 16.37% | 2.411 | 4.532 | 0.391 | 2.914 | — |
| MeloTTS YO-CPT E27 | 33.53% | 19.57% | 1.813 | 4.326 | 0.425 | 3.765 | — |

UTMOSv2 and Distill-MOS are automatic MOS predictions, not formal human MOS.

## Evaluation contract

The headline comparison uses the same frozen semantic utterances and intended pronunciation/stress target. Project MeloTTS receives explicit U+0301 stress marks; external systems use their native frontends. WER/CER always use `text_plain` as reference.

WER/CER use pinned Whisper-small with historical `ru_simple_v1`. UTMOSv2 5 folds × 5 repetitions is the primary AutoMOS metric; Distill-MOS v7 is the secondary cross-check.

## S04 sensitivity

S04 contains verbalized numbers. Whisper frequently outputs the same content as digits. Primary WER/CER retain this frozen orthographic mismatch. `s04_numeric_orthography_sensitivity.csv` is diagnostic only and must not replace the headline metrics.

## Statistics

`paired_bootstrap.csv` contains all pairwise paired prompt-level 95% bootstrap CIs across S01-S09 for WER, CER, UTMOSv2 and Distill-MOS.
