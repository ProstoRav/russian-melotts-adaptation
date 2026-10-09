# Cross-model Russian TTS benchmark

Controlled comparison of seven Russian TTS configurations used in the
diploma project:

- MeloTTS YO-CPT E27
- MeloTTS Dialogs E38 neutral
- MeloTTS Dialogs E38 positive
- dialogs-ru-vits2
- Silero v5.5 Russian
- Piper ru_RU irina medium
- XTTS-v2 conditioned on spk_298 references

## Final reporting metrics

- WER — mean per-prompt WER, S04 excluded
- CER — mean per-prompt CER, S04 excluded
- DistillMOS — mean across all 9 prompts
- RTF — mean across all 9 prompts

S04 is excluded only from WER/CER because the pinned ASR evaluator
systematically represents spoken numerals differently from the textual
reference; manual listening did not reveal a corresponding TTS failure.

UTMOS is not used in the final comparative conclusions because its ranking
on this small Russian TTS benchmark did not agree sufficiently well with
manual listening.

RTF is measured on the same CPU host under the benchmark's fixed thread
contract. Model loading and warm-up are excluded.

See:

- `mean_comparison.csv`
- `final/`
- `performance/`
- `provenance/`
