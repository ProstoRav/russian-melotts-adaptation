# Final Evaluation Package V1

## Scope

This package consolidates the objective evaluation trajectory:

- English-v3 pretraining → Russian YO-CPT adaptation → E27 chosen candidate.
- E27 initialization → Dialogs neutral/positive adaptation → E38 chosen candidate.

YO-CPT uses the frozen **validation holdout** (n=194).
Dialogs uses the frozen **test holdout** (n=55).

`E27` and `E38` are called **chosen candidate**. E30/E40 are shown simply as later measured epochs.

## Primary intelligibility results

### Dialogs

- Corpus WER: 29.35% at E1 → 12.44% at E38.
- Relative reduction: 57.61%.
- Bootstrap 95% CI for relative reduction:
  [44.09%, 69.08%].

- Corpus CER: 11.52% at E1 → 4.06% at E38.
- Relative reduction: 64.78%.
- Bootstrap 95% CI for relative reduction:
  [52.37%, 75.58%].

### YO-CPT

- Corpus WER: 63.67% at E1 → 24.69% at E27.
- Relative reduction: 61.22%.
- Bootstrap 95% CI for relative reduction:
  [56.30%, 66.82%].

- Corpus CER: 32.88% at E1 → 11.82% at E27.
- Relative reduction: 64.05%.
- Bootstrap 95% CI for relative reduction:
  [58.60%, 69.53%].

## Style separation

At Dialogs E38, controlled matched-text / matched-seed evaluation shows:

- positive − neutral duration median:
  1.969%,
  direction consistency 9/9 prompts;

- positive − neutral RMS median:
  -0.796 dB,
  direction consistency 9/9 prompts.

Pitch-based features are not consistently directional at E38 and therefore are supporting diagnostics only.

The learned SID0/SID1 embedding L2 distance increased from
0.240599 at E1 to 0.501917 at E38.

## Speaker / voice identity

### YO-CPT

The natural validation median similarity to the spk_298 train enrollment centroid is
0.988673.

The synthetic trajectory reaches:

- E1: 0.977103
- E27 chosen candidate: 0.988410

This supports speaker preservation for the single-speaker YO-CPT stage.

### Dialogs

Dialogs contains two real source actors (M/S), but the adaptation labels encode neutral/positive rather than actor identity.
Therefore M/S affinity is treated as an appendix diagnostic, not speaker preservation.

For controlled S01–S09 matched prompts, SID0 neutral vs SID1 positive median external speaker-embedding cosine is
0.991871 at E38
(95% CI [0.985974, 0.994211]).

This supports common synthetic voice identity across the two conditioning labels while their acoustic realization differs.

## Numeric orthography audit

No digit-containing reference texts occur in the current frozen evaluation holdouts:

- Dialogs: 0 / 55.
- YO-CPT: 0 / 194.

Therefore numeral normalization is not a contributor to the reported current holdout CER/WER.

## Interpretation limits

- CER/WER are ASR-mediated intelligibility metrics, not human MOS.
- Natural-reference ASR is a baseline, not an error floor to subtract from synthetic results.
- Dialogs F0 / energy / duration support style analysis but do not independently prove perceived emotion.
- Dialogs actor-centroid affinity is not a speaker-preservation metric because M/S identity was not used as conditioning.
- P10–P90 whiskers in publication figures are a presentation convention and are explicitly labeled as such.
