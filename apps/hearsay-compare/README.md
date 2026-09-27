# HEARSAY website

Author: Alex Picon <alexnpc@me.com>

Authenticated uploads support Speed, Accuracy, and both modes. Scores, timings
and optional supplied-label agreement appear in the results table. Uploaded
recordings are deleted after processing. Demonstration clips and their source
credits are in `samples/`. See `MODEL_CREDITS.md` for model usage restrictions.

## Listening game and learning sections

The home page includes 48 genuine/synthetic pairs across 14 speakers. Pair order
and A/B placement are randomized. Playing both clips unlocks one guess per round;
the reveal uses dataset labels, not a saved model prediction. Session scores stay
in memory. Speaker selection, score reset, and per-clip handoff to the existing
analysis form are supported. Playing the game needs no access code; inference
still requires authentication. Pair labels are public educational data.

The educational sections cover impersonation, listening limitations, the analysis
workflow, observed deployment timings, and a hypothetical false-alarm exercise.
The slider's 90% detection and 5% false-positive rates are teaching assumptions,
not HEARSAY evaluation results. No universal accuracy or superiority is claimed.

Source statistics were checked on September 27, 2026:

- FTC, June 2026: $3.5 billion in reported imposter-scam losses in 2025 across
  channels, not a voice-cloning-specific estimate:
  https://www.ftc.gov/news-events/news/press-releases/2026/06/ftc-data-show-people-reported-losing-3-point-5-billion-imposter-scams-2025
- UCL / PLOS ONE, August 2023: 529 participants identified 73% of the synthetic
  speech in the study's English/Mandarin experiment. This does not estimate
  HEARSAY accuracy or the player's expected game score:
  https://www.ucl.ac.uk/news/2023/aug/humans-unable-detect-over-quarter-deepfake-speech-samples
- FTC practical guidance:
  https://consumer.ftc.gov/articles/scammers-use-fake-emergencies-steal-your-money

Browser verification covered correct and wrong guesses, one vote per round,
nonrepeating pairs within a deck, all four JFK pairs, score reset, selection for
analysis, slider calculations, and a 390px mobile viewport without overflow.
All 48 pairs were checked against the recording catalog for labels and speakers.
