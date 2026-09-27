# HEARSAY prediction files

Author: Alex Picon <alexnpc@me.com>

## Accuracy export — September 27, 2026

`AlexPicon_accuracy_predictions.tsv` contains newly computed scores from the
website's frozen Accuracy model for all 1,671 challenge recordings. Use this file
when submitting the Accuracy model. It has the exact `filename` and `cm-score`
columns and preserves the supplied template's row order. Scores run from 0
(real) to 1 (synthetic); they are decision scores, not calibrated probabilities.
Full floating-point precision is retained to avoid unnecessary ranking ties.

The model, decision threshold and preprocessing were not retuned on these files.
Scoring ran in offline, non-root containers with read-only input/model mounts.
Eight workers completed the batch after resuming 84 previously completed rows.
No missing predictions were filled with defaults or original-model scores.

Validation confirmed exact filename coverage, unique rows, finite scores in
[0,1], unchanged input hashes, and agreement with the existing submission
validator. Three independent resident-worker checks (first filename, last
filename and longest recording) reproduced batch predictions exactly.
`accuracy_validation.json` records these checks and model/artifact hashes.
The underlying input manifest and batch logs remain in private storage.

The Accuracy model samples up to three spaced 4.0375-second windows and combines
that feature path with the original detector's bounded analysis. This export
uses the deployed coverage policy; it does not claim full temporal coverage of
long recordings. Hidden-test accuracy and minDCF remain unknown. Format and
reproduction checks are not evidence of correctness on the hidden labels.

## Original export

`AlexPicon_predictions.tsv` remains unchanged and uses the original detector.
Keep it as a fallback; it is not the Accuracy export. Neither file has been
sent to the organizers by this workflow.
