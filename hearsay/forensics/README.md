# Forensic experts and the router

Author: Alex Picon <alexnpc@me.com>

This optional package reports acoustic and container measurements, calibrated
expert scores where available, and the reasons each expert ran. It is separate
from the learned detector used to produce the main HEARSAY submission. Its
probabilities and explanations are prototype outputs, not proof of synthesis.

## Running it

Use the environment described in the parent HEARSAY documentation. Container
inspection uses `ffprobe` and `exiftool`; `ffmpeg` supports additional formats
and the compression round trip.

```bash
cd hearsay
# One file with JSON and a self-contained HTML report.
uv run python -m forensics analyze clip.m4a --out reports --html
# Recursively analyze a directory, with a source-to-report TSV index.
uv run python -m forensics analyze clips --out reports --tsv reports/scores.tsv -j 4
# Every eligible expert; keep the optional deep detector disabled.
uv run python -m forensics analyze clip.wav --policy full --deep never
```

`--deep always`, `uncertain` (default), or `never` controls the learned detector.
The speaker expert may download WavLM on first use. Set
`HEARSAY_SPEAKER_BACKEND=mfcc` to collect lightweight speaker measurements
without that download. **MFCC measurements have no calibrated speaker score:**
the shipped WavLM calibration is not transferable to that representation.
A failed WavLM load also falls back to unscored MFCC measurements.

`HEARSAY_DEEP_SCORES` can supply a precomputed `filename<TAB>cm-score` table.
`HEARSAY_TORCH_THREADS` controls speaker-model CPU threads.

Each report filename contains the input filename, including its extension,
plus the SHA-256 of its canonical absolute source path. Files with the same
basename in different folders therefore have separate outputs. Repeating an
input or overlapping input folders analyzes that canonical file once. The
identity remains stable while the source path remains unchanged; moving the
file changes its identity. A detected identity collision stops before reports
are written. JSON includes `source_path` and `report_id`; the TSV adds those
columns after `filename`, `cm-score`, and `experts_run`.

## Experts and routing

| Expert | Measurements |
|---|---|
| Container | Codec, encoder/device tags, RIFF consistency, timestamps and sample-count grids |
| Spectral | Bandwidth, roll-off, flatness, spectral dynamics, tonal lines and cepstral prominence |
| Compression | Spectral holes, sample histograms, clipping and change after MP3 re-encoding |
| Prosody | Pitch variation, voicing, pauses, breath candidates and rhythm |
| Environment | Hum, noise floor/drift, near-zero segments and post-speech decay |
| Splice | Clicks, phase/DC discontinuities and changes between pauses |
| Speaker | Similarities among overlapping speech-window embeddings |
| Deep | Optional learned detector or precomputed score |

The router always inspects container and spectral evidence. Compression runs
for lossy or band-limited audio; prosody requires enough detected speech;
environment analysis runs when duration or pauses make it useful. It escalates
to eligible splice, speaker and deep experts while the fused probability is
inside the configured uncertainty band. Metadata inconsistencies can force
splice analysis. Every run, skip and expert failure is recorded in `trace`.
`--policy full` runs all eligible experts, subject to `--deep`.

A batch survey suppresses shared sample-count grids as possible collection
pipeline cues. This is a heuristic tailored to the challenge collection, not
a reliable inference about arbitrary batches.

## Scores and evidence limits

The shipped [calibration.json](calibration.json) contains small standardized
logistic models, reference feature statistics and fusion weights. Expert
log-odds are weighted by confidence, fused, and shifted to the configured
synthetic prevalence (default 0.30). Unavailable or uncalibrated experts abstain.
The default alert threshold is 0.8 under a fourfold real-audio false-alarm
cost. That decision rule assumes useful posterior calibration; calibration on
these corpora does not establish reliability on arbitrary recordings.

The speaker calibration accepts `wavlm-xvector` measurements only. Both live
scoring and rescoring old caches suppress scores from MFCC or unidentified
speaker backends. These unsupported rows are also excluded from calibration
fitting. Their measurements remain available for inspection.

Near-zero segments are **processing evidence**, compatible with quiet
recording, quantization, noise gating or editing of genuine speech. They do
not establish synthetic speech. The retained feature name
`digital_silence_frac` measures runs with amplitude below `2e-5`, not exact
zeros. Its learned coefficient remains part of the unchanged calibration,
but the former fixed `+3` log-odds bonus is removed. Fusion ignores old cached
rule bonuses too. Container rules remain heuristic scored evidence rather
than a learned corpus classifier.

## Validation and calibration candidates

The corrected evaluation encloses **all expert and fusion fitting** inside
outer held-out folds. Within each outer training fold, inner out-of-fold expert
scores train the fuser; refitted experts then score the outer test fold.
Repeated source paths stay in one fold. These are clip-disjoint stratified
folds, **not** held-out-speaker or held-out-generator protocols. Generator labels
only identify diagnostic slices. Impossible grouped splits raise an error.

Router validation uses the same nested procedure: each clip is replayed with
a calibration trained outside its fold. `replay()` with a supplied fixed
calibration remains available, but its output is explicitly labeled a
possibly resubstitution diagnostic; `replay_oof()` is the held-out API.
Nested evaluation excludes deep scores without fold provenance, and held-out
router replay requires `deep="never"`.
Compute-share results sum cached expert timings and do not measure current
end-to-end latency.

Expert-omission results remove one expert at inference from held-out
predictions; they do not refit fusion. They are named
`heldout_expert_omission_auc_drop` to distinguish them from the previous
refitted ablation. Single-feature AUCs are descriptive associations.
Cross-dataset transfer results do not assert source independence: a clean
clip and its channel-transformed copy can contain the same speech.

With the labeled corpora and extracted measurements already available:

```bash
uv run python -m forensics ablation --p-spoof 0.5
```

This writes `results/forensics_ablation.json` and
`results/forensics_calibration.candidate.json`. It does **not** replace the
shipped calibration by default. `--calibration-out PATH` selects a different
candidate destination. Review genuinely held-out results before installing a
candidate; do not promote it based only on training fit or unlabeled scores.

The metric prior `--p-spoof` is separate from the router's deployment prior.
Its default remains 0.5, matching the supplied scorer archive. For prior `p`,
the analyst cost is `p * missed_fake + 4 * (1-p) * flagged_real`; the as-coded
reading is `(1-p) * flagged_real + 4 * p * missed_fake`. Each is normalized by
the cheaper constant decision. Resolve any organizer change to prior, score
direction or penalized error before comparing submissions.

## Current cached-feature evaluation

The [ablation results](../results/forensics_ablation.json) were regenerated
on September 26, 2026 at 22:49 UTC using the corrected nested procedure. All
five labeled caches and the unlabeled test cache were checked for completeness
first. No expert models were downloaded and no acoustic features were
re-extracted. Five outer folds enclose expert/fusion fitting; deep scores are
excluded. The metric prior is 0.5, while router probabilities use prior 0.3.

| Dataset | Real | Synthetic | Nested AUC | EER | Analyst minDCF |
|---|---:|---:|---:|---:|---:|
| LJ Speech / DiffSSD | 242 | 500 | 0.9864 | 6.30% | 0.1631 |
| LJ / DiffSSD, trimmed + channel | 242 | 485 | 0.9230 | 14.55% | 0.4602 |
| LA19 clean | 856 | 850 | 0.9373 | 12.72% | 0.3325 |
| LA19 + channel | 856 | 850 | 0.8183 | 25.09% | 0.7896 |
| LA19 trimmed + channel | 856 | 850 | 0.7410 | 33.65% | 0.8240 |
| LA19 trimmed + channel, at least 3 s | 111 | 258 | 0.8358 | 26.05% | 0.5050 |

The transformed LJ/DiffSSD cache contains 15 repeated path entries; evaluation
uses its 727 unique clips, matching the extraction pipeline's last-entry
semantics. The report records cache hashes, source-code hashes, counts,
generation time and warnings. Fully missing feature columns produced 72
NumPy warnings and used the fitter's existing mean/scale fallback; there were
no convergence warnings or evaluation failures.

On the 369-clip last row, held-out router replay gives **AUC 0.8048**, EER
27.08% and analyst minDCF 0.5283, versus 0.8358 / 26.05% / 0.5050 with all
eligible experts. Routing uses an estimated 73.3% of cached expert compute;
45.5% of clips escalate. Routing therefore loses some discrimination here;
these results do not support the earlier claim of unchanged accuracy.

Cross-corpus diagnostics remain weak: LJ/DiffSSD to trimmed LA19 gives AUC
0.5999 and analyst minDCF 0.9751; trimmed LA19 to transformed LJ/DiffSSD gives
AUC 0.5011 and minDCF 0.9732. This limits general-purpose claims despite strong
within-corpus results. These folds hold out clips, not generators or speakers.

These numbers evaluate the **refitted procedure**, not an independent holdout
for the shipped calibration. A new full-data candidate and its hash-bound
unlabeled test scores remain under
`_internal/hearsay/release-validation/forensics/`; neither was deployed or
substituted for the primary detector submission. Shipped calibration values,
the primary detector, its TSV and the old forensic test-score TSV are unchanged.

## Historical artifacts

The [example reports](../results/forensic_reports) and existing forensic
test-score TSV predate these corrections. The previous ablation JSON is
preserved as `forensics_ablation.historical.json` in the internal validation
directory. Its stacked-fusion evaluation was not fully nested, router replay
reused training calibration, and near-silence had a fixed bonus. Those old
accuracy and compute claims do not describe current validation.
