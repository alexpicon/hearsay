# HEARSAY: synthetic-speech scoring and evidence

*Author: Alex Picon ([alexnpc@me.com](mailto:alexnpc@me.com))*

HEARSAY produces one **0–1 synthetic-speech score per audio file** for the NSA HackGT 13 challenge: **0 means real/bona fide; 1 means synthetic/spoof**. The submitted detector combines a frozen pretrained speech representation with locally trained classifier heads and fusion. It is an experimental detector, not proof that a recording is authentic or fabricated.

The existing [submission](submissions/AlexPicon_predictions.tsv) contains 1,671 filename/score rows. The [verification record](results/release_verification.json) and [release manifest](results/release.json) bind the tested package, input files, model and predictions. There are no official held-out labels or official challenge accuracy results in this release. The low “Protocol C” score in older handoff notes belongs to a development experiment, not an evaluation of the final submitted model.

## What is included

| Component | Purpose | Relationship to the submission |
|---|---|---|
| [`hearsay/`](hearsay/) and [`models/detector.joblib`](models/detector.joblib) | Frozen XLS-R features, learned linear/spectral heads, fusion, CLI and Python API | Generates the submitted scores |
| [`forensics/`](forensics/README.md) | Optional acoustic/container experts, routing and analyst reports | Separate scoring/reporting pipeline; can consult the detector, but is not the submitted TSV recipe |
| [`results/dev_metrics.json`](results/dev_metrics.json) | Historical development protocols, final-model training composition and out-of-fold diagnostics | Evidence with the limitations below |
| [`submissions/AlexPicon_predictions.tsv`](submissions/AlexPicon_predictions.tsv) | Challenge-format filename and synthetic-score table | The existing submission artifact |

Read the [detector method and model card](docs/detector.md) for architecture, data provenance, temporal coverage and evaluation limitations. Read the [forensics guide](forensics/README.md) for optional reports and their separate calibration.


## Quickstart from a source checkout

Use Python **3.14+** and `uv`. The nested [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) define this environment; the repository-root web-app environment is separate.

```bash
cd hearsay
uv sync --frozen

# Inspect one clip, including its score and evidence.
uv run hearsay score /absolute/path/clip.wav

# Score an audio directory into a fresh TSV.
uv run hearsay predict /absolute/path/audio --out /absolute/path/predictions.tsv

# Check schema, score range, uniqueness and exact filename coverage.
uv run hearsay validate-submission --scores /absolute/path/predictions.tsv --folder /absolute/path/audio
```

The first model load needs the Hugging Face XLS-R checkpoint unless it is already cached or supplied with the release environment. Dependency/model installation requires network access; normal inference can run from prepared local assets. `ffmpeg` provides a fallback for containers that `soundfile` cannot decode. See the [reproduction limits](docs/detector.md#reproduction-and-environment) before rebuilding features or training.

`predict` recursively scans supported audio extensions, ignores hidden paths and uses each file's basename as its submission ID. Duplicate basenames are rejected. An undecodable audio file receives the saved fallback score. Inspect decode warnings or per-file `score` JSON: the two-column TSV cannot carry error status. Missing model assets or inference failures are not evidence about the file's authenticity.

To validate the existing submission against the supplied filename template and audio directory:

```bash
uv run hearsay validate-submission --scores submissions/AlexPicon_predictions.tsv --template /absolute/path/submission-template.tsv --folder /absolute/path/HackGTHearsayTesting
```

Validation checks the submission contract and coverage, **not prediction accuracy**. Never substitute the template's placeholder scores for ground-truth labels.

## Portable inference and release records

From the repository root, build the CPU image and run inference with network access disabled and the host user's permissions:

```bash
docker build -t hearsay:submission hearsay
mkdir -p /absolute/output
docker run --rm --network none --user "$(id -u):$(id -g)" -v /absolute/audio:/input:ro -v /absolute/output:/output hearsay:submission predict /input --out /output/predictions.tsv --threads 4
```

Replace the absolute input/output paths with your directories. The [Dockerfile](Dockerfile) installs the locked environment, packages the detector, and downloads the pinned XLS-R backbone **during the build**. Its preflight imports the installed package outside the source checkout. Building requires network access; the prepared image runs primary-detector inference offline. It does not preload the optional forensic WavLM speaker model, which may abstain offline unless separately prepared.

To build the Python wheel instead, from the repository root:

```bash
cd hearsay
uv build
```

The wheel includes the learned detector and forensic calibration. It does not include Python dependencies or the large pretrained backbone; use the source lock or the image when reproducing the environment.

After fresh inference, generate checksummed release records from `hearsay/`:

```bash
uv run python -m hearsay.release --scores submissions/AlexPicon_predictions.tsv --template /absolute/path/submission-template.tsv --audio-dir /absolute/path/HackGTHearsayTesting --recomputed /absolute/path/fresh-predictions.tsv --recomputed /absolute/output/predictions.tsv --tolerance 5e-5 --out results/release.json
```

Each `--recomputed` adds an independently generated score file. The example explicitly selects **`5e-5` absolute tolerance for cross-runtime reproduction**; the tool's stricter default remains `1e-5`. It writes `results/release.json` and `results/release.inputs.json`, including SHA-256 hashes for the submission, template, input audio, model, source files and lock/build configuration. `results/release_verification.json` records executed checks, including the initial strict-tolerance failure and the measured numerical differences. These records establish provenance and numerical reproduction, not accuracy.

Full local and offline, unprivileged Docker runs both scored all **1,671** challenge files. The container run had **zero decode failures**. Comparison with the original submission:

| Inference environment | Largest absolute TSV-score difference | Entries differing at six decimal places |
|---|---:|---:|
| Local raw-audio inference | approximately `1e-6` | 22 |
| Offline CPU container | approximately `1.9e-5` | 559 |

The container run exceeded the default `1e-5` tolerance on 15 entries and passed the explicitly selected `5e-5` tolerance. No decisions changed at either saved operating threshold or at 0.5. Scores are **not bit-identical** across these runtimes; the [model card](docs/detector.md#reproduction-and-environment) explains the inspected floating-point variation. This agreement is not a measurement of hidden-label accuracy.

## Score interpretation and temporal coverage

The score is a learned sigmoid output, not a demonstrated probability of fabrication for arbitrary real-world audio. The frozen SSL branch analyzes **only the first 12 seconds**; the spectral branch measures the full decoded file. The shipped fusion assigns the spectral branch weight **0**, so content after 12 seconds does not affect the submitted classification. This system does not provide whole-file splice detection for longer recordings.

Per-file JSON exposes `duration_s`, `ssl_analyzed_duration_s`, `ssl_truncated`, and separate SSL/spectral intervals in `analysis_coverage`. Use those fields when describing how much of a recording was analyzed; the submission TSV does not contain them. The hashed challenge input manifest records two clips longer than 12 seconds; their scores retain the same prefix policy as the original submission.

The saved detector labels scores at or above approximately **0.385415** as `synthetic`, below **0.150188** as `bona fide`, and the interval between them as `uncertain`. These are development operating points under two cost interpretations, not universally valid decision thresholds. The decode fallback is approximately **0.249763**; retain the accompanying error/log evidence when reporting results.

```python
from hearsay.detector import score_file

result = score_file("/absolute/path/clip.wav")
print(result["probability"], result["verdict"], result["error"])
print(result["evidence"])
```

For an optional analyst report, after preparing the system tools described in the [forensics guide](forensics/README.md):

```bash
uv run python -m forensics analyze /absolute/path/clip.wav --out /absolute/path/reports --html
```

Its router score and diagnostic findings are separate from the detector-only submission. Two [current container-generated examples](results/current_forensic_reports/README.md) exercise the complete report path with deep inference. The current [forensic ablation](results/forensics_ablation.json) reruns nested clip-disjoint evaluation on cached measurements, with deep inference excluded. Neither establishes challenge accuracy; older example reports and the forensic test-score TSV remain historical.

## Evaluation: keep model, split and cost definition together

The following stored results use spoof prior **0.5**, miss cost **1** and false-alarm cost **4**, with the documented analyst interpretation. Values come from [dev_metrics.json](results/dev_metrics.json), not a newly rerun benchmark.

| Evaluation | Records scored | Analyst minDCF | EER | AUC |
|---|---:|---:|---:|---:|
| Final-model recipe: grouped base-head out-of-fold predictions | 34,632 | 0.073194 | 1.4755% | 0.996802 |
| Historical Protocol C development model: LA unseen attacks plus held-out VCTK speaker subset | 3,962 | 0.007566 | 0.2020% | 0.999981 |

The final artifact was refit on all 34,632 training records, including LA evaluation-partition attacks. It has no separate final holdout. Fusion weights and operating thresholds were fitted using the same out-of-fold predictions summarized in the first row, so that row is not a fully nested estimate of the entire selected pipeline. Protocol C used a different 18,910-record fit; it also informed development choices and does not establish the final model's challenge performance.

The supplied scorer archive uses prior **0.5** and a bona-fide-high score convention, while the written challenge contract asks for synthetic-high scores and describes a fourfold penalty for real audio falsely flagged as fake. Confirm the final class prior directly with the organizers. Preserve the submission's `0=real, 1=synthetic` direction and report both metric interpretations with the chosen prior.

With an actual labeled key (`filename`, `cm-label`; labels `bonafide` or `spoof`):

```bash
uv run hearsay evaluate --scores /absolute/path/predictions.tsv --key /absolute/path/labeled-key.tsv --spoof-prior 0.5
uv run hearsay evaluate --scores /absolute/path/predictions.tsv --key /absolute/path/labeled-key.tsv --spoof-prior 0.3
```

These commands are sensitivity analyses unless the organizer confirms the final evaluation contract. They do not reveal the unknown challenge labels. The historical unlabeled Gaussian-mixture estimates are **not measured EER, minDCF or accuracy**.

## Known limits and reproduction

- Training/channel choices used unlabeled test-set statistics and model-score distributions: this is **test-informed/transductive development**, not an untouched blind evaluation.
- VCTK and ASVspoof LA share speakers and source material. Grouping and held-out attack names do not prove disjoint utterances, speakers or pretrained-model data across every corpus.
- Training uses repeated augmented views of some source material; record counts are not counts of independent recordings. A linear readout can still memorize or exploit high-dimensional corpus shortcuts.
- Real-negative diversity is limited. An earlier LA-only model flagged 47.1% of the provided LJ real clips above 0.5; the final model includes those clips in training, so they no longer provide an independent generalization check.
- Metadata, bandwidth, pauses and pitch describe processing/acoustics; they cannot alone prove synthesis, identify a generator or establish speaker identity.

For the exact final training recipe, dependency/download boundaries, evidence interpretation and artifact pointers, see [docs/detector.md](docs/detector.md).
