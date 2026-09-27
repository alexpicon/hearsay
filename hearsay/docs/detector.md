# HEARSAY detector: method, data and model card

*Author: Alex Picon ([alexnpc@me.com](mailto:alexnpc@me.com))*

This document describes the learned detector used for [AlexPicon_predictions.tsv](../submissions/AlexPicon_predictions.tsv). Scores run from **0=real/bona fide to 1=synthetic/spoof**. The intended use is challenge scoring and exploratory analyst review. Neither a score nor an explanation certifies authenticity, proves a manipulation, or identifies a generator.

The [top-level quickstart](../README.md) covers inference and submission validation. The optional [forensic router](../forensics/README.md) has separate experts, calibration and reports. The repository's [Tell API](../../server/routers/tell.py) imports `analysis.tell`, not this detector; its 0–100 scores and web-app metrics must not be presented as HEARSAY submission results.

## What we used and what we trained

```text
audio -> decode, average channels, resample to 16 kHz mono
  |-- first <=12 s -> frozen pretrained XLS-R, first 8 transformer blocks
  |     mean/std pooling of hidden states 3..7 -> 10,240 dimensions
  |     -> fitted standardizer -> L2 logistic regression
  |-- full waveform -> 99 named acoustic features, 3 excluded
  |     -> LightGBM on 96 features -> acoustic contribution report
  '-- non-negative logistic fusion of both head logits -> synthetic score
```

**Used:** the pretrained `facebook/wav2vec2-xls-r-300m` representation at release revision `1a640f32ac3e39899438a2931f9924c02f080a54`, PyTorch/Transformers, LightGBM, scikit-learn, NumPy/SciPy, soundfile and optional ffmpeg. We did not pretrain or fine-tune the XLS-R network. The implementation truncates the loaded encoder to eight blocks and runs it in inference mode. [Source: ssl.py](../hearsay/ssl.py).

**Built/trained:** corpus manifests and deterministic sampling, channel augmentation, layer-statistics feature extraction, the standardized linear readout, the acoustic tree head, constrained fusion, evaluation adapters and evidence/CLI output. The linear head uses hidden-state indices 3–7, L2 `C=0.01`, and 10,240 pooled inputs. [Sources: corpus.py](../hearsay/corpus.py), [extract.py](../hearsay/extract.py), [backend.py](../hearsay/backend.py), [train.py](../hearsay/train.py).

The saved [detector bundle](../models/detector.joblib) has fusion weights **SSL=0.3861, spectral=0.0** and bias approximately **−0.221871**. Thus the submitted ranking is driven by the SSL linear head. LightGBM is retained for a separate acoustic explanation; its contributions do not cause the final score when its fusion weight is zero. Non-negative fusion prevents a head from receiving a negative coefficient; it does not guarantee transfer or calibration.

The acoustic extractor names 99 measurements; duration, leading silence and trailing silence are excluded, leaving 96 tree inputs. This removes three identified shortcuts, not all possible corpus/channel shortcuts. A linear classifier over 10,240 learned representation features can memorize or exploit dataset regularities. Freezing the backbone and using regularization reduce flexibility; they do not make memorization impossible.

## Input, output and coverage

[Audio decoding](../hearsay/audio.py) first tries soundfile, averages channels, and resamples to 16 kHz with a polyphase filter. ffmpeg is the fallback decoder. An empty or undecodable input is reported as an error rather than being silently treated as a successful classification.

[Inference](../hearsay/inference.py) keeps the full duration for reporting but limits the SSL representation to the **prefix ending at 12 seconds**. Clips shorter than 0.5 seconds are padded for SSL. Spectral features use the full waveform. Since the released fusion gives those spectral features zero weight, a synthetic segment occurring only after the first 12 seconds can be missed. There is no sliding-window score aggregation in this release. A full-file duration field does not mean the classifier analyzed the full file.

`score_file` returns a dictionary with filename, synthetic score, verdict, duration, evidence and error status; `float(result)` returns the synthetic score. Temporal coverage is explicit:

| JSON field | Meaning |
|---|---|
| `duration_s` | Full decoded duration |
| `ssl_analyzed_duration_s` | Decoded audio covered by the SSL prefix, at most 12 seconds |
| `ssl_truncated` | Whether the waveform extends beyond that prefix |
| `analysis_coverage.ssl` / `.spectral` | Start/end intervals for the SSL prefix and whole-file acoustic measurements |

Decode failures carry zero duration/SSL coverage, empty coverage intervals and a non-null error. The submission TSV contains only:

```text
filename<TAB>cm-score
```

Scores are written to six decimal places. `predict` recursively scans supported audio extensions while ignoring hidden paths; submission IDs are basenames, and duplicates are rejected. `submit` uses cached feature IDs. `validate-submission` checks unique identifiers, finite scores within [0,1] and exact coverage against a filename template, audio directory, or both. This validation establishes file coverage and formatting, not ground truth.

The artifact's saved operating points are **0.3854147939** (`analyst`) and **0.1501876711** (`as_coded`). The UI/API verdict is `synthetic` above the first, `bona fide` below the second, and `uncertain` between them. The saved default for decode failure is **0.2497626305**. These values were selected during development under prior 0.5 and two different error-cost readings; they are not an independently calibrated deployment policy.

## Training data and provenance

The final bundle metadata and [dev_metrics.json → final_model](../results/dev_metrics.json) record **34,632 training records: 17,822 bona fide and 16,810 spoof**. “Records” matters: different named feature sets can contain augmented views of the same source recording.

| Source | Final records | Provenance recorded locally |
|---|---:|---|
| ASVspoof 2019 LA bona fide | 8,580 | `Bisher/ASVspoof_2019_LA` train/validation/test parquet files |
| ASVspoof 2019 LA spoof | 14,610 | Same source, attacks A01–A19 across the selected partitions/views |
| VCTK bona fide | 9,000 | `sanchit-gandhi/vctk`, VCTK 0.92, both microphones; 4,000 + 5,000 sampled records |
| DiffSSD spoof | 2,200 | Provided sample, ten generators; default and room-tone feature sets |
| LJ Speech bona fide | 242 | Provided resampled clips |

[Source manifest code](../hearsay/data.py) records `uid`, corpus, partition, speaker, generator/system, label, path and parquet row coordinates. [Set definitions](../hearsay/corpus.py) specify sampling caps, seed offsets and augmentation configuration. Raw corpora and feature caches live outside the release under `HEARSAY_DATA`; they are not bundled with the model or wheel. Dataset access, revisions and redistribution terms must be handled separately. The recorded corpus names do not establish an exact immutable upstream dataset revision.

The final training set includes ASVspoof's **evaluation partition**. That is external labeled training data for this final challenge artifact; it cannot simultaneously be advertised as an untouched test of that artifact. VCTK and ASVspoof share speakers/source material. There is no published cross-corpus near-duplicate audit proving utterance independence, and the pretrained backbone's own data overlap was not audited.

### Test-informed channel adaptation

The team inspected unlabeled challenge audio statistics and transcripts, then chose source corpora and augmentation accordingly. [data_match.json](../results/data_match.json) records an approximately 3.41-second median duration, 24.18 dB median estimated SNR and attenuated upper-band energy. Those measurements support a **VCTK/LA-style corpus/channel hypothesis**, not proof of the hidden corpus, individual clip labels or exact preprocessing chain. Model-selected “real” subsets are predictions, not known-real controls.

[augment.py](../hearsay/augment.py) generates colored noise, low-frequency rumble, optional MP3/resampling round trips, a 6.9–7.6 kHz low-pass, normalization and quantization. The room-tone configuration applies analogous noise to spoof examples to reduce the shortcut “room tone means real.” These are simulated conditions, not a recovered ground-truth channel.

Sampling and augmentation are seeded per record. Training SSL crops normally use up to 4.5 seconds with deterministic random offsets; length-sorted batches crop to the shortest member within a small length range. Test/inference SSL uses a deterministic first-12-second prefix. Acoustic features are extracted before those SSL crops. ffmpeg availability affects MP3 augmentation, so the same seed alone is insufficient for identical features across environments.

The unlabeled test score distribution also informed model iteration. This is **transductive/test-informed development**. No challenge labels were used in the recorded recipe, but the challenge inputs were not held untouched. A shifted channel, language, generator or recording environment can invalidate apparent gains.

## Evaluation and what the numbers establish

The following are **stored development results**, not a fresh reproduction or official challenge evaluation. [dev_metrics.json](../results/dev_metrics.json) separates historical protocols from `final_model`; those sections must remain separate when quoting a score.

| Fit and evaluation | Training records | Scored records | Analyst minDCF | As-coded minDCF | EER | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Final recipe, grouped base-head out-of-fold predictions | 34,632 | 34,632 | 0.073194 | 0.051628 | 1.4755% | 0.996802 |
| Historical Protocol C: room-tone LA eval spoof vs held-out VCTK subset | 18,910 | 3,962 | 0.007566 | 0.005964 | 0.2020% | 0.999981 |
| Historical A2: LA eval through simulated channel | 18,910 | 6,900 | 0.006308 | 0.004385 | 0.2026% | 0.999980 |

All displayed minDCF values use **prior 0.5, Cmiss=1, Cfa=4** under the interpretations below. A result field named `official_scorer` records whether the supplied calculation module was imported; it does not mean the organizers evaluated this model.

### Final-model out-of-fold diagnostics

[train.py](../hearsay/train.py) builds four GroupKFold splits. Spoof records are grouped by generator/system; bona fide records by speaker. Each base head predicts records excluded from that head's fit. The grouping is **class-dependent**: it does not ensure every speaker is disjoint between all bona fide and spoofed examples, or prove independence across related corpora.

Fusion is then fitted on all pooled out-of-fold logits, and its reported metric and operating points are computed on those same logits/labels. This is useful development evidence, but it is not a nested outer holdout for fusion, threshold selection or recipe selection. The delivered base heads are subsequently refit on all training records. There is **no separate final-model holdout** in the artifact.

### Historical Protocol C

Protocol C evaluates a different fit, omitting A07–A19 attacks and 25 VCTK speakers from the relevant training subsets. Its 3,962 scored records comprise 1,950 spoof and 2,012 bona fide records. The recorded caveat already notes that VCTK and LA share speakers, so “25 held-out speakers” applies within VCTK, not automatically across every source. The frontend/layer/augmentation decisions were also compared using development data; the result is not an untouched model-selection test.

The tracked metrics preserve the composition and description, but do not publish a standalone exact Protocol C row/speaker split manifest and dedicated reproduction command. Re-running the final training command below does **not** recreate this earlier fit or independently verify its headline.

### Unlabeled mixtures and real-negative generalization

The `unlabeled_test_check` section fits a two-component Gaussian mixture to challenge scores. Its “implied EER/minDCF” values assume Gaussian components correspond to the true classes. Component separation and an approximately 73%/27% mixture cannot establish labels, actual EER, accuracy, or that one model improved on the hidden test. Do not quote the historical **1.6% implied EER** as measured performance.

The recorded `cross_corpus_check` shows an earlier LA-only model incorrectly flagging **47.1%** of the provided real LJ clips above 0.5. Those clips were then added to the final fit. This demonstrates a development failure and motivates broader real-speech coverage; it does not validate final generalization. Independent real negatives spanning speakers, languages, microphones, codecs, noise and conversational speech remain a key evaluation need.

## Score direction, prior and cost ambiguity

The submission follows the written contract: **high means synthetic**. The supplied ASVspoof-style scorer naturally expects **high means bona fide**. Two readings are retained in [metrics.py](../hearsay/metrics.py):

| Output | Meaning |
|---|---|
| `min_dcf` | Analyst interpretation: real clips flagged as synthetic carry the fourfold false-alarm cost |
| `min_dcf_as_coded` | Supplied-code orientation, using bona-fide-high `1 - score`; the fourfold error cost then falls on spoof clips accepted as real |
| `eer`, `auc` | Discrimination metrics on synthetic-high scores; neither is submission accuracy at a fixed threshold |

The supplied archive uses `Pspoof=0.5`. Confirm the organizer's final score direction, class prior and penalized error before interpreting a challenge minDCF.

`hearsay evaluate --spoof-prior 0.5` and `--spoof-prior 0.3` make this sensitivity explicit; outputs identify the selected prior and costs. Changing that option recomputes evaluation, not the trained model or saved verdict thresholds. For a fixed labeled set and cost/prior definition, minDCF minimizes over thresholds and is governed by score ordering; it is not evidence of probability calibration. Compare systems only on the same labeled data and metric contract.

## Reading the evidence

`evidence.summary` reports the fused and individual head scores and their influence without assigning a second verdict. The top-level `verdict` comes from the saved operating thresholds. A zero-weight head is explicitly diagnostic, even when its individual score is large.

`ssl_layer_logit` partitions the linear head's logit by representation layer; the head's intercept completes the sum before fusion. These additive feature contributions describe this classifier, not interpretable acoustic causes or a proof that a particular layer detected fabrication.

`spectral_evidence` reports named acoustic measurements and TreeSHAP-style contributions to the **spectral head**. `pushes: synthetic` means a feature increases that head's logit relative to its model baseline. It is not independent forensic confirmation. Always read `fusion_weights`: for the released detector, the spectral head contributes zero to the final classification.

The optional forensic system's container/processing findings, route trace and score are separate outputs. Processing metadata can arise from recording, transcoding or editing real speech. The current [ablation artifact](../results/forensics_ablation.json) records a corrected nested clip-disjoint evaluation on six cached-feature datasets, excluding the deep model. Outer folds isolate clips, while inner out-of-fold scores fit fusion; this does not establish speaker/generator independence or rerun acoustic extraction. Its generation metadata records cache/source hashes and metric prior 0.5. The shipped calibration was not replaced by the candidate fit. [Example reports](../results/forensic_reports/) and `results/forensics_test_scores.tsv` remain historical outputs. Follow the current [forensics guide](../forensics/README.md) for the separate protocol and results.

## Reproduction and environment

From the nested project directory, `uv sync --frozen` uses [uv.lock](../uv.lock); Python 3.14+ is required. The lock selects the CPU PyTorch index. Decoder/model/library versions affect features, resource use and scores. Inference loads the frozen Hugging Face backbone separately from the small learned-head joblib bundle; a wheel containing the head is not a self-contained copy of XLS-R. Prepare the backbone/cache before an offline run. `HF_HOME` controls the Hugging Face cache; `HEARSAY_MODEL` or the CLI's `--model-path` selects another trusted detector bundle. The default XLS-R revision is pinned in [ssl.py](../hearsay/ssl.py); `HEARSAY_SSL_REVISION` overrides it for experiments and must be recorded as a changed configuration.

The [portable release instructions](../README.md#portable-inference-and-release-records) cover `uv build`, the CPU Docker image and unprivileged inference with `--network none`. The wheel embeds the learned bundle at `hearsay/assets/detector.joblib`; source checkouts also support `models/detector.joblib`. The image preloads the pinned primary backbone during its networked build, but not the optional forensic speaker model. Primary-detector offline readiness does not imply every optional expert is available offline.

[release.py](../hearsay/release.py) checks coverage and raw-score reproduction, hashes input audio and inference/build files, records the model/backbone/environment, and emits `results/release.json` plus `results/release.inputs.json`. Its default comparison tolerance remains `1e-5`. The documented release command explicitly uses **`--tolerance 5e-5`** for cross-runtime comparisons; `different_rounded_scores` makes numerical changes visible. `results/release_verification.json` records executed package/container/test checks and the initial stricter comparison failure. Matching scores verify the implementation path on those bytes, not hidden-label accuracy or exact reproduction of training.

Both the local raw-audio run and the offline, unprivileged CPU container scored all 1,671 files; the container reported zero decode failures. Against the original TSV, the local run's largest absolute difference was approximately `1e-6` with 22 changed six-decimal entries. The container's largest TSV difference was approximately `1.9e-5`, with 559 changed entries and 15 exceeding `1e-5`. All passed the explicitly declared `5e-5` tolerance; none changed classification at the two saved thresholds or 0.5. This is numerical agreement within a stated tolerance, not bit identity.

For the largest-difference clip, the decoded waveform and spectral features were byte-identical between runtimes. Nine of 18,432 FP16 pooled SSL values differed, by at most one FP16 step (maximum approximately `0.0009766`), producing an unrounded probability difference of approximately `1.907e-5`. This diagnostic supports small floating-point representation differences for that clip; it is not a general accuracy guarantee. The release retains the original scores and records this measured variance instead of silently relaxing the tool's default check.

Training additionally requires the externally obtained data below, under an absolute `HEARSAY_DATA` directory:

```text
ext/la19_train.parquet
ext/la19_validation.parquet
ext/la19_test.parquet
ext/vctk_*.parquet
ext/vctk_b/vctk_*.parquet
diffssd/DiffSSD/generated_speech/<generator>/...
lj/resampled/...
test/HackGTHearsayTesting/...
features/wav2vec2-xls-r-300m/...
```

The package reads these files; `extract` does not download the training corpora. Feature archives include source/partition/speaker/system metadata, but the existing cache format does not completely fingerprint every augmentation/library/backbone revision. Keep raw-data and environment manifests with a reproduction. The exact historic result cannot be promised from corpus names and a random seed alone.

To rebuild the final recipe after preparing matching source data, run from `hearsay/` and use fresh output paths:

```bash
export HEARSAY_DATA=/absolute/path/hearsay-data
for s in la_train la_dev la_eval diffssd lj vctk vctk_b la_bona_room la_train_room la_eval_room diffssd_room la_dev_room la_eval_room2 test; do
  uv run hearsay extract --set "$s" --seed 0 --threads 4
done
uv run hearsay train --train la_train,la_dev,la_eval,diffssd,lj,vctk,vctk_b,la_bona_room,la_train_room,la_eval_room,diffssd_room,la_dev_room,la_eval_room2 --use-layers 3,4,5,6,7 --c 0.01 --out models/retrained-detector.joblib --metrics results/retrained-metrics.json
uv run hearsay submit --model-path models/retrained-detector.joblib --template /absolute/path/submission-template.tsv --out submissions/retrained-predictions.tsv
uv run hearsay validate-submission --scores submissions/retrained-predictions.tsv --template /absolute/path/submission-template.tsv --folder /absolute/path/HackGTHearsayTesting
```

Do not include `test` in `--train`: its labels are unknown. The final-recipe command deliberately includes externally labeled LA eval material and does not reproduce a held-out LA test. `submit` uses cached, unaugmented test features; `predict` recomputes from audio. Fresh features/inference and cached submission should be compared on the same assets and reported with a numerical tolerance, not assumed bit-for-bit identical.

Rebuilding features may download the pretrained backbone and can be CPU/memory intensive. ffmpeg is optional for decoding files supported by soundfile but required to reproduce the MP3 augmentation branch. Optional forensic reports additionally use system tools/model assets described in their own guide. No runtime benchmark is claimed for an unspecified machine.
