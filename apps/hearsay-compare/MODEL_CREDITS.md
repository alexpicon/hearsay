# Model credits and limits

Author: Alex Picon <alexnpc@me.com>

HEARSAY uses pretrained components; those backbones were not trained by us.

- Facebook XLS-R: `facebook/wav2vec2-xls-r-300m`, revision
  `1a640f32ac3e39899438a2931f9924c02f080a54`.
  Model and license: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- DF-Arena: `Speech-Arena-2025/DF_Arena_500M_V_1`, revision
  `8258fa8e74ff9b8ad20d4c939c1a7f694a6e4080`.
  Restricted to research/non-commercial use; commercial use requires permission.
  Model and license: https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1

The Accuracy decision head is stored in `hearsay/models/accuracy_v2.json`.
It combines features and applies a validation-selected operating point. Its
output is not calibrated confidence. Speed analyzes up to 12 seconds; Accuracy
samples up to three evenly spaced 4.0375-second windows. Scores do not prove
whether a recording is authentic. Sample-audio credits are in `samples/README.md`.
