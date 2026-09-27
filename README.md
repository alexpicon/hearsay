# HEARSAY

Author: Alex Picon <alexnpc@me.com>

Audio deepfake detection with Speed and Accuracy modes. Upload a recording,
inspect its score and processing time, or run both modes on the same audio.

## Run

```sh
uv sync
uv run uvicorn server.main:app --host 127.0.0.1 --port 8888
```

Inference requires the reviewed Docker images and pretrained weights described
in [deployment instructions](deploy/README.md). Workers run offline with a
read-only filesystem and no secret mounts. The private access code is stored
in `_internal/runtime/dashboard/access-key`. Runtime assets and `.env` are ignored.

The original model CLI is under `hearsay/`; install its environment separately
with `cd hearsay && uv sync`. The original submission artifact is preserved in
`hearsay/submissions/`; it does not use the Accuracy fusion head.

## Models and limits

Speed uses the original detector. Accuracy combines pretrained features with
our trained decision head. Scores are synthetic-high decision outputs, not
calibrated probabilities or proof of authenticity. Model usage restrictions and
credits are in [MODEL_CREDITS.md](apps/hearsay-compare/MODEL_CREDITS.md).

## Tests

```sh
uv run python -m unittest discover -s tests -p test_hearsay_compare.py -v
```

The complete Accuracy prediction export and validation notes are available in
[hearsay/submissions/](hearsay/submissions/README.md).
