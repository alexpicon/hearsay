# Author: Alex Picon <alexnpc@me.com>
FROM ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 AS uv
FROM python:3.14-slim-bookworm@sha256:82bc3c539b8813ada9d68c63b40158fa002f7f33de9bf3312a3dfdc0620dff56

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Keep the large, fixed backbone cache independent of source/document edits.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/wav2vec2-xls-r-300m', revision='1a640f32ac3e39899438a2931f9924c02f080a54', allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'])"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY README.md ./
COPY hearsay ./hearsay
COPY forensics ./forensics
COPY models/detector.joblib ./models/detector.joblib
RUN uv sync --frozen --no-dev --no-editable

# Exercise the installed wheel rather than importing the copied source tree.
WORKDIR /tmp
ENV HOME=/tmp NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib
RUN python -c "from hearsay.inference import get_detector; from hearsay.ssl import get_embedder; from hearsay.model import MODEL_PATH; assert '/site-packages/' in str(MODEL_PATH); d=get_detector(); get_embedder(d.ssl_model, d.n_layers)"

ENTRYPOINT ["hearsay"]
CMD ["--help"]
