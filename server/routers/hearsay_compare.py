# Author: Alex Picon <alexnpc@me.com>
"""Authenticated control plane for isolated HEARSAY comparisons."""
import fcntl
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from urllib.parse import unquote
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from hearsay.benchmark.dashboard import ENV, MODELS, STATE, save

router = APIRouter(prefix='/api/hearsay-compare', tags=['hearsay-compare'])
ROOT = Path(__file__).resolve().parents[2]
STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
(STATE / 'jobs').mkdir(exist_ok=True, mode=0o700)
(STATE / 'docker-config').mkdir(exist_ok=True, mode=0o700)
KEY = STATE / 'access-key'
try:
    fd = os.open(KEY, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
except FileExistsError:
    pass
else:
    with os.fdopen(fd, 'w') as handle:
        handle.write(secrets.token_urlsafe(32) + '\n')


def authorize(request):
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Cross-origin requests are forbidden.')
    if not hmac.compare_digest(request.headers.get('x-comparison-key', ''), KEY.read_text().strip()):
        raise HTTPException(401, 'Enter the comparison access code.')


@router.get('/models')
def models():
    return {'models': [{'id': key, 'name': name, 'available': True} for key, name in MODELS.items()]}


@router.get('/jobs')
def jobs(request: Request):
    authorize(request)
    records = [json.loads(path.read_text()) for path in (STATE / 'jobs').glob('*/result.json')]
    return sorted(records, key=lambda r: r['created'], reverse=True)[:50]


@router.post('/jobs', status_code=202)
async def create_job(request: Request):
    authorize(request)
    selected = request.headers.get('x-models', '').split(',')
    if not selected or len(set(selected)) != len(selected) or any(m not in MODELS for m in selected):
        raise HTTPException(400, 'Choose supported detectors.')
    extension = request.headers.get('x-audio-extension', '').lower()
    if extension not in {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}:
        raise HTTPException(400, 'Use WAV, FLAC, MP3, OGG or M4A.')
    expected = request.headers.get('x-expected-label', 'unknown')
    if expected not in {'unknown', 'real', 'synthetic'}:
        raise HTTPException(400, 'Expected label must be real, synthetic or unknown.')
    display_name = unquote(request.headers.get('x-audio-name', 'Uploaded audio'))
    display_name = re.sub(r'[\x00-\x1f\x7f]', '', display_name.replace('\\', '/').rsplit('/', 1)[-1])[:120] or 'Uploaded audio'
    with (STATE / 'submission.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(429, 'Another upload is in progress.') from None
        records = [json.loads(p.read_text()) for p in (STATE / 'jobs').glob('*/result.json')]
        if sum(r['status'] in {'queued', 'running'} for r in records) >= 3:
            raise HTTPException(429, 'Queue full; wait for a comparison to finish.')
        if len(records) >= 100:
            raise HTTPException(429, 'Local job limit reached; archive completed results first.')
        job_id = uuid.uuid4().hex
        job = STATE / 'jobs' / job_id
        (job / 'input').mkdir(parents=True)
        # Parent directories remain private to the operator; input alone is mounted.
        filename = 'clip' + extension
        try:
            size = 0
            with (job / 'input' / filename).open('wb') as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 10 * 1024 * 1024:
                        raise HTTPException(413, 'Maximum upload is 10 MiB.')
                    handle.write(chunk)
            if not size:
                raise HTTPException(400, 'Audio file is empty.')
            (job / 'template.tsv').write_text('filename\tcm-score\n' + filename + '\t0\n')
            record = {'id': job_id, 'created': time.time(), 'status': 'queued',
                      'models': selected, 'results': [], 'current': None,
                      'filename': display_name, 'expected_label': expected}
            save(job / 'result.json', record)
            # A detached trusted controller survives Uvicorn reloads. It receives
            # no .env/API keys and never imports model code on this host.
            process = subprocess.Popen(
                [sys.executable, '-m', 'hearsay.benchmark.dashboard', job_id],
                cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
            record['worker_pid'] = process.pid
            # The worker owns result.json once launched; do not overwrite its state.
            return record
        except BaseException:
            shutil.rmtree(job, ignore_errors=True)
            raise
