# Author: Alex Picon <alexnpc@me.com>
"""Bounded, offline HEARSAY scoring controller."""
import csv
import fcntl
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '_internal/runtime/dashboard'
MODELS = {'ours': 'HEARSAY Speed', 'ours_v2': 'HEARSAY Accuracy'}
PINNED = {
 'ours': 'sha256:cb967acc260fd97532384c5f32aae2b581056260b0380ef4779bb2e3882220d1',
 'ours_v2': 'sha256:aa60dbf47347e6ead1ec8b1a0832d6b1d92b37021f997c244f2655e3c35333e2',
}
MANIFESTS = {
 'ours': 'sha256:3c8337e5ed349b859a00e22e6ab991099d9cdc07eb1f1b2e96081235ad4f21f0',
 'ours_v2': 'sha256:318121a5d66586a49f3e2975be0ac34532006f75ef3b305b041f3eed39e7fb32',
}
ENV = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp',
       'DOCKER_HOST': 'unix:///var/run/docker.sock', 'DOCKER_CONFIG': str(STATE / 'docker-config')}

def docker(*args, timeout=15):
    return subprocess.check_output(['docker', *args], env=ENV, text=True,
                                   stderr=subprocess.DEVNULL, timeout=timeout)

def pinned_image(model):
    for digest in (PINNED[model], MANIFESTS[model]):
        try:
            actual = docker('image', 'inspect', digest, '--format', '{{.Id}}').strip()
        except subprocess.CalledProcessError:
            continue
        if actual == digest:
            return digest
    raise ValueError('Reviewed image unavailable; no pull permitted')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def parse_score(raw, filename):
    if len(raw) > 16384:
        raise ValueError('Oversized output')
    rows = list(csv.DictReader(io.StringIO(raw), delimiter='\t'))
    if len(rows) != 1 or rows[0].get('filename') != filename:
        raise ValueError('Incomplete or unexpected output IDs')
    score = float(rows[0]['cm-score'])
    if not math.isfinite(score):
        raise ValueError('Nonfinite score')
    return score


def command_for(model, job, name):
    from hearsay.benchmark.resident import command, worker_digest
    return command(pinned_image(model), worker_digest(model), model)


def execute(model, job, job_id):
    from hearsay.benchmark.resident import execute as resident_execute
    return resident_execute(job, pinned_image(model), ENV, model)


def worker(job_id):
    job = STATE / 'jobs' / job_id
    record = json.loads((job / 'result.json').read_text())
    with (STATE / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record['status'] = 'running'
        save(job / 'result.json', record)
        try:
            for model in record['models']:
                record['current'] = model
                save(job / 'result.json', record)
                record['results'].append(execute(model, job, job_id))
                save(job / 'result.json', record)
            record['status'] = 'complete' if all(r['status'] == 'complete' for r in record['results']) else 'completed_with_errors'
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = str(error)[:180]
        finally:
            record['current'] = None
            record['finished'] = time.time()
            save(job / 'result.json', record)
            for path in (job / 'input').iterdir():
                path.unlink()


if __name__ == '__main__':
    worker(sys.argv[1])
