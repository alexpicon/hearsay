# Author: Alex Picon <alexnpc@me.com>
"""Manage the isolated HEARSAY resident detectors."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

NAME = 'hearsay-standalone-ours-resident'
WORKER = Path(__file__).with_name('resident_worker.py').resolve()
ROOT = WORKER.parents[2]
CONFIG = ROOT / 'hearsay/models/accuracy_v2.json'
RUNTIME = WORKER.with_name('v2_runtime.py')
HUB = ROOT / '_internal/runtime/assets/hf/hub'

def mounts_for(variant):
    if variant not in {'ours','ours_v2'}:raise ValueError('Unknown resident variant')
    mounts = [(WORKER, '/resident_worker.py')]
    if variant == 'ours_v2':
        mounts += [(RUNTIME,'/v2_runtime.py'),(CONFIG,'/candidate.json')]
        for name in ['models--Speech-Arena-2025--DF_Arena_500M_V_1','models--facebook--wav2vec2-xls-r-300m']:
            mounts.append((HUB/name,'/assets/hf/hub/'+name))
    return mounts

def worker_digest(variant):
    return hashlib.sha256(b''.join(p.read_bytes() for p,_ in mounts_for(variant) if p.is_file())).hexdigest()

def container_name(variant):
    return NAME if variant == 'ours' else 'hearsay-standalone-v2-resident'


def command(image, worker_hash, variant="ours"):
    extra=[]
    for source,target in mounts_for(variant):
        extra += ['--mount',f'type=bind,src={source},dst={target},readonly']
    if variant == 'ours_v2':
        extra += ['-e','HF_HOME=/assets/hf','-e','HF_MODULES_CACHE=/tmp/hf_modules']
    memory='10g' if variant=='ours_v2' else '6g'
    return ['docker','run','-d','--rm','--name',container_name(variant),'--pull','never',
            '--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','128',
            '--cpus','2','--memory',memory,'--memory-swap',memory,
            '--user',f'{os.getuid()}:{os.getgid()}',
            '--tmpfs','/tmp:rw,nosuid,nodev,size=1073741824',
            '--ulimit','fsize=16777216:16777216','--log-driver','none',
            '--label',f'hearsay.worker={worker_hash}',
            *extra,
            '-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2',
            '-e','OPENBLAS_NUM_THREADS=1','-e','HF_HUB_OFFLINE=1',
            '-e','TRANSFORMERS_OFFLINE=1','-e','HF_HUB_DISABLE_TELEMETRY=1',
            '--entrypoint','timeout',image,'-k','5','900',
            'python','/resident_worker.py']


def validate(info, image, worker_hash, variant="ours"):
    host = info['HostConfig']
    mounts = info['Mounts']
    expected={str(source):target for source,target in mounts_for(variant)}
    memory=(10 if variant=='ours_v2' else 6)*1024**3
    return (info['Image'] == image and info['State']['Running']
            and info['Config'].get('Labels',{}).get('hearsay.worker') == worker_hash
            and host['NetworkMode'] == 'none' and host['ReadonlyRootfs']
            and not host['Privileged'] and not host.get('PortBindings')
            and host.get('PidMode', '') == '' and host.get('IpcMode') == 'private'
            and host['Memory'] == memory and host['NanoCpus'] == 2_000_000_000
            and host['PidsLimit'] == 128 and host['CapDrop'] == ['ALL'] and 'no-new-privileges' in host['SecurityOpt']
            and info['Config']['User'] == f'{os.getuid()}:{os.getgid()}'
            and len([m for m in mounts if m['Type']=='bind']) == len(expected)
            and all(not m['RW'] and m['Source'] in expected and m['Destination']==expected[m['Source']]
                    for m in mounts if m['Type']=='bind'))


def execute(job, image, env, variant="ours"):
    started = time.monotonic()
    name=container_name(variant)
    digest = worker_digest(variant)
    def call(*args, timeout=15):
        return subprocess.check_output(['docker',*args],env=env,stderr=subprocess.DEVNULL,text=True,timeout=timeout)
    def remove():
        subprocess.run(['docker','rm','-f',name],env=env,stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=15)
    try:
        try:
            info = json.loads(call('inspect',name))[0]
        except subprocess.CalledProcessError:
            info = None
        reused = info is not None and validate(info,image,digest,variant)
        if not reused:
            if info is not None:
                remove()
            subprocess.run(command(image,digest,variant),env=env,check=True,stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=30)
        info = json.loads(call('inspect',name))[0]
        if not validate(info,image,digest,variant):
            raise ValueError('Resident isolation verification failed')
        for _ in range(120):
            try:
                call('exec',name,'test','-S','/tmp/hearsay.sock')
                break
            except subprocess.CalledProcessError:
                time.sleep(.25)
        else:
            raise TimeoutError('Model did not become ready')
        audio = next((job/'input').iterdir())
        with audio.open('rb') as handle:
            reply = subprocess.check_output(['docker','exec','-i',name,'python',
                '/resident_worker.py','--client',audio.suffix],stdin=handle,env=env,
                stderr=subprocess.DEVNULL,timeout=140)
        if len(reply)>16384:
            raise ValueError('Oversized result')
        result = json.loads(reply)
        if result.get('error'):
            raise ValueError(result['error'])
        score = float(result['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        return {'model':variant,'status':'complete','score':score,
                'seconds':round(time.monotonic()-started,2), 'image':image,'network':'none',
                'runtime':'resident-v2' if variant=='ours_v2' else 'resident-v1','model_reused':reused,'worker_sha256':digest,
                'inference_seconds':round(result['inference_seconds'],3),
                'duration_seconds':result['duration_seconds'],'analyzed_seconds':result['analyzed_seconds'],'coverage_policy':result.get('coverage_policy')}
    except Exception as error:
        remove()
        return {'model':variant,'status':'failed','error':str(error)[:180],
                'seconds':round(time.monotonic()-started,2),'runtime':'resident-v2' if variant=='ours_v2' else 'resident-v1'}
