# Author: Alex Picon <alexnpc@me.com>
"""Pinned DF-Arena/XLS-R scoring with a validation-frozen, JSON-only decision head."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from transformers import pipeline

from hearsay.audio import load_audio

REVISION='8258fa8e74ff9b8ad20d4c939c1a7f694a6e4080'


def build():
    config=json.loads(Path('/candidate.json').read_text())
    if config['df_revision']!=REVISION:raise ValueError('Unreviewed backbone revision')
    weights=np.array(config['weights'],dtype=float)
    bias=float(config['intercept']); threshold=float(config['decision_threshold_logit'])
    if weights.shape!=(2,) or not np.isfinite(np.r_[weights,bias,threshold]).all():
        raise ValueError('Invalid decision head')
    local='/assets/hf/hub/models--Speech-Arena-2025--DF_Arena_500M_V_1/snapshots/'+REVISION
    pipe=pipeline('antispoofing',model=local,trust_remote_code=True,device='cpu')
    original=None
    if weights[0]!=0:
        from hearsay.inference import score_file,get_detector
        get_detector(None).ssl_model='/opt/huggingface/hub/models--facebook--wav2vec2-xls-r-300m/snapshots/'+config['xlsr_revision']
        original=score_file
    def score(path,explain=False):
        audio=load_audio(path)
        if len(audio)>120*16000:raise ValueError('Maximum supported duration is 120 seconds')
        width=64600
        if len(audio)<width:
            windows=[np.resize(audio,width)]
        else:
            count=min(3,int(np.ceil(len(audio)/width)))
            windows=[audio[int(i):int(i)+width] for i in np.linspace(0,len(audio)-width,count)]
        margins=[]
        with torch.inference_mode():
            for chunk in windows:
                values=np.asarray(pipe(chunk)['logits']).reshape(-1)
                if values.size!=2 or not np.isfinite(values).all():raise ValueError('Invalid backbone output')
                margins.append(float(values[0]-values[1]))
        features=np.array([0.,np.mean(margins)])
        if original is not None:
            result=original(path,explain=False)
            if result.get('error'):raise ValueError('Original feature path failed')
            eps=config['original_logit_epsilon']
            features[0]=logit(np.clip(result['probability'],eps,1-eps))
        value=float(expit(features@weights+bias-threshold))
        duration=len(audio)/16000
        # This is sampled coverage, not a claim that all intervening audio was read.
        covered=min(duration,3*width/16000)
        if original is not None:covered=min(covered,12.)
        return {'probability':value,'duration_s':duration,'ssl_analyzed_duration_s':covered,
                'coverage_policy':'up to three evenly spaced 4.0375-second windows', 'error':None}
    return score
