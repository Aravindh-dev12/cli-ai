from __future__ import annotations

import numpy as np

def validate_audio(waveform, sample_rate:int=16000, max_seconds:float=12.0):
    x=np.asarray(waveform,dtype=np.float32).reshape(-1)
    if x.size==0: return x
    if sample_rate<=0 or sample_rate>192000: raise ValueError('unsupported audio sample rate')
    if not np.isfinite(x).all(): raise ValueError('audio contains non-finite values')
    if x.size>int(sample_rate*max_seconds): x=x[:int(sample_rate*max_seconds)]
    peak=float(np.max(np.abs(x)))
    if peak>1e-6: x=x/peak
    return x

def validate_accelerometer(samples, min_axes:int=3, max_samples:int=4096):
    x=np.asarray(samples,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]<min_axes: raise ValueError('accelerometer data must be [time,3+]')
    if not np.isfinite(x).all(): raise ValueError('accelerometer data contains non-finite values')
    return x[:max_samples,:3]

def subject_split(records, subject_key='subject_id', train=0.8, val=0.1, seed=7):
    import random
    subjects=sorted({str(item[subject_key]) for item in records}); rng=random.Random(seed); rng.shuffle(subjects)
    a=int(len(subjects)*train); b=a+int(len(subjects)*val); groups={'train':set(subjects[:a]),'validation':set(subjects[a:b]),'test':set(subjects[b:])}
    return {name:[item for item in records if str(item[subject_key]) in group] for name,group in groups.items()}