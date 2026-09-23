from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class Example:
    text: str
    audio: np.ndarray
    accel: np.ndarray
    labels: dict[str, Any]
    subject_id: str = ""

TEXT_BY_ROUTE = {
    'safety': [
        'A patient developed a serious adverse reaction after starting the suspect medicine. The physician reporter provided patient age and treatment details.',
        'The patient was hospitalized after taking the product and the reporter supplied the product name and event description.',
    ],
    'quality': [
        'The bottle arrived with a cracked seal and leakage. The batch number is recorded and the customer reports a product defect.',
        'A product quality complaint reports discoloration, broken packaging, and a possible contamination issue.',
    ],
    'medical_information': [
        'Can this medicine be taken with food and is there an interaction with an antacid?',
        'Please provide dosing and administration information for the product.',
    ],
    'general': [
        'The team meeting is moved to Friday afternoon.',
        'Please approve the office supply invoice attached to this message.',
    ],
}

def _audio(route: str, rng: np.random.Generator, n: int = 4000) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / 16000.0
    freq = {'safety':220.0,'quality':440.0,'medical_information':660.0,'general':110.0}[route]
    wave = 0.25 * np.sin(2 * math.pi * freq * t)
    if route == 'safety':
        for start in (900, 2100, 3200): wave[start:start+120] += 0.4 * np.hanning(120)
    if route == 'quality': wave += 0.05 * rng.normal(size=n)
    if route == 'medical_information': wave *= np.linspace(0.3, 0.8, n, dtype=np.float32)
    if route == 'general': wave *= 0.2
    return wave.astype(np.float32)

def _accel(route: str, rng: np.random.Generator, n: int = 96) -> np.ndarray:
    t = np.linspace(0, 6, n, dtype=np.float32)
    freq = {'safety':0.8,'quality':2.2,'medical_information':0.3,'general':1.2}[route]
    base = 0.5 * np.sin(2 * math.pi * freq * t)
    axes = np.stack([base,0.4*np.cos(2*math.pi*freq*t),0.2*base],axis=1)
    if route == 'safety': axes[36:42] += 2.0
    if route == 'quality': axes[:,0] += np.linspace(0,1.0,n)
    axes += 0.03 * rng.normal(size=axes.shape)
    return axes.astype(np.float32)

def build_dataset(size: int = 256, seed: int = 7) -> list[Example]:
    rng = np.random.default_rng(seed)
    routes = list(TEXT_BY_ROUTE)
    labels = {
        'safety': {'icsr':1.0,'pqc':0.0,'mi':0.0,'not_relevant':0.0,'urgency':3.0,'requires_human_review':1.0,'route':'safety','agent_action':'escalate'},
        'quality': {'icsr':0.0,'pqc':1.0,'mi':0.0,'not_relevant':0.0,'urgency':2.0,'requires_human_review':1.0,'route':'quality','agent_action':'review'},
        'medical_information': {'icsr':0.0,'pqc':0.0,'mi':1.0,'not_relevant':0.0,'urgency':1.0,'requires_human_review':1.0,'route':'medical_information','agent_action':'enrich'},
        'general': {'icsr':0.0,'pqc':0.0,'mi':0.0,'not_relevant':1.0,'urgency':0.0,'requires_human_review':1.0,'route':'general','agent_action':'hold'},
    }
    result = []
    for i in range(size):
        route = routes[i % len(routes)]
        text = TEXT_BY_ROUTE[route][int(rng.integers(0, len(TEXT_BY_ROUTE[route])))]
        result.append(Example(text, _audio(route,rng), _accel(route,rng), labels[route].copy(), subject_id=f"subject-{i // 4:04d}"))
    rng.shuffle(result)
    return result

def split_dataset(records: list[Example], train: float = 0.8, val: float = 0.1, seed: int = 7):
    from ml.signal_schema import subject_split
    raw = [
        {'subject_id': item.subject_id, 'index': index, 'example': item}
        for index, item in enumerate(records)
    ]
    groups = subject_split(raw, train=train, val=val, seed=seed)
    return (
        [row['example'] for row in groups['train']],
        [row['example'] for row in groups['validation']],
        [row['example'] for row in groups['test']],
    )

__all__=['Example','build_dataset','split_dataset']