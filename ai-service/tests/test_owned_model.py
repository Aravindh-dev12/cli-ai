import base64

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.owned_model import ClinevoOne, _question_text, text_to_ids
from app.decision_engine import INBOX_DECISION_QUESTIONS
from ml.signal_schema import subject_split


def test_owned_model_typed_multimodal_heads():
    model = ClinevoOne().eval()
    state = text_to_ids("synthetic safety event with patient, reporter and product")
    question = INBOX_DECISION_QUESTIONS["route"]
    audio = torch.randn(1600)
    accel = torch.randn(32, 3)

    with torch.inference_mode():
        choice = model(
            state,
            text_to_ids(_question_text("route", question)),
            "choice",
            choice_count=len(question["criteria"]),
            audio_waveform=audio,
            accel=accel,
        )
        score = model(
            state,
            text_to_ids(_question_text("urgency", INBOX_DECISION_QUESTIONS["urgency"])),
            "score",
            audio_waveform=audio,
            accel=accel,
        )
        noul = model(
            state,
            text_to_ids(_question_text("icsr", INBOX_DECISION_QUESTIONS["icsr"])),
            "noul",
            audio_waveform=audio,
            accel=accel,
        )

    assert choice["choice_logits"].shape == (4,)
    assert score["score_logits"].shape == (4,)
    assert noul["noul_logit"].ndim == 0


def test_predict_many_reuses_one_multimodal_state_encode():
    model = ClinevoOne().eval()
    state = text_to_ids("synthetic quality complaint")
    specs = [
        (
            name,
            text_to_ids(_question_text(name, question)),
            str(question["type"]),
            len(question.get("criteria", {})) if isinstance(question.get("criteria"), dict) else 0,
        )
        for name, question in INBOX_DECISION_QUESTIONS.items()
    ]

    with torch.inference_mode():
        outputs = model.predict_many(
            state,
            specs,
            audio_waveform=torch.zeros(800),
            accel=torch.zeros(24, 3),
        )

    assert [name for name, _ in outputs] == list(INBOX_DECISION_QUESTIONS)
    assert outputs[-1][1]["choice_logits"].shape == (4,)
    assert outputs[4][1]["score_logits"].shape == (4,)


def test_subject_split_has_no_subject_overlap():
    records = [
        {"subject_id": "dog-a", "value": 1},
        {"subject_id": "dog-a", "value": 2},
        {"subject_id": "dog-b", "value": 3},
        {"subject_id": "dog-c", "value": 4},
        {"subject_id": "dog-d", "value": 5},
    ]
    groups = subject_split(records, seed=11)
    subject_sets = [{row["subject_id"] for row in groups[name]} for name in ("train", "validation", "test")]
    assert subject_sets[0].isdisjoint(subject_sets[1])
    assert subject_sets[0].isdisjoint(subject_sets[2])
    assert subject_sets[1].isdisjoint(subject_sets[2])


def test_multimodal_state_changes_when_audio_or_accelerometer_is_present():
    model = ClinevoOne().eval()
    state = text_to_ids("same text")
    with torch.inference_mode():
        text_only = model.encode_state(state)
        audio = model.encode_state(state, audio_waveform=torch.ones(800))
        imu = model.encode_state(state, accel=torch.ones(24, 3))
    assert not torch.allclose(text_only, audio)
    assert not torch.allclose(text_only, imu)


def test_pair_normalizes_single_sample_question_shape():
    model = ClinevoOne().eval()
    state = text_to_ids("shape regression")
    question = text_to_ids(_question_text("route", INBOX_DECISION_QUESTIONS["route"]))
    with torch.inference_mode():
        encoded = model.encode_state(state)
        pair = model._pair(encoded, question)
    assert pair.shape == (512,)


def test_audio_inputs_validate_base64_and_resample_to_model_rate():
    from app.owned_model import _audio_tensor

    source = np.linspace(-0.5, 0.5, 80, dtype=np.float32)
    state = {
        "audio_base64": base64.b64encode(source.tobytes()).decode("ascii"),
        "audio_sample_rate": 8000,
    }
    audio = _audio_tensor(state)
    assert audio.dtype == torch.float32
    assert audio.numel() == 160
    assert float(audio.abs().max()) == pytest.approx(1.0)
