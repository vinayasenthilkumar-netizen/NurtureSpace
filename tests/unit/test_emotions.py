# numpy is used to create a simple image array for video-expression testing
import numpy as np
# imports the standard labels expected from the text, audio and video models
from core.constants import (
    AUDIO_OBSERVATION_CALM, AUDIO_OBSERVATION_ENERGETIC,
    AUDIO_OBSERVATION_FRUSTRATED, AUDIO_OBSERVATION_LOW_ENERGY,
    AUDIO_OBSERVATION_UNCLEAR, TEXT_EMOTION_CONNECTION,
    TEXT_EMOTION_LOW_MOOD, TEXT_EMOTION_NEUTRAL, TEXT_EMOTION_POSITIVE,
    VIDEO_EXPRESSION_NEUTRAL, VIDEO_EXPRESSION_SADNESS,
    VIDEO_EXPRESSION_SURPRISE,
)
# imports the model adapters and text-analysis service being tested
from model_adapters import audio_emotion, text_emotion, video_expression
from services import text_analysis


# converts a list of scored results into a simple lookup dictionary
def lookup(items, key):
    return {item[key]: item["score"] for item in items}

# checks text-emotion mapping, suggestions and empty-input handling
def test_text_emotions(monkeypatch):
    # fake model returns fixed emotion labels and scores
    class Model:
        def __call__(self, text):
            return [
                {"label": "joy", "score": 0.40},
                {"label": "gratitude", "score": 0.20},
                {"label": "sadness", "score": 0.15},
                {"label": "neutral", "score": 0.25},
            ]

    # replace the real text model with the fixed test model
    monkeypatch.setattr(text_emotion, "load_text_emotion_model", lambda: Model())

    result = text_emotion.analyse_text_emotions("I had a mixed day.", 2)
    scores = lookup(result["all_scores"], "category")
    assert scores[TEXT_EMOTION_POSITIVE] == 0.40
    assert scores[TEXT_EMOTION_CONNECTION] == 0.20
    assert scores[TEXT_EMOTION_LOW_MOOD] == 0.15
    assert scores[TEXT_EMOTION_NEUTRAL] == 0.25
    # only the two highest-scoring categories should be suggested
    assert [x["category"] for x in result["suggestions"]] == [
        TEXT_EMOTION_POSITIVE, TEXT_EMOTION_CONNECTION
    ]
    # blank reflection text should return no emotion results
    assert text_emotion.analyse_text_emotions("   ") == {
        "all_scores": [], "suggestions": []
    }


# checks that the text-analysis service cleans text and keeps model results
def test_text_service(monkeypatch):
    all_scores = [
        {"category": TEXT_EMOTION_POSITIVE, "score": 0.55},
        {"category": TEXT_EMOTION_LOW_MOOD, "score": 0.45},
    ]
    # replace text-emotion analysis with a predictable result
    monkeypatch.setattr(
        text_analysis,
        "analyse_text_emotions",
        lambda text, maximum_suggestions: {
            "suggestions": [all_scores[0]],
            "all_scores": all_scores
        }
    )
    result = text_analysis.analyse_written_reflection("  Mixed day.  ", 1)
    # surrounding spaces should be removed from the reflection text
    assert result["combined_text"] == "Mixed day."
    assert result["emotion_suggestions"] == [all_scores[0]]
    assert result["all_emotion_scores"] == all_scores


# checks audio-label mapping and confidence handling
def test_audio_emotions(monkeypatch, tmp_path):
    path = tmp_path / "prepared.wav"
    path.write_bytes(b"test")
    # fake model returns fixed audio-emotion probabilities
    class Model:
        def __call__(self, path, top_k=None):
            return [
                {"label": "LABEL_4", "score": 0.45},
                {"label": "LABEL_5", "score": 0.25},
                {"label": "LABEL_0", "score": 0.20},
                {"label": "LABEL_1", "score": 0.10},
            ]

    monkeypatch.setattr(audio_emotion, "load_audio_emotion_model", lambda: Model())
    result = audio_emotion.predict_audio_observation(path)
    scores = lookup(result["all_scores"], "observation")

    assert result["clear_result"] is True
    assert result["observation"] == AUDIO_OBSERVATION_ENERGETIC
    assert scores[AUDIO_OBSERVATION_ENERGETIC] == 0.45
    assert scores[AUDIO_OBSERVATION_CALM] == 0.25
    assert scores[AUDIO_OBSERVATION_LOW_ENERGY] == 0.20
    assert scores[AUDIO_OBSERVATION_FRUSTRATED] == 0.10
    # weak model scores should produce an unclear observation
    class WeakModel:
        def __call__(self, path, top_k=None):
            return [
                {"label": "LABEL_4", "score": 0.35},
                {"label": "LABEL_5", "score": 0.33},
                {"label": "LABEL_0", "score": 0.32},
            ]
    monkeypatch.setattr(
        audio_emotion,
        "load_audio_emotion_model",
        lambda: WeakModel()
    )

    weak = audio_emotion.predict_audio_observation(path)
    assert weak["clear_result"] is False
    assert weak["observation"] == AUDIO_OBSERVATION_UNCLEAR
    assert weak["all_scores"] != []


# checks visible-expression score calculation and majority rules
def test_video_emotions(monkeypatch):
    # build expression probabilities from fixed frame predictions
    scores = video_expression.build_expression_scores([
        "Neutral", "Neutral", "Surprise", "Sadness"
    ])

    values = lookup(scores, "raw_label")
    assert values[VIDEO_EXPRESSION_NEUTRAL] == 0.50
    assert values[VIDEO_EXPRESSION_SURPRISE] == 0.25
    assert values[VIDEO_EXPRESSION_SADNESS] == 0.25
    assert round(sum(values.values()), 4) == 1.0

    # create a simple image so real face files are not needed
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    monkeypatch.setattr(video_expression, "MIN_VIDEO_EXPRESSION_FACES", 4)
    monkeypatch.setattr(
        video_expression,
        "load_face_crop",
        lambda path: image.copy()
    )

    # return an even split between neutral and surprise
    class Model:
        def predict_multi_emotions(self, images, logits=False):
            return ["Neutral", "Neutral", "Surprise", "Surprise"], None

    monkeypatch.setattr(
        video_expression,
        "load_video_expression_model",
        lambda: Model()
    )
    result = video_expression.predict_visible_expression(
        ["f1", "f2", "f3", "f4"]
    )
    # enough faces are present but there is no strict majority
    assert result["enough_face_evidence"] is True
    assert result["has_strict_majority"] is False
    assert result["clear_result"] is False
    assert result["all_scores"] != []