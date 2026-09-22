# reflection graph module so its processing functions can be replaced during tests
import graphs.reflection_graph as ref_mod
# review graph module so its analysis functions can be replaced during tests
import graphs.review_graph as review_mod
# constants used to build controlled test inputs and expected results
from core.constants import (
    AUDIO_OBSERVATION_CALM, AUDIO_OBSERVATION_LOW_ENERGY,
    TEXT_EMOTION_LOW_MOOD, TEXT_EMOTION_NEUTRAL, TEXT_REFLECTION,
    VIDEO_EXPRESSION_NEUTRAL, VIDEO_EXPRESSION_SADNESS,
    VIDEO_REFLECTION, VOICE_REFLECTION,
)

# imports the compiled reflection workflow used in the tests
from graphs.reflection_graph import reflection_graph

# imports the compiled review workflow used in the tests
from graphs.review_graph import review_graph


# checks that the removed skip reflection mode cannot continue through the workflow
def test_no_skip():
    # try to invoke the reflection graph using the old skip option
    result = reflection_graph.invoke({"reflection_mode": "skip_reflection"})
    # the invalid mode should not be marked as ready for review
    assert result["review_ready"] is False
    # an error message should be returned for the unsupported mode
    assert result["error"] != ""


# checks that text analysis scores are passed correctly through the review graph
def test_review_scores(monkeypatch):
    # create fixed text emotion scores so the test does not depend on the real model
    scores = [
        {"category": TEXT_EMOTION_LOW_MOOD, "score": 0.65},
        {"category": TEXT_EMOTION_NEUTRAL, "score": 0.35},
    ]

    # replace the real text analysis function with predictable test output
    monkeypatch.setattr(review_mod, "analyse_written_reflection", lambda typed_reflection, maximum_suggestions=3: {
        "combined_text": typed_reflection.strip(),
        "emotion_suggestions": [scores[0]],
        "all_emotion_scores": scores,
    })

    # replace practical theme detection so this test focuses only on review scoring
    monkeypatch.setattr(review_mod, "detect_practical_themes", lambda text: {
        "supportive_factors": [], "strain_factors": []
    })

    # run the review graph using a text reflection and known check-in status
    result = review_graph.invoke({
        "reflection_mode": TEXT_REFLECTION,
        "reflection_text": "I felt low today.",
        "checkin_status": "Experiencing some strain today",
    })

    # the review analysis should complete successfully
    assert result["analysis_complete"] is True
    # all supplied emotion scores should be preserved in the graph result
    assert result["all_text_emotion_scores"] == scores
    # the original structured check-in status should remain unchanged
    assert result["checkin_status"] == "Experiencing some strain today"


# checks that all audio observation scores are preserved for a voice reflection
def test_voice_scores(monkeypatch):
    # create fixed audio observation scores for predictable testing
    scores = [
        {"observation": AUDIO_OBSERVATION_LOW_ENERGY, "score": 0.70},
        {"observation": AUDIO_OBSERVATION_CALM, "score": 0.30},
    ]

    # replace real voice processing so no audio file or model is required during the test
    monkeypatch.setattr(ref_mod, "process_voice_reflection", lambda audio_file: {
        "processing_complete": True,
        "transcript": "I feel tired.",
        "transcription_success": True,
        "audio_observation": {
            "observation": AUDIO_OBSERVATION_LOW_ENERGY,
            "raw_label": "sadness", "score": 0.70,
            "clear_result": True, "all_scores": scores, "error": "",
        },
        "duration_seconds": 4.0,
        "prepared_audio_path": "prepared.wav",
        "temporary_files": ["original.wav", "prepared.wav"],
        "errors": [],
    })

    # invoke the voice reflection workflow with a placeholder input object
    result = reflection_graph.invoke({"reflection_mode": VOICE_REFLECTION, "voice_input": object()})
    # successful processing should make the reflection ready for review
    assert result["review_ready"] is True
    # all audio model scores should remain available in the processing result
    assert result["voice_processing_result"]["audio_observation"]["all_scores"] == scores


# checks that video and audio scores are preserved during video reflection processing
def test_video_scores(monkeypatch):
    # create fixed visible expression probabilities for the test
    video_scores = [
        {"raw_label": VIDEO_EXPRESSION_NEUTRAL, "score": 0.60},
        {"raw_label": VIDEO_EXPRESSION_SADNESS, "score": 0.40},
    ]

    # create fixed audio observation probabilities for the video audio track
    audio_scores = [
        {"observation": AUDIO_OBSERVATION_CALM, "score": 0.55},
        {"observation": AUDIO_OBSERVATION_LOW_ENERGY, "score": 0.45},
    ]

    # replace video preparation so the test does not require a real video file
    monkeypatch.setattr(ref_mod, "prepare_video_media", lambda video_file: {
        "preparation_complete": True,
        "video_path": "video.mp4", "extracted_audio_path": "audio.wav",
        "duration_seconds": 8.0,
        "temporary_video_paths": ["video.mp4"],
        "temporary_audio_paths": ["audio.wav"],
        "temporary_frame_paths": ["frame1.jpg", "frame2.jpg"],
        "temporary_face_paths": ["face1.jpg", "face2.jpg"],
        "temporary_files": ["video.mp4", "audio.wav", "frame1.jpg", "frame2.jpg", "face1.jpg", "face2.jpg"],
        "errors": [],
    })

    # replace visible expression prediction with fixed frame-level test evidence
    monkeypatch.setattr(ref_mod, "predict_visible_expression", lambda paths: {
        "observation": "Neutral-looking expression",
        "dominant_raw_label": VIDEO_EXPRESSION_NEUTRAL,
        "raw_predictions": [VIDEO_EXPRESSION_NEUTRAL, VIDEO_EXPRESSION_SADNESS],
        "all_scores": video_scores,
        "frame_agreement": 0.5, "analysed_face_count": 2,
        "dominant_prediction_count": 1, "required_majority_count": 2,
        "minimum_required_face_count": 2, "enough_face_evidence": True,
        "has_strict_majority": False, "evidence_reason": "Test evidence.",
        "clear_result": False, "error": "",
    })

    # replace speech transcription with a successful fixed transcript
    monkeypatch.setattr(ref_mod, "transcribe_audio", lambda path: {
        "transcript": "Today was tiring.", "language": "en", "success": True, "error": ""
    })

    # replace audio emotion prediction with fixed scores
    monkeypatch.setattr(ref_mod, "predict_audio_observation", lambda path: {
        "observation": AUDIO_OBSERVATION_CALM, "raw_label": "neutral",
        "score": 0.55, "clear_result": True, "all_scores": audio_scores, "error": "",
    })

    # run the video reflection graph using a placeholder video object
    result = reflection_graph.invoke({"reflection_mode": VIDEO_REFLECTION, "video_input": object()})
    # keep the video processing section for the remaining assertions
    video = result["video_processing_result"]
    # successful media processing should make the reflection ready for review
    assert result["review_ready"] is True
    # all visible expression probabilities should be preserved
    assert video["visible_expression"]["all_scores"] == video_scores
    # all audio observation probabilities should also be preserved
    assert video["audio_observation"]["all_scores"] == audio_scores
    # text emotion analysis should not run during this reflection processing stage
    assert video["text_emotion_suggestions"] == []
    # supportive factor suggestions should also remain empty at this stage
    assert video["supportive_factor_suggestions"] == []
    # strain factor suggestions should also remain empty at this stage
    assert video["strain_factor_suggestions"] == []