
# dashboard graph module so its llm function can be replaced during testing
import graphs.dashboard_graph as dash_mod
#  media-processing functions can be replaced during testing
import graphs.reflection_graph as ref_mod
# eview graph module so text and theme analysis can be replaced during testing
import graphs.review_graph as review_mod
#  controlled test values and expected outputs
from core.constants import (
    AUDIO_OBSERVATION_ENERGETIC,AUDIO_OBSERVATION_LOW_ENERGY,
    TEXT_EMOTION_LOW_MOOD,TEXT_EMOTION_POSITIVE,
    TEXT_REFLECTION,VIDEO_EXPRESSION_HAPPINESS,
    VIDEO_EXPRESSION_SADNESS,VIDEO_REFLECTION,
    VOICE_REFLECTION,
)

# structured check-in graph used for direct validation tests
from graphs.guided_checkin_graph import guided_checkin_graph

from graphs.reflection_graph import reflection_graph
# imports the parent workflow stages and compiled guided workflow graph
from graphs.guided_workflow_graph import (
    GUIDED_STAGE_DASHBOARD,
    GUIDED_STAGE_REFLECTION,
    GUIDED_STAGE_REVIEW,
    GUIDED_STAGE_STRUCTURED,
    guided_workflow_graph,
)

# provides a fixed set of core slider answers for the tests
def core():
    # these values should produce a question score of 2.25 after stress reversal
    return {
        "sleep": 2,
        "mood": 3,
        "stress": 4,
        "support": 2,
    }

# provides a fixed set of context answers for the tests
def context():
    # context answers are used for interpretation but do not change check-in status
    return {
        "food": 2,
        "medication": 0,
        "physical_recovery": 2,
        "baby_care": 4,
        "other_responsibilities": 1,
        "personal_care": 2,
    }


# provides predictable text-analysis output without loading the real model
def fake_text(typed_reflection, maximum_suggestions=3):
    # return a fixed low-mood result and full emotion score distribution
    return {
        "combined_text": typed_reflection,
        "emotion_suggestions": [
            {
                "category": TEXT_EMOTION_LOW_MOOD,
                "score": 0.8,
            }
        ],
        "all_emotion_scores": [
            {
                "category": TEXT_EMOTION_LOW_MOOD,
                "score": 0.8,
            },
            {
                "category": TEXT_EMOTION_POSITIVE,
                "score": 0.2,
            },
        ],
    }


# provides a fixed practical-theme result for review tests
def fake_themes(text):
    # return one strain factor and no supportive factors
    return {
        "supportive_factors": [],
        "strain_factors": [
            {
                "name": "Sleep difficulty",
                "matched_phrase": "sleep",
            }
        ],
    }


# provides fixed audio scores so scoring tests do not depend on the real audio model
def fake_audio_score():
    # low energy is the dominant observation in this controlled test result
    return {
        "observation": AUDIO_OBSERVATION_LOW_ENERGY,
        "raw_label": "sadness",
        "score": 0.6,
        "clear_result": True,
        "all_scores": [
            {
                "observation": AUDIO_OBSERVATION_LOW_ENERGY,
                "score": 0.6,
            },
            {
                "observation": AUDIO_OBSERVATION_ENERGETIC,
                "score": 0.4,
            },
        ],
        "error": "",
    }


# provides fixed visible-expression scores for video scoring tests
def fake_video_score():
    # sadness is the dominant visibleexpression result in this test
    return {
        "observation": "Downturned-looking expression",
        "dominant_raw_label": VIDEO_EXPRESSION_SADNESS,
        "clear_result": True,
        "enough_face_evidence": True,
        "all_scores": [
            {
                "raw_label": VIDEO_EXPRESSION_HAPPINESS,
                "score": 0.25,
            },
            {
                "raw_label": VIDEO_EXPRESSION_SADNESS,
                "score": 0.75,
            },
        ],
        "error": "",
    }


# checks structured-question scoring, status and context behaviour
def test_structured():
    # run the parent workflow at the structured check-in stage
    result = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_STRUCTURED,
        "core_answers": core(),
        "context_answers": context(),
    })

    # a valid structured check-in should not return an error
    assert result["error"] == ""
    # the fixed core answers should produce the expected status
    assert result["checkin_status"] == (
        "Experiencing significant strain today"
    )

    # context analysis should be returned for valid context answers
    assert result["context_result"] is not None
    # the four core answers should produce a question score of 2.25
    assert result["question_score"] == 2.25
    # the internal status average should match the calculated question score
    assert result["status_breakdown"]["internal_average"] == 2.25
    # the chosen context answers should produce at least one difficulty point
    assert len(
        result["context_result"]["difficulty_points"]
    ) > 0

    # changing only context must not change status
    positive = {
        "food": 5,
        "medication": 5,
        "physical_recovery": 5,
        "baby_care": 5,
        "other_responsibilities": 5,
        "personal_care": 5,
    }

    # run the same core answers with more positive context answers
    result2 = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_STRUCTURED,
        "core_answers": core(),
        "context_answers": positive,
    })

    # status should remain based only on the four core questions
    assert result2["checkin_status"] == result["checkin_status"]
    # reflection completed before questions
    with_reflection = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_STRUCTURED,
        "core_answers": {
            "sleep": 3,
            "mood": 3,
            "stress": 3,
            "support": 3,
        },
        "context_answers": context(),
        "reflection_score": 4.0,
    })

    # equal core values should produce a question score of 3.0
    assert with_reflection["question_score"] == 3.0
    # an existing reflection score should be combined with the question score
    assert with_reflection["combined_wellbeing_score"] == 3.7


# checks that missing or invalid structured answers are rejected
def test_bad_structured():
    # begin with a valid core-answer dictionary
    bad_core = core()
    # remove one required core answer
    bad_core.pop("support")
    # run the structured check-in graph with the incomplete core answers
    result = guided_checkin_graph.invoke({
        "core_answers": bad_core,
        "context_answers": context(),
    })

    # missing a required core answer should produce an error
    assert result["error"] != ""
    # no valid status should be produced after core validation fails
    assert result.get("checkin_status") is None
    # context processing should also stop when the structured input is invalid
    assert result.get("context_result") is None
    # begin with valid context answers
    bad_context = context()
    # remove one required context answer
    bad_context.pop("personal_care")
    # run the graph with incomplete context answers
    result = guided_checkin_graph.invoke({
        "core_answers": core(),
        "context_answers": bad_context,
    })

    # missing context data should produce an error
    assert result["error"] != ""
    # context results should not be produced from incomplete answers
    assert result.get("context_result") is None
    # begin with valid core answers again
    wrong_value = core()
    # replace one slider answer with an out-of-range value
    wrong_value["sleep"] = 0

    # run the graph with the invalid slider value
    result = guided_checkin_graph.invoke({
        "core_answers": wrong_value,
        "context_answers": context(),
    })

    # values outside the allowed scale should be rejected
    assert result["error"] != ""

# checks valid, empty and unsupported text reflection inputs
def test_reflection():
    # run a valid text reflection through the parent workflow
    result = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REFLECTION,
        "reflection_mode": TEXT_REFLECTION,
        "reflection_text": "  I felt tired today.  ",
    })

    # a valid text reflection should be ready for review
    assert result["review_ready"] is True
    # no error should remain after successful processing
    assert result["error"] == ""
    # surrounding spaces should be removed from the reflection text
    assert result["reflection_text"] == "I felt tired today."
    # test an empty text reflection
    empty = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REFLECTION,
        "reflection_mode": TEXT_REFLECTION,
        "reflection_text": "   ",
    })

    # empty reflection text must not continue to review
    assert empty["review_ready"] is False
    # an explanation should be returned for the invalid input
    assert empty["error"] != ""
    # test a reflection mode that is not supported
    invalid = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REFLECTION,
        "reflection_mode": "wrong",
    })
    # an invalid mode must not continue to review
    assert invalid["review_ready"] is False
    # the workflow should report an error for the unsupported mode
    assert invalid["error"] != ""


# checks successful, failed and missing voice reflection processing
def test_voice(monkeypatch):
    # use a simple object as a placeholder for uploaded audio
    audio = object()
    # keep track of calls made to the mocked voice processor
    calls = []

    # creates predictable voice-processing output without using a real audio model
    def fake_voice(audio_file):
        # record the supplied audio input
        calls.append(audio_file)
        # return a successful fixed voice-processing result
        return {
            "processing_complete": True,
            "transcript": "I feel tired today.",
            "transcription_success": True,
            "audio_observation": {
                "observation": "Low energy",
                "raw_label": "sadness",
                "score": 0.8,
                "clear_result": True,
                "error": "",
            },
            "duration_seconds": 5.0,
            "prepared_audio_path": "prepared.wav",
            "temporary_files": [
                "original.wav",
                "prepared.wav",
            ],
            "errors": [],
        }
    # replace the real voice processor with the fixed test version
    monkeypatch.setattr(
        ref_mod,
        "process_voice_reflection",
        fake_voice,
    )
    # invoke the reflection graph using the placeholder audio
    result = reflection_graph.invoke({
        "reflection_mode": VOICE_REFLECTION,
        "voice_input": audio,
    })

    # the voice processor should receive the same audio object once
    assert calls == [audio]
    # successful voice processing should make the result ready for review
    assert result["review_ready"] is True
    # the reflection processing flag should show completion
    assert result["reflection_processing_complete"] is True
    # the processed transcript should be preserved in the graph state
    assert result[
        "voice_processing_result"
    ]["transcript"] == "I feel tired today."

    # failed processing must stop review
    monkeypatch.setattr(
        ref_mod,
        "process_voice_reflection",
        lambda **kwargs: {
            "processing_complete": False,
            "errors": ["Voice processing failed."],
        },
    )

    # run the graph again using a processor that reports failure
    failed = reflection_graph.invoke({
        "reflection_mode": VOICE_REFLECTION,
        "voice_input": audio,
    })

    # failed processing should not become ready for review
    assert failed["review_ready"] is False
    # the processing-complete flag should also remain false
    assert failed["reflection_processing_complete"] is False
    # the returned error should include the processing failure
    assert "Voice processing failed" in failed["error"]
    # missing media must also stop
    missing = reflection_graph.invoke({
        "reflection_mode": VOICE_REFLECTION,
    })
    # voice processing without audio should not continue to review
    assert missing["review_ready"] is False
    # a missing audio input should produce an error
    assert missing["error"] != ""

# checks successful video preparation and multimodal processing
def test_video(monkeypatch):
    # use a simple object as a placeholder for uploaded video
    video = object()
    # count how many times each video-processing stage is called
    calls = {
        "prepare": 0,
        "face": 0,
        "speech": 0,
        "audio": 0,
    }

    # provides predictable media preparation without using a real video file
    def fake_prepare(video_file):
        # record that video preparation was called
        calls["prepare"] += 1
        # make sure the same test video object was passed through
        assert video_file is video
        # return prepared file paths used by the remaining mocked processors
        return {
            "preparation_complete": True,
            "video_path": "video.mp4",
            "extracted_audio_path": "audio.wav",
            "duration_seconds": 12.0,
            "temporary_video_paths": ["video.mp4"],
            "temporary_audio_paths": ["audio.wav"],
            "temporary_frame_paths": [
                "frame1.jpg",
                "frame2.jpg",
                "frame3.jpg",
            ],
            "temporary_face_paths": [
                "face1.jpg",
                "face2.jpg",
                "face3.jpg",
            ],
            "temporary_files": [
                "video.mp4",
                "audio.wav",
                "frame1.jpg",
                "frame2.jpg",
                "frame3.jpg",
                "face1.jpg",
                "face2.jpg",
                "face3.jpg",
            ],
            "errors": [],
        }

    # provides a fixed visible-expression result
    def fake_face(paths):
        # record one visible-expression model call
        calls["face"] += 1

        # return a consistent neutral result across three test frames
        return {
            "observation": "Neutral-looking expression",
            "dominant_raw_label": "Neutral",
            "raw_predictions": [
                "Neutral",
                "Neutral",
                "Neutral",
            ],
            "frame_agreement": 1.0,
            "analysed_face_count": 3,
            "clear_result": True,
            "error": "",
        }

    # provides a fixed speech-to-text result
    def fake_speech(path):
        # record one transcription call
        calls["speech"] += 1

        # return a successful english transcript
        return {
            "transcript": "I feel tired today.",
            "language": "en",
            "success": True,
            "error": "",
        }

    # provides a fixed audio-observation result
    def fake_audio(path):
        # record one audio model call
        calls["audio"] += 1

        # return a low-energy audio observation
        return {
            "observation": "Low energy",
            "raw_label": "sadness",
            "score": 0.8,
            "clear_result": True,
            "error": "",
        }

    # replace real video preparation with controlled test behaviour
    monkeypatch.setattr(
        ref_mod,
        "prepare_video_media",
        fake_prepare,
    )

    # replace the real visible-expression model
    monkeypatch.setattr(
        ref_mod,
        "predict_visible_expression",
        fake_face,
    )
    # replace the real speech-to-text model
    monkeypatch.setattr(
        ref_mod,
        "transcribe_audio",
        fake_speech,
    )
    # replace the real audio-observation model
    monkeypatch.setattr(
        ref_mod,
        "predict_audio_observation",
        fake_audio,
    )

    # run the full video reflection processing stage
    result = reflection_graph.invoke({
        "reflection_mode": VIDEO_REFLECTION,
        "video_input": video,
    })

    # each expected processing component should run exactly once
    assert calls == {
        "prepare": 1,
        "face": 1,
        "speech": 1,
        "audio": 1,
    }

    # successfully processed video should be ready for review
    assert result["review_ready"] is True
    # the reflection processing stage should be marked complete
    assert result["reflection_processing_complete"] is True
    # keep the video-processing result for the following checks
    video_result = result["video_processing_result"]
    # the transcript should be preserved in the video result
    assert video_result["transcript"] == "I feel tired today."
    # the visible-expression observation should match the mocked output
    assert video_result[
        "visible_expression"
    ]["observation"] == "Neutral-looking expression"

    # the audio observation should also match the mocked output
    assert video_result[
        "audio_observation"
    ]["observation"] == "Low energy"

    # text analysis waits for review
    assert video_result["text_emotion_suggestions"] == []
    # practical supportive factors should also wait for review
    assert video_result["supportive_factor_suggestions"] == []
    # practical strain factors should also wait for review
    assert video_result["strain_factor_suggestions"] == []


# checks that later video models do not run when media preparation fails
def test_video_error(monkeypatch):
    # count whether any later multimodal models are called
    calls = {
        "face": 0,
        "speech": 0,
        "audio": 0,
    }

    # force video preparation to fail
    monkeypatch.setattr(
        ref_mod,
        "prepare_video_media",
        lambda video_file: {
            "preparation_complete": False,
            "errors": ["Video validation failed."],
        },
    )

    # records an unexpected visible-expression call
    def no_face(*args, **kwargs):
        calls["face"] += 1

    # records an unexpected transcription call
    def no_speech(*args, **kwargs):
        calls["speech"] += 1

    # records an unexpected audio-model call
    def no_audio(*args, **kwargs):
        calls["audio"] += 1

    # replace the visible-expression function with the call counter
    monkeypatch.setattr(
        ref_mod,
        "predict_visible_expression",
        no_face,
    )
    # replace speech transcription with the call counter
    monkeypatch.setattr(
        ref_mod,
        "transcribe_audio",
        no_speech,
    )

    # replace audio prediction with the call counter
    monkeypatch.setattr(
        ref_mod,
        "predict_audio_observation",
        no_audio,
    )
    # invoke video processing with media preparation forced to fail
    result = reflection_graph.invoke({
        "reflection_mode": VIDEO_REFLECTION,
        "video_input": object(),
    })

    # a failed video should not become ready for review
    assert result["review_ready"] is False
    # the preparation error should be carried into the graph result
    assert "Video validation failed" in result["error"]
    # models must not run after preparation fails
    assert calls == {
        "face": 0,
        "speech": 0,
        "audio": 0,
    }

# checks text, voice and video reflection scoring in the review stage
def test_review(monkeypatch):
    # replace real text analysis with predictable model output
    monkeypatch.setattr(
        review_mod,
        "analyse_written_reflection",
        fake_text,
    )

    # replace practical-theme detection with predictable theme output
    monkeypatch.setattr(
        review_mod,
        "detect_practical_themes",
        fake_themes,
    )

    # use a fixed check-in status throughout the scoring tests
    status = "Experiencing some strain today"
    # use a fixed question score when checking combined wellbeing scoring
    question_score = 3.0
    # text: 100% text
    text = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REVIEW,
        "reflection_mode": TEXT_REFLECTION,
        "reflection_text": "I did not sleep well.",
        "checkin_status": status,
        "question_score": question_score,
    })

    # text analysis should complete successfully
    assert text["analysis_complete"] is True
    # review should not change the existing structured check-in status
    assert text["checkin_status"] == status
    # one text emotion suggestion should be returned
    assert len(text["suggested_text_emotions"]) == 1
    # one strain factor should be returned by the mocked theme detector
    assert len(text["suggested_strain_factors"]) == 1
    # the fixed text probabilities should produce this text component score
    assert text["text_component_score"] == 1.8
    # text-only reflections do not have an audio score
    assert text["audio_component_score"] is None
    # text-only reflections do not have a video score
    assert text["video_component_score"] is None
    # text reflections use the text component as the full reflection score
    assert text["reflection_score"] == 1.8
    # the reflection and question scores should produce the expected combined score
    assert text["combined_wellbeing_score"] == 2.16

    # voice: 50% text + 50% audio
    voice = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REVIEW,
        "reflection_mode": VOICE_REFLECTION,
        "confirmed_transcript": "I am very tired.",
        "checkin_status": status,
        "question_score": question_score,
        "suggested_audio_observation": fake_audio_score(),
    })

    # voice review should complete successfully
    assert voice["analysis_complete"] is True
    # the confirmed transcript should be used as the review analysis text
    assert voice["analysis_text"] == "I am very tired."
    # review should preserve the existing structured status
    assert voice["checkin_status"] == status
    # the text part should use the same fixed text score
    assert voice["text_component_score"] == 1.8
    # the fixed audio probabilities should produce this audio score
    assert voice["audio_component_score"] == 2.6
    # voice reflections do not contain a video component
    assert voice["video_component_score"] is None
    # equal text and audio weighting should produce this reflection score
    assert voice["reflection_score"] == 2.2
    # the voice reflection and question score should produce this combined score
    assert voice["combined_wellbeing_score"] == 2.44

    # video: 40% text + 40% audio + 20% video
    video = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REVIEW,
        "reflection_mode": VIDEO_REFLECTION,
        "confirmed_transcript": "Today was difficult.",
        "checkin_status": status,
        "question_score": question_score,
        "suggested_audio_observation": fake_audio_score(),
        "suggested_visible_expression": fake_video_score(),
    })

    # video review should complete successfully
    assert video["analysis_complete"] is True
    # the confirmed transcript should be used for text analysis
    assert video["analysis_text"] == "Today was difficult."
    # review should preserve the structured check-in status
    assert video["checkin_status"] == status
    # check the calculated text component
    assert video["text_component_score"] == 1.8
    # check the calculated audio component
    assert video["audio_component_score"] == 2.6
    # check the calculated visible-expression component
    assert video["video_component_score"] == 2.0
    # the three weighted components should produce the expected reflection score
    assert video["reflection_score"] == 2.16
    # the reflection and question scores should produce the expected combined score
    assert video["combined_wellbeing_score"] == 2.41
    # voice/video must use confirmed transcript
    empty = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REVIEW,
        "reflection_mode": VOICE_REFLECTION,
        "confirmed_transcript": "",
        "checkin_status": status,
    })

    # review should stop when no confirmed transcript is available
    assert empty["analysis_complete"] is False
    # an error should explain why analysis could not continue
    assert empty["error"] != ""
    # the existing structured status should still remain unchanged
    assert empty["checkin_status"] == status

# checks dashboard generation and safe parent-workflow routing
def test_dashboard(monkeypatch):
    # keep track of which llm tasks are requested
    calls = []

    # provides predictable llm output without calling the real local model
    def fake_llm(
        application_state,
        user_message,
        task="conversation",
        retrieved_resources=None,
    ):
        # record the requested llm task
        calls.append(task)

        # return different fixed content for summary and appointment-point tasks
        return {
            "assistant_reply": "",
            "summary_draft": (
                "Test summary."
                if task == "summary"
                else ""
            ),
            "appointment_points": (
                ["Test appointment point."]
                if task == "appointment_points"
                else []
            ),
            "proposed_dashboard_updates": [],
            "resource_categories": [],
            "needs_user_confirmation": False,
            "urgent_safety_response": False,
        }

    # replace the real llm response generator with the fixed test version
    monkeypatch.setattr(
        dash_mod,
        "generate_llm_response",
        fake_llm,
    )

    # run the dashboard stage with enough application state for generation
    result = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_DASHBOARD,
        "dashboard_application_state": {
            "status": "Experiencing some strain today",
            "core_responses": {
                "sleep": "Poor",
                "mood": "Somewhat low",
            },
            "approved_strain_factors": [
                "Sleep difficulty"
            ],
        },
    })

    # the dashboard should request the summary and appointment-point tasks once each
    assert calls == [
        "summary",
        "appointment_points",
    ]

    # dashboard generation should complete successfully
    assert result["dashboard_generation_complete"] is True
    # the generated summary should match the fixed llm response
    assert result["dashboard_summary_draft"] == "Test summary."
    # the appointment points should also match the fixed response
    assert result["dashboard_appointment_points"] == [
        "Test appointment point."
    ]

    # successful dashboard generation should not leave an error
    assert result["error"] == ""
    # empty dashboard context must stop before llm use
    before = len(calls)
    # run the dashboard stage without the required application context
    missing = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_DASHBOARD,
        "dashboard_application_state": {},
    })

    # generation should stop when there is not enough dashboard context
    assert missing["dashboard_generation_complete"] is False
    # the missing context should produce an error
    assert missing["error"] != ""
    # the llm should not be called when dashboard input validation fails
    assert len(calls) == before
    # unknown parent stage must stop safely
    invalid = guided_workflow_graph.invoke({
        "workflow_stage": "wrong_stage"
    })
    # an unsupported workflow stage should return an error
    assert invalid["error"] != ""