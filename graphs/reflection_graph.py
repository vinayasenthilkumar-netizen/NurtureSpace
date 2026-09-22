
# langgraph is used to build and connect the reflection-processing stages
from langgraph.graph import StateGraph, START, END
# imports the three supported reflection modes
from core.constants import (
    TEXT_REFLECTION,
    VOICE_REFLECTION,
    VIDEO_REFLECTION,
)
# shared temporary state used by the guided check-in workflow
from graphs.graph_state import GuidedCheckinState
# voice reflections are handled through the voice-processing service
from services.voice_processing import process_voice_reflection
# video preparation creates temporary media files before model analysis
from services.video_processing import prepare_video_media, add_processing_error
# whisper transcription is used for voice extracted from video
from model_adapters.speech_to_text import transcribe_audio
# audio observation model and its safe fallback result
from model_adapters.audio_emotion import (
    predict_audio_observation,
    create_unclear_result,
)
# visible-expression model and its safe fallback result
from model_adapters.video_expression import (
    predict_visible_expression,
    create_empty_expression_result,
)

# returns a copy so video-processing state can be updated safely by each node
def _video_result(state):
    return dict(state.get("video_processing_result", {}) or {})


# gets the temporary audio file extracted from the video
def _video_audio_path(video_result):
    return str(video_result.get("extracted_audio_path", "") or "").strip()


# adds one processing/model error to the shared video error list
def _add_result_error(video_result, result):
    errors = list(video_result.get("errors", []) or [])
    # the helper ignores empty or duplicate error messages
    add_processing_error(
        errors,
        result.get("error", ""),
    )
    video_result["errors"] = errors


# clears routing flags before choosing the reflection pathway
def prepare_reflection_route(state):
    return {
        "review_ready": False,
        "error": "",
    }


# sends the workflow to the selected text, voice or video branch
def route_reflection_mode(state):
    mode = state.get("reflection_mode")

    routes = {
        TEXT_REFLECTION: "text",
        VOICE_REFLECTION: "voice",
        VIDEO_REFLECTION: "video",
    }

    # any unsupported mode is routed to the safe invalid handler
    return routes.get(mode, "invalid")


# validates written reflection text before sending it to Review
def prepare_text_reflection(state):
    reflection_text = str(
        state.get("reflection_text", "") or ""
    ).strip()
    # written reflection is compulsory for the text route
    if not reflection_text:
        return {
            "reflection_text": "",
            "review_ready": False,
            "error": "Please enter a written reflection before continuing.",
        }
    # no model analysis is run here; Review handles text analysis later
    return {
        "reflection_text": reflection_text,
        "review_ready": True,
        "error": "",
    }

# prepares and analyses a voice reflection before transcript review
def run_voice_reflection_processing(state):
    voice_input = state.get("voice_input")
    # voice reflection cannot continue without a recording or upload
    if voice_input is None:
        error_message = (
            "Please record or upload a voice reflection before continuing."
        )
        return {
            "voice_processing_result": {},
            "reflection_processing_complete": False,
            "reflection_processing_errors": [error_message],
            "review_ready": False,
            "error": error_message,
        }

    try:
        # service prepares temporary audio, transcribes it and analyses vocal pattern
        voice_result = process_voice_reflection(
            audio_file=voice_input
        )

    except Exception as error:
        error_message = f"The voice reflection could not be processed: {error}"

        return {
            "voice_processing_result": {},
            "reflection_processing_complete": False,
            "reflection_processing_errors": [error_message],
            "review_ready": False,
            "error": error_message,
        }

    processing_complete = bool(
        voice_result.get("processing_complete", False)
    )

    processing_errors = list(
        voice_result.get("errors", []) or []
    )
    # failed processing should stop before the user reaches Review
    if not processing_complete:
        error_message = (
            str(processing_errors[0])
            if processing_errors
            else "The voice reflection could not be processed."
        )
        return {
            "voice_processing_result": voice_result,
            "reflection_processing_complete": False,
            "reflection_processing_errors": processing_errors,
            "review_ready": False,
            "error": error_message,
        }
    # successful voice processing can move to transcript review
    return {
        "voice_processing_result": voice_result,
        "reflection_processing_complete": True,
        "reflection_processing_errors": processing_errors,
        "review_ready": True,
        "error": "",
    }

# prepares temporary files required for the video analysis stages
def prepare_video_reflection(state):
    video_input = state.get("video_input")
    if video_input is None:
        return {
            "video_processing_result": {},
            "review_ready": False,
            "error": (
                "Please record or upload a video reflection before continuing."
            ),
        }
    try:
        # this prepares the video and extracts temporary audio, frames and face crops
        video_result = prepare_video_media(video_input)

    except Exception as error:
        return {
            "video_processing_result": {},
            "review_ready": False,
            "error": f"The video reflection could not be prepared: {error}",
        }

    # preparation must succeed before any of the model stages run
    if not video_result.get("preparation_complete", False):
        errors = video_result.get("errors", [])

        return {
            "video_processing_result": video_result,
            "review_ready": False,
            "error": (
                str(errors[0])
                if errors
                else "The video reflection could not be prepared."
            ),
        }

    return {
        "video_processing_result": video_result,
        "review_ready": False,
        "error": "",
    }


# runs visible-expression analysis on the temporary face crops
def analyse_video_expression(state):
    video_result = _video_result(state)
    face_paths = list(
        video_result.get("temporary_face_paths", []) or []
    )
    if face_paths:
        try:
            # all usable face crops are passed together to the expression adapter
            visible_expression = predict_visible_expression(
                face_paths
            )

        except Exception as error:
            visible_expression = create_empty_expression_result(
                f"Visible-expression analysis failed: {error}"
            )

    else:
        # no usable face evidence produces an unclear result rather than failing the graph
        visible_expression = create_empty_expression_result(
            "No clear face was found in the sampled frames."
        )

    video_result["visible_expression"] = visible_expression

    # preserve any expression-analysis error for later display/debugging
    _add_result_error(
        video_result,
        visible_expression,
    )

    return {
        "video_processing_result": video_result
    }


# transcribes the audio extracted from the video using whisper
def transcribe_video_audio(state):
    video_result = _video_result(state)
    audio_path = _video_audio_path(video_result)

    if audio_path:
        try:
            transcription_result = transcribe_audio(
                audio_path
            )

        except Exception as error:
            transcription_result = {
                "transcript": "",
                "language": "en",
                "success": False,
                "error": f"Transcription failed: {error}",
            }

    else:
        transcription_result = {
            "transcript": "",
            "language": "en",
            "success": False,
            "error": "No audio was available for transcription.",
        }

    # store only the temporary transcript and if whisper succeeded
    video_result["transcript"] = str(
        transcription_result.get("transcript", "") or ""
    ).strip()

    video_result["transcription_success"] = bool(
        transcription_result.get("success", False)
    )

    _add_result_error(
        video_result,
        transcription_result,
    )

    return {
        "video_processing_result": video_result
    }


# analyses the vocal pattern in the same extracted video audio
def analyse_video_audio_pattern(state):
    video_result = _video_result(state)
    audio_path = _video_audio_path(video_result)

    if audio_path:
        try:
            audio_observation = predict_audio_observation(
                audio_path
            )

        except Exception as error:
            audio_observation = create_unclear_result(
                f"Audio analysis failed: {error}"
            )

    else:
        # missing video audio is represented as an unclear observation
        audio_observation = create_unclear_result(
            "No audio was available for vocal-pattern analysis."
        )

    video_result["audio_observation"] = audio_observation

    _add_result_error(
        video_result,
        audio_observation,
    )

    return {
        "video_processing_result": video_result
    }


# marks video processing complete before the result is passed to Review
def finalise_video_reflection(state):
    video_result = _video_result(state)

    # text-based suggestions are intentionally deferred until transcript confirmation
    video_result.update(
        {
            "processing_complete": True,
            "text_emotion_suggestions": [],
            "supportive_factor_suggestions": [],
            "strain_factor_suggestions": [],
            "theme_found": False,
        }
    )

    # preparation flag is no longer needed once all video stages have completed
    video_result.pop(
        "preparation_complete",
        None,
    )

    return {
        "video_processing_result": video_result,
        "reflection_processing_complete": True,
        "reflection_processing_errors": list(
            video_result.get("errors", []) or []
        ),
        "review_ready": True,
        "error": "",
    }


# safely handles any unsupported reflection mode
def handle_invalid_reflection(state):
    return {
        "review_ready": False,
        "error": "The selected reflection method could not be opened.",
    }


# stops the video branch when temporary media preparation failed
def route_after_video_preparation(state):
    return "stop" if state.get("error") else "continue"


# builds the reflection graph and connects the text, voice and video branches
def build_reflection_graph():
    graph = StateGraph(GuidedCheckinState)

    # each node handles one clear part of reflection processing
    nodes = {
        "prepare_reflection_route": prepare_reflection_route,
        "prepare_text_reflection": prepare_text_reflection,
        "process_voice_reflection": run_voice_reflection_processing,
        "prepare_video_reflection": prepare_video_reflection,
        "analyse_video_expression": analyse_video_expression,
        "transcribe_video_audio": transcribe_video_audio,
        "analyse_video_audio_pattern": analyse_video_audio_pattern,
        "finalise_video_reflection": finalise_video_reflection,
        "handle_invalid_reflection": handle_invalid_reflection,
    }

    # register all reflection-processing nodes
    for name, node in nodes.items():
        graph.add_node(name, node)

    # every reflection request first clears the routing state
    graph.add_edge(
        START,
        "prepare_reflection_route",
    )

    # choose the branch based on the user's selected reflection mode
    graph.add_conditional_edges(
        "prepare_reflection_route",
        route_reflection_mode,
        {
            "text": "prepare_text_reflection",
            "voice": "process_voice_reflection",
            "video": "prepare_video_reflection",
            "invalid": "handle_invalid_reflection",
        },
    )

    # text and voice complete their work in a single branch node
    graph.add_edge(
        "prepare_text_reflection",
        END,
    )

    graph.add_edge(
        "process_voice_reflection",
        END,
    )

    # video analysis only starts when temporary preparation was successful
    graph.add_conditional_edges(
        "prepare_video_reflection",
        route_after_video_preparation,
        {
            "continue": "analyse_video_expression",
            "stop": END,
        },
    )

    # video uses separate stages so each modality can fail independently
    graph.add_edge(
        "analyse_video_expression",
        "transcribe_video_audio",
    )

    graph.add_edge(
        "transcribe_video_audio",
        "analyse_video_audio_pattern",
    )

    graph.add_edge(
        "analyse_video_audio_pattern",
        "finalise_video_reflection",
    )

    # finalised video results are now ready for the Review graph
    graph.add_edge(
        "finalise_video_reflection",
        END,
    )

    graph.add_edge(
        "handle_invalid_reflection",
        END,
    )

    return graph.compile()


# compile once so the parent guided workflow can reuse the reflection graph
reflection_graph = build_reflection_graph()