
# langgraph is used to build the temporary reflection review workflow
from langgraph.graph import StateGraph, START, END

# imports the supported reflection modes
from core.constants import (
    TEXT_REFLECTION,
    VOICE_REFLECTION,
    VIDEO_REFLECTION,
)

# shared temporary state used by the guided check-in workflow
from graphs.graph_state import GuidedCheckinState

# deterministic scoring helpers for text, audio, video and combined results
from modules.scoring import (
    calculate_audio_reflection_score,
    calculate_combined_wellbeing_score,
    calculate_reflection_score,
    calculate_text_reflection_score,
    calculate_video_reflection_score,
    get_combined_wellbeing_indicator,
    get_reflection_indicator,
)

# text-emotion analysis is run only on the reviewed reflection text
from services.text_analysis import analyse_written_reflection

# practical supportive and strain factors are detected separately from scoring
from services.theme_detection import detect_practical_themes


# selects the correct reviewed text depending on the reflection mode
def prepare_analysis_text(state):
    mode = state.get("reflection_mode")

    # written reflections use the original typed text
    if mode == TEXT_REFLECTION:
        analysis_text = str(
            state.get("reflection_text", "") or ""
        ).strip()

    # voice and video use the transcript after the user has reviewed it
    elif mode in (
        VOICE_REFLECTION,
        VIDEO_REFLECTION,
    ):
        analysis_text = str(
            state.get("confirmed_transcript", "") or ""
        ).strip()

    else:
        return {
            "analysis_complete": False,
            "error": (
                "The selected reflection cannot be analysed."
            ),
        }

    # analysis should never run without confirmed reflection text
    if not analysis_text:
        return {
            "analysis_complete": False,
            "error": (
                "There is no confirmed reflection text "
                "available for analysis."
            ),
        }

    return {
        "analysis_text": analysis_text,
        "analysis_complete": False,
        "error": "",
    }


# stops the graph when no valid reviewed text is available
def route_after_text_preparation(state):
    return "stop" if state.get("error") else "continue"


# runs the text-emotion service on the reviewed reflection text
def analyse_review_text(state):
    try:
        # return both display suggestions and the full score distribution
        text_result = analyse_written_reflection(
            typed_reflection=state.get(
                "analysis_text",
                "",
            ),
            maximum_suggestions=3,
        )
    except Exception:
        return {
            "analysis_complete": False,
            "error": (
                "The reflection could not be analysed. "
                "Today's Check-In Status has not been affected."
            ),
        }

    return {
        # cleaned text is reused by the practical-theme detector
        "combined_analysis_text": text_result.get(
            "combined_text",
            "",
        ),

        # these are the smaller suggestions shown during review
        "suggested_text_emotions": text_result.get(
            "emotion_suggestions",
            [],
        ),

        # the full category distribution is kept for reflection scoring
        "all_text_emotion_scores": text_result.get(
            "all_emotion_scores",
            [],
        ),
        "error": "",
    }


# continues only when text analysis completed without an error
def route_after_text_analysis(state):
    return "stop" if state.get("error") else "continue"


# detects practical supportive and strain themes from reviewed text
def detect_review_themes(state):
    combined_text = str(
        state.get(
            "combined_analysis_text",
            "",
        )
        or ""
    ).strip()

    if not combined_text:
        return {
            "analysis_complete": False,
            "error": (
                "The reflection could not be prepared "
                "for practical-theme analysis."
            ),
        }

    try:
        # rule-based detection is separate from model emotion scoring
        theme_result = detect_practical_themes(
            combined_text
        )

    except Exception:
        return {
            "analysis_complete": False,
            "error": (
                "The reflection could not be analysed. "
                "Today's Check-In Status has not been affected."
            ),
        }

    return {
        "suggested_supportive_factors": (
            theme_result.get(
                "supportive_factors",
                [],
            )
        ),
        "suggested_strain_factors": (
            theme_result.get(
                "strain_factors",
                [],
            )
        ),
        "error": "",
    }


# gets audio evidence either directly from state or from earlier media processing
def get_audio_evidence(state):
    # direct evidence takes priority if it has already been supplied
    direct = state.get(
        "suggested_audio_observation"
    )
    if isinstance(direct, dict):
        return direct
    mode = state.get("reflection_mode")
    # voice stores audio evidence inside the voice-processing result
    if mode == VOICE_REFLECTION:
        processing_result = (
            state.get(
                "voice_processing_result",
                {},
            )
            or {}
        )

    # video stores its extracted-audio evidence inside the video result
    elif mode == VIDEO_REFLECTION:
        processing_result = (
            state.get(
                "video_processing_result",
                {},
            )
            or {}
        )
    else:
        return {}
    return (
        processing_result.get(
            "audio_observation",
            {},
        )
        or {}
    )


# gets visible-expression evidence from the video-processing result
def get_video_evidence(state):
    # direct evidence is used first when available
    direct = state.get(
        "suggested_visible_expression"
    )

    if isinstance(direct, dict):
        return direct

    processing_result = (
        state.get(
            "video_processing_result",
            {},
        )
        or {}
    )

    return (
        processing_result.get(
            "visible_expression",
            {},
        )
        or {}
    )


# calculates all deterministic reflection component and combined scores
def calculate_review_scores(state):
    #calculate deterministic component, Reflection and Combined scores.

    #Text:
        #100% Text
    #Voice:
        #50% Text + 50% Audio
    #Video:
        #40% Text + 40% Audio + 20% Video
    ##Combined:
        #70% Reflection + 30% questions when both scores are available.
    

    mode = state.get("reflection_mode")

    # text scoring uses the full category distribution rather than display suggestions
    text_score = calculate_text_reflection_score(
        state.get(
            "all_text_emotion_scores",
            [],
        )
    )

    audio_score = None
    video_score = None

    # audio contributes only to voice and video reflections
    if mode in (
        VOICE_REFLECTION,
        VIDEO_REFLECTION,
    ):
        audio_score = (
            calculate_audio_reflection_score(
                get_audio_evidence(state)
            )
        )

    # visible-expression scoring is used only for video reflections
    if mode == VIDEO_REFLECTION:
        video_score = (
            calculate_video_reflection_score(
                get_video_evidence(state)
            )
        )

    # combine only the components used by the selected reflection mode
    if mode == TEXT_REFLECTION:
        reflection_score = (
            calculate_reflection_score(
                text_score=text_score,
            )
        )

    elif mode == VOICE_REFLECTION:
        reflection_score = (
            calculate_reflection_score(
                text_score=text_score,
                audio_score=audio_score,
            )
        )

    elif mode == VIDEO_REFLECTION:
        reflection_score = (
            calculate_reflection_score(
                text_score=text_score,
                audio_score=audio_score,
                video_score=video_score,
            )
        )

    else:
        reflection_score = None

    # convert the numerical reflection result into cautious user-facing wording
    reflection_indicator = (
        get_reflection_indicator(
            reflection_score
        )
    )

    # question score may be unavailable on non-questionnaire days
    question_score = state.get(
        "question_score"
    )

    # combined score is created only when both question and reflection scores exist
    combined_score = (
        calculate_combined_wellbeing_score(
            reflection_score,
            question_score,
        )
    )

    # convert the internal combined score into the overall indicator
    combined_indicator = (
        get_combined_wellbeing_indicator(
            combined_score
        )
    )

    return {
        "text_component_score": text_score,
        "audio_component_score": audio_score,
        "video_component_score": video_score,
        "reflection_score": reflection_score,
        "reflection_indicator": reflection_indicator,
        "combined_wellbeing_score": combined_score,
        "combined_wellbeing_indicator": combined_indicator,
        "analysis_complete": True,
        "error": "",
    }


# builds the review graph in the order text -> themes -> deterministic scoring
def build_review_graph():
    graph = StateGraph(
        GuidedCheckinState
    )

    # prepares the correct reviewed text for the selected reflection mode
    graph.add_node(
        "prepare_analysis_text",
        prepare_analysis_text,
    )

    # runs text-emotion analysis on the reviewed text
    graph.add_node(
        "analyse_review_text",
        analyse_review_text,
    )

    # detects practical supportive and strain themes
    graph.add_node(
        "detect_review_themes",
        detect_review_themes,
    )

    # calculates the final deterministic multimodal scores
    graph.add_node(
        "calculate_review_scores",
        calculate_review_scores,
    )

    # every review starts by selecting valid confirmed text
    graph.add_edge(
        START,
        "prepare_analysis_text",
    )

    # stop if no valid reflection text is available
    graph.add_conditional_edges(
        "prepare_analysis_text",
        route_after_text_preparation,
        {
            "continue": "analyse_review_text",
            "stop": END,
        },
    )

    # stop if text-emotion analysis fails
    graph.add_conditional_edges(
        "analyse_review_text",
        route_after_text_analysis,
        {
            "continue": "detect_review_themes",
            "stop": END,
        },
    )

    # theme detection is followed by deterministic score calculation
    graph.add_edge(
        "detect_review_themes",
        "calculate_review_scores",
    )

    # review is complete after all component and combined scores are prepared
    graph.add_edge(
        "calculate_review_scores",
        END,
    )

    return graph.compile()


# compile once so the parent guided workflow can reuse the review graph
review_graph = build_review_graph()