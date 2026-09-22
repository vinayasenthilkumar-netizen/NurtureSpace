
# Any is used for uploaded media objects and TypedDict defines the graph state shape
from typing import Any, TypedDict

# temporary state shared across the guided check-in workflow and its subgraphs
class GuidedCheckinState(TypedDict, total=False):
    # tracks the current stage of the parent workflow
    workflow_stage: str

    # structured question answers and their deterministic interpretation
    core_answers: dict
    context_answers: dict
    checkin_status: str
    status_breakdown: dict
    context_result: dict

    # numerical score created only from the four core questions
    question_score: float

    # reflection mode and temporary user input
    reflection_mode: str
    reflection_text: str
    voice_input: Any
    video_input: Any

    # temporary results returned by voice and video preparation
    voice_processing_result: dict
    video_processing_result: dict

    # tentative observations produced from audio and visible-expression analysis
    suggested_audio_observation: dict
    suggested_visible_expression: dict

    # tracks whether reflection processing finished and whether review can begin
    reflection_processing_complete: bool
    reflection_processing_errors: list
    review_ready: bool

    # transcript and cleaned text used during the review-analysis stage
    confirmed_transcript: str
    analysis_text: str
    combined_analysis_text: str

    # text-emotion suggestions for display and full scores for internal scoring
    suggested_text_emotions: list
    all_text_emotion_scores: list

    # practical factors suggested from the approved reflection text
    suggested_supportive_factors: list
    suggested_strain_factors: list

    analysis_complete: bool

    # separate reflection component scores before they are combined
    text_component_score: float
    audio_component_score: float
    video_component_score: float

    # final reflection score and cautious user-facing indicator
    reflection_score: float
    reflection_indicator: str

    # internal 70/30 combined value and its user-facing indicator
    combined_wellbeing_score: float
    combined_wellbeing_indicator: str

    # temporary dashboard-generation inputs and outputs
    dashboard_application_state: dict
    dashboard_summary_result: dict
    dashboard_appointment_result: dict
    dashboard_summary_draft: str
    dashboard_appointment_points: list
    dashboard_generation_complete: bool

    # shared error message used to stop or route workflow stages
    error: str


# temporary state used by the separate bounded assistant graph
class AssistantState(TypedDict, total=False):
    # current message, previous conversation and detected bounded task
    detected_intent: str
    assistant_chat_history: list
    user_message: str

    # approved application context and personalisation used for retrieval
    user_id: int
    application_state: dict
    relevance_context: list
    relevance_result: dict

    # temporary semantic resource retrieval information
    resource_query: str
    retrieved_resources: list
    resource_retrieval_complete: bool

    # structured output returned by the local llm
    assistant_result: dict

    # urgent safety state checked before normal assistant routing
    urgent_safety_detected: bool
    safety_category: str

    # shared error message for stopping the assistant graph
    error: str