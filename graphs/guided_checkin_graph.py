
# langgraph is used to build the structured check-in workflow
from langgraph.graph import StateGraph, START, END
# imports the question keys used to check that all answers are present
from core.constants import (
    CORE_STATUS_QUESTION_KEYS,
    CONTEXT_QUESTION_KEYS,
)
# shared temporary state used by the guided check-in graph
from graphs.graph_state import GuidedCheckinState
# scoring functions used for status, question score and combined result
from modules.scoring import (
    calculate_checkin_status,
    generate_status_breakdown,
    calculate_combined_wellbeing_score,
    get_combined_wellbeing_indicator,
)
# context questions are interpreted separately from the core score
from modules.context_interpretation import (
    interpret_context_answers,
)
# reads the validated numerical Question Score from the status breakdown
def get_question_score(breakdown):
    if not isinstance(breakdown, dict):
        return None

    try:
        score = float(breakdown.get("internal_average"))
    except (TypeError, ValueError):
        return None

    # only scores on the normal 1-5 scale are accepted
    if not 1.0 <= score <= 5.0:
        return None

    return round(score, 2)


# validates all structured answers and calculates the core check-in results
def process_core_answers(state):
    core_answers = state.get("core_answers", {})
    context_answers = state.get("context_answers", {})
    # all four core answers are required before Question Score calculation
    missing_core = [
        key
        for key in CORE_STATUS_QUESTION_KEYS
        if key not in core_answers
    ]

    if missing_core:
        return {
            "error": "One or more core check-in answers are missing."
        }

    # context answers are also checked because this graph handles the full questionnaire
    missing_context = [
        key
        for key in CONTEXT_QUESTION_KEYS
        if key not in context_answers
    ]

    if missing_context:
        return {
            "error": (
                "One or more postpartum context answers are missing."
            )
        }

    try:
        # the four core responses determine Today's Check-In Status
        status = calculate_checkin_status(
            sleep=core_answers["sleep"],
            mood=core_answers["mood"],
            stress=core_answers["stress"],
            support=core_answers["support"],
        )

        # keep the technical breakdown so the Question Score can also be retrieved
        breakdown = generate_status_breakdown(
            sleep=core_answers["sleep"],
            mood=core_answers["mood"],
            stress=core_answers["stress"],
            support=core_answers["support"],
        )

    except ValueError as error:
        return {"error": str(error)}

    # the internal average from the validated breakdown becomes the Question Score
    question_score = get_question_score(breakdown)

    return {
        "checkin_status": status,
        "status_breakdown": breakdown,
        "question_score": question_score,
        "error": "",
    }


# stops the graph when the structured-question validation failed
def route_after_core_answers(state):
    return "stop" if state.get("error") else "continue"


# interprets the six context questions without affecting the Question Score
def process_context_answers(state):
    # context answers only create descriptive summary and appointment information
    context_result = interpret_context_answers(
        state.get("context_answers", {})
    )

    return {
        "context_result": context_result
    }


# creates the internal combined result when both question and reflection scores exist
def calculate_combined_score(state):
    question_score = state.get("question_score")
    reflection_score = state.get("reflection_score")

    # the scoring helper returns None if either component is unavailable
    combined_score = calculate_combined_wellbeing_score(
        reflection_score=reflection_score,
        question_score=question_score,
    )

    # convert the internal value into the user-facing Overall Check-In Indicator
    return {
        "combined_wellbeing_score": combined_score,
        "combined_wellbeing_indicator": (
            get_combined_wellbeing_indicator(combined_score)
        ),
    }


# builds the structured-question subgraph used by the guided workflow
def build_guided_checkin_graph():
    graph = StateGraph(GuidedCheckinState)

    # first node validates the core and context answers and creates the Question Score
    graph.add_node(
        "process_core_answers",
        process_core_answers,
    )

    # second node interprets the six context questions
    graph.add_node(
        "process_context_answers",
        process_context_answers,
    )

    # final node combines the Question Score with an existing Reflection Score
    graph.add_node(
        "calculate_combined_score",
        calculate_combined_score,
    )

    # every structured check-in starts with answer validation and scoring
    graph.add_edge(
        START,
        "process_core_answers",
    )

    # invalid answers stop the graph before context interpretation
    graph.add_conditional_edges(
        "process_core_answers",
        route_after_core_answers,
        {
            "continue": "process_context_answers",
            "stop": END,
        },
    )

    # context interpretation does not alter the already calculated status
    graph.add_edge(
        "process_context_answers",
        "calculate_combined_score",
    )

    # the structured check-in branch ends after the optional combined result
    graph.add_edge(
        "calculate_combined_score",
        END,
    )

    return graph.compile()


# compile the graph once so it can be reused by the parent guided workflow
guided_checkin_graph = build_guided_checkin_graph()