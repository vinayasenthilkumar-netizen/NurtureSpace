
# langgraph is used to build the dashboard generation workflow
from langgraph.graph import StateGraph, START, END
# imports the shared guided check-in state used across the workflow
from graphs.graph_state import GuidedCheckinState
# local llm helper used to generate the summary and appointment points
from services.llm_service import generate_llm_response
# checks that approved dashboard information is available before generation starts
def prepare_dashboard_generation(state):
    application_state = state.get("dashboard_application_state", {})
    # generation should not continue without prepared approved check-in data
    if not isinstance(application_state, dict) or not application_state:
        return {
            "dashboard_generation_complete": False,
            "error": (
                "The confirmed check-in information could not be prepared "
                "for summary generation."
            ),
        }

    return {
        "dashboard_generation_complete": False,
        "error": "",
    }


# stops the graph when dashboard preparation returned an error
def route_after_dashboard_preparation(state):
    return "stop" if state.get("error") else "continue"


# creates the temporary personal summary shown for user review
def generate_dashboard_summary(state):
    application_state = state.get("dashboard_application_state", {})

    try:
        # the llm receives only the prepared dashboard application state
        result = generate_llm_response(
            application_state=application_state,
            user_message="Please create my personalised check-in summary.",
            task="summary",
        )

    except Exception as error:
        return {
            "dashboard_summary_result": {},
            "dashboard_summary_draft": "",
            "dashboard_generation_complete": False,
            "error": f"The personal summary could not be generated: {error}",
        }

    # some local-model responses place the summary in assistant_reply
    # instead of the expected summary_draft field
    personal_summary = str(
        result.get("summary_draft")
        or result.get("assistant_reply")
        or ""
    ).strip()

    # an empty llm response is treated as failed summary generation
    if not personal_summary:
        return {
            "dashboard_summary_result": result,
            "dashboard_summary_draft": "",
            "dashboard_generation_complete": False,
            "error": "The personal summary could not be generated.",
        }

    return {
        "dashboard_summary_result": result,
        "dashboard_summary_draft": personal_summary,
        "error": "",
    }


# allows appointment-point generation only after the summary succeeds
def route_after_summary_generation(state):
    return "stop" if state.get("error") else "continue"


# creates simple appointment points when the llm does not return usable ones
def build_grounded_appointment_fallback(application_state):
    points = []

    # use existing context-based appointment suggestions first
    for point in application_state.get("context_appointment_points", []) or []:
        clean_point = str(point or "").strip()

        # avoid adding the same suggestion more than once
        if clean_point and clean_point not in points:
            points.append(clean_point)

        # dashboard displays no more than four points
        if len(points) >= 4:
            return points[:4]

    # approved strain factors are used only when more points are still needed
    for factor in application_state.get("approved_strain_factors", []) or []:
        clean_factor = str(factor or "").strip()

        if not clean_factor:
            continue

        # convert the approved factor into a simple first-person discussion point
        point = (
            f"I have been experiencing {clean_factor.lower()} "
            "and would like to discuss this."
        )

        if point not in points:
            points.append(point)

        if len(points) >= 4:
            break

    return points[:4]


# generates appointment discussion points and falls back to approved data if needed
def generate_dashboard_appointment_points(state):
    application_state = state.get("dashboard_application_state", {})

    try:
        # ask the local llm to produce structured appointment points
        result = generate_llm_response(
            application_state=application_state,
            user_message=(
                "Please create possible appointment discussion points "
                "from my check-in or approved reflection."
            ),
            task="appointment_points",
        )

    except Exception as error:
        # appointment generation can still continue using the fallback
        print("Appointment generation error:", error)
        result = {}

    # clean any points returned by the model and remove empty values
    appointment_points = [
        str(point).strip()
        for point in result.get("appointment_points", []) or []
        if str(point or "").strip()
    ]

    # use only approved context and strain factors when the model gives no points
    if not appointment_points:
        appointment_points = build_grounded_appointment_fallback(
            application_state
        )

    return {
        "dashboard_appointment_result": result,
        "dashboard_appointment_points": appointment_points[:4],
        "dashboard_generation_complete": True,
        "error": "",
    }


# builds the dashboard graph in the order summary -> appointment points
def build_dashboard_graph():
    graph = StateGraph(GuidedCheckinState)

    # register the three dashboard processing nodes
    graph.add_node(
        "prepare_dashboard_generation",
        prepare_dashboard_generation,
    )

    graph.add_node(
        "generate_dashboard_summary",
        generate_dashboard_summary,
    )

    graph.add_node(
        "generate_dashboard_appointment_points",
        generate_dashboard_appointment_points,
    )

    # every dashboard generation request begins with context validation
    graph.add_edge(START, "prepare_dashboard_generation")

    # invalid dashboard context stops before any llm generation
    graph.add_conditional_edges(
        "prepare_dashboard_generation",
        route_after_dashboard_preparation,
        {
            "continue": "generate_dashboard_summary",
            "stop": END,
        },
    )

    # appointment points are generated only after a valid summary is created
    graph.add_conditional_edges(
        "generate_dashboard_summary",
        route_after_summary_generation,
        {
            "continue": "generate_dashboard_appointment_points",
            "stop": END,
        },
    )

    # once appointment points are prepared, dashboard generation is complete
    graph.add_edge("generate_dashboard_appointment_points", END)

    return graph.compile()


# compile the graph once so it can be reused by the flask application
dashboard_graph = build_dashboard_graph()