
# langgraph is used to build the parent workflow that connects the subgraphs
from langgraph.graph import StateGraph, START, END
# shared state passed between the parent workflow and each guided subgraph
from graphs.graph_state import GuidedCheckinState
# imports the four guided workflow subgraphs
from graphs.guided_checkin_graph import guided_checkin_graph
from graphs.reflection_graph import reflection_graph
from graphs.review_graph import review_graph
from graphs.dashboard_graph import dashboard_graph
# stage names used by flask when calling the parent guided workflow
GUIDED_STAGE_STRUCTURED = "structured"
GUIDED_STAGE_REFLECTION = "reflection"
GUIDED_STAGE_REVIEW = "review"
GUIDED_STAGE_DASHBOARD = "dashboard"
# only these stages are allowed to be routed through the workflow
_VALID_STAGES = {
    GUIDED_STAGE_STRUCTURED,
    GUIDED_STAGE_REFLECTION,
    GUIDED_STAGE_REVIEW,
    GUIDED_STAGE_DASHBOARD,
}


# decides which guided subgraph should handle the current request
def route_guided_stage(state):
    stage = state.get("workflow_stage")
    # unknown or missing stage values are sent to the safe error node
    return stage if stage in _VALID_STAGES else "invalid"

# returns a controlled error when flask requests an unsupported stage
def handle_invalid_stage(state):
    return {
        "error": "The requested guided check-in stage could not be opened."
    }
# builds the parent graph that routes requests into the correct subgraph
def build_guided_workflow_graph():
    graph = StateGraph(GuidedCheckinState)
    # each guided stage is kept as its own subgraph
    nodes = {
        "structured_checkin": guided_checkin_graph,
        "reflection": reflection_graph,
        "review": review_graph,
        "dashboard": dashboard_graph,
        "invalid_stage": handle_invalid_stage,
    }

    # register all subgraphs and the invalid-stage handler
    for name, node in nodes.items():
        graph.add_node(name, node)
    # route directly from the start based on the stage requested by flask
    graph.add_conditional_edges(
        START,
        route_guided_stage,
        {
            "structured": "structured_checkin",
            "reflection": "reflection",
            "review": "review",
            "dashboard": "dashboard",
            "invalid": "invalid_stage",
        },
    )

    # each stage handles one request and then returns control to flask
    for node_name in nodes:
        graph.add_edge(node_name, END)
    return graph.compile()
# compile the parent workflow once so it can be reused for later requests
guided_workflow_graph = build_guided_workflow_graph()