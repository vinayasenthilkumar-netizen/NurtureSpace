# langgraph is used to build and connect the assistant workflow nodes
from langgraph.graph import StateGraph, START, END

# imports the shared state used while one assistant request moves through the graph
from graphs.graph_state import AssistantState

# personal pattern information can be added to resource retrieval
from rag.personalisation import get_relevance_context, build_personalised_query

# main semantic resource retriever
from rag.retriever import retrieve_resources

# provides the cached embedding model used by resource retrieval
from services.assistant_service import get_embedding_model

# local llm helpers for intent, safety checking and response generation
from services.llm_service import (
    classify_assistant_intent,
    classify_urgent_safety,
    generate_llm_response,
)

# database functions used to retrieve curated resources and saved-resource ids
from database.resources import (
    get_active_resources,
    get_bookmarked_resource_ids,
)


# only these intents are allowed to control assistant routing
ALLOWED_AGENT_INTENTS = {
    "conversation",
    "resources",
    "summary",
    "appointment_points",
}

# resource types are grouped so requests such as "something to listen to" can be filtered
LISTENING_TYPES = {"podcast", "audio"}
VIDEO_TYPES = {"video"}
READING_TYPES = {"article", "guide", "blog", "webpage", "website"}


# gets short personal pattern phrases that may help retrieval or conversation
def _relevance_context(state):
    return get_relevance_context(
        relevance_result=state.get("relevance_result", {}) or {},
        include_supporting=True,
    )


# converts a resource id into a consistent string value
def _resource_id(resource):
    return str(resource.get("id", ""))


# normalises a resource title for comparison with previously shown resources
def _resource_title(resource):
    return str(resource.get("title", "") or "").strip().lower()


# returns a normalised resource type such as article, video or podcast
def _resource_type(resource):
    return str(resource.get("resource_type", "") or "").strip().lower()


# groups similar resource types into reading, listening or video families
def _resource_family(resource):
    resource_type = _resource_type(resource)

    if resource_type in LISTENING_TYPES:
        return "listening"

    if resource_type in VIDEO_TYPES:
        return "video"

    if resource_type in READING_TYPES:
        return "reading"

    return resource_type or "other"


# checks whether the user specifically requested listening, video or reading content
def _requested_media_group(user_message):
    text = str(user_message or "").strip().lower()

    # include a few natural phrases and common spelling variations
    listening_terms = (
        "podcast",
        "podacst",
        "pod cast",
        "audio resource",
        "audio resources",
        "something to listen",
        "resources to listen",
        "resource to listen",
        "listen to",
        "listening resource",
    )

    video_terms = (
        "video",
        "something to watch",
        "resources to watch",
        "resource to watch",
        "watching resource",
    )

    reading_terms = (
        "article",
        "something to read",
        "resources to read",
        "resource to read",
        "guide to read",
    )

    if any(term in text for term in listening_terms):
        return "listening"

    if any(term in text for term in video_terms):
        return "video"

    if any(term in text for term in reading_terms):
        return "reading"

    return ""


# quickly detects obvious resource requests before using the normal intent classifier
def _looks_like_resource_request(user_message):
    text = str(user_message or "").strip().lower()

    direct_terms = (
        "resource",
        "resources",
        "podcast",
        "podacst",
        "pod cast",
        "something to listen",
        "something to watch",
        "something to read",
        "find me an article",
        "show me an article",
        "recommend an article",
        "find me a video",
        "show me a video",
        "recommend a video",
    )

    return any(term in text for term in direct_terms)


# reads assistant history to find resource titles that were recently displayed
def _recently_shown_resource_titles(chat_history):
    titles = set()

    for message in chat_history or []:
        # only assistant messages can contain the resource tracking line
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue

        for line in str(message.get("content", "") or "").splitlines():
            clean_line = line.strip()

            if not clean_line.startswith("Resources shown:"):
                continue

            title_text = clean_line.replace("Resources shown:", "", 1).strip()

            # multiple titles are stored as a comma-separated list
            for title in title_text.split(","):
                clean_title = title.strip().lower()

                if clean_title:
                    titles.add(clean_title)

    return titles


# keeps only resources that match an explicitly requested media family
def _filter_resources_for_media(resources, media_group):
    # no media preference means the full curated collection can be considered
    if not media_group:
        return list(resources)

    if media_group == "listening":
        allowed_types = LISTENING_TYPES

    elif media_group == "video":
        allowed_types = VIDEO_TYPES

    else:
        allowed_types = READING_TYPES

    return [
        resource
        for resource in resources
        if _resource_type(resource) in allowed_types
    ]


# moves unsaved and recently unseen resources ahead of repeated ones
def _prioritise_saved_and_recent(resources, state):
    bookmarked_ids = set()
    user_id = state.get("user_id")

    # bookmarked ids are used only for ordering, not for excluding resources
    if user_id:
        try:
            bookmarked_ids = {
                str(resource_id)
                for resource_id in get_bookmarked_resource_ids(user_id)
            }

        except Exception:
            # resource retrieval can continue even if bookmarks cannot be read
            bookmarked_ids = set()

    # place resources the user has not already saved before bookmarked resources
    unsaved = [
        resource
        for resource in resources
        if _resource_id(resource) not in bookmarked_ids
    ]

    saved = [
        resource
        for resource in resources
        if _resource_id(resource) in bookmarked_ids
    ]

    preferred = unsaved + saved

    # titles from recent assistant messages are used to reduce repetition
    recently_shown = _recently_shown_resource_titles(
        state.get("assistant_chat_history", [])
    )

    fresh = [
        resource
        for resource in preferred
        if _resource_title(resource) not in recently_shown
    ]

    repeated = [
        resource
        for resource in preferred
        if _resource_title(resource) in recently_shown
    ]

    return fresh + repeated


# tries to return different media families for a general resource request
def _select_diverse_resources(resources, top_k=3):
    selected = []
    used_ids = set()
    used_families = set()

    # first pass prefers one resource from each available media family
    for resource in resources:
        resource_id = _resource_id(resource)
        family = _resource_family(resource)

        if resource_id in used_ids or family in used_families:
            continue

        selected.append(resource)
        used_ids.add(resource_id)
        used_families.add(family)

        if len(selected) >= top_k:
            return selected

    # second pass fills any remaining places without repeating resource ids
    for resource in resources:
        resource_id = _resource_id(resource)

        if resource_id in used_ids:
            continue

        selected.append(resource)
        used_ids.add(resource_id)

        if len(selected) >= top_k:
            break

    return selected


# checks that the user supplied a message before the graph continues
def validate_assistant_request(state):
    user_message = str(state.get("user_message", "") or "").strip()

    if not user_message:
        return {"error": "Please enter a message before continuing."}

    return {"error": ""}


# checks for urgent harm-related content before normal assistant routing
def check_urgent_safety(state):
    safety_result = classify_urgent_safety(
        user_message=state.get("user_message", ""),
        chat_history=state.get("assistant_chat_history", []),
    )

    return {
        "urgent_safety_detected": bool(
            safety_result.get("urgent_safety_detected", False)
        ),
        "safety_category": str(
            safety_result.get("safety_category", "none") or "none"
        ),
    }


# returns a fixed safety response instead of continuing through normal llm tasks
def run_urgent_safety_response(state):
    category = state.get("safety_category", "none")

    # wording changes slightly depending on who may be at immediate risk
    if category == "baby":
        reply = (
            "What you described could involve an immediate risk of harm "
            "to your baby. Please move away from the situation if you can "
            "do so safely, contact local emergency services now, and ask "
            "a trusted person to stay with you and your baby."
        )

    elif category == "other_person":
        reply = (
            "What you described could involve an immediate risk of harm "
            "to another person. Please create distance from the situation "
            "if you can do so safely, contact local emergency services now, "
            "and reach out to a trusted person for immediate support."
        )

    else:
        reply = (
            "What you described could involve an immediate risk of harm "
            "to yourself. Please contact local emergency services now and "
            "reach out to a trusted person who can stay with you or support you."
        )

    # normal resource and drafting outputs are disabled for this route
    return {
        "assistant_result": {
            "assistant_reply": reply,
            "summary_draft": "",
            "appointment_points": [],
            "proposed_dashboard_updates": [],
            "resource_categories": [],
            "needs_user_confirmation": False,
            "urgent_safety_response": True,
        },
        "retrieved_resources": [],
        "error": "",
    }


# sends urgent cases directly to the fixed safety response
def route_after_safety_check(state):
    return "urgent" if state.get("urgent_safety_detected", False) else "normal"


# decides which bounded assistant task should handle the current message
def interpret_user_intent(state):
    user_message = state.get("user_message", "")

    # direct resource wording is handled before the general intent classifier
    if _looks_like_resource_request(user_message):
        return {"detected_intent": "resources"}

    intent = classify_assistant_intent(
        user_message=user_message,
        chat_history=state.get("assistant_chat_history", []),
    )

    # unknown intents fall back to ordinary conversation
    if intent not in ALLOWED_AGENT_INTENTS:
        intent = "conversation"

    return {"detected_intent": intent}


# stops the graph immediately when request validation produced an error
def route_after_validation(state):
    return "stop" if state.get("error") else "continue"


# routes the recognised intent to one of the assistant's allowed task branches
def route_detected_intent(state):
    intent = state.get("detected_intent", "conversation")

    if intent in {"summary", "appointment_points", "resources"}:
        return intent

    return "conversation"


# runs normal bounded conversation through the local llm
def run_conversation(state):
    application_state = dict(state.get("application_state", {}) or {})
    relevance_context = _relevance_context(state)

    # relevant approved personal patterns can be supplied as extra context
    if relevance_context:
        application_state["personal_pattern_relevance"] = relevance_context

    result = generate_llm_response(
        application_state=application_state,
        user_message=state.get("user_message", ""),
        task="conversation",
        chat_history=state.get("assistant_chat_history", []),
    )

    return {
        "assistant_result": result,
        "retrieved_resources": [],
        "relevance_context": relevance_context,
    }


# prepares a semantic search query using the user's message and relevant patterns
def prepare_resource_query(state):
    user_message = str(state.get("user_message", "") or "").strip()
    relevance_context = _relevance_context(state)

    # combine the current request with selected personal pattern context
    resource_query = build_personalised_query(
        user_query=user_message,
        relevant_patterns=relevance_context,
    )

    if not resource_query:
        return {
            "relevance_context": relevance_context,
            "resource_query": "",
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": "A resource-search query could not be prepared.",
        }

    return {
        "relevance_context": relevance_context,
        "resource_query": resource_query,
        "resource_retrieval_complete": False,
        "retrieved_resources": [],
        "error": "",
    }


# retrieves and ranks curated resources using the application's embedding model
def run_minilm_resource_retrieval(state):
    resource_query = str(state.get("resource_query", "") or "").strip()

    if not resource_query:
        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": "The resource-search query was empty.",
        }

    # load only active resources from the curated database collection
    try:
        resources = get_active_resources()

    except Exception as error:
        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": (
                "The curated resource collection could not be loaded "
                f"from the database: {error}"
            ),
        }

    if not resources:
        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": "No active curated resources are currently available.",
        }

    # retrieve the cached embedding model before semantic search begins
    try:
        embedding_model = get_embedding_model()

    except Exception as error:
        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": f"The semantic resource model could not be loaded: {error}",
        }

    # explicit media requests search only the matching resource types
    media_group = _requested_media_group(state.get("user_message", ""))
    candidate_resources = _filter_resources_for_media(resources, media_group)

    if media_group and not candidate_resources:
        media_label = {
            "listening": "podcast or listening",
            "video": "video",
            "reading": "reading",
        }.get(media_group, "requested")

        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": (
                f"No active curated {media_label} resources are currently available."
            ),
        }

    try:
        # general requests rank the full collection so different media types
        # can later be selected without running several embedding searches
        retrieval_count = (
            len(candidate_resources)
            if not media_group
            else min(8, len(candidate_resources))
        )

        # semantic similarity ranks the candidate resources for this query
        ranked_resources = retrieve_resources(
            query=resource_query,
            resources=candidate_resources,
            embedding_model=embedding_model,
            top_k=retrieval_count,
        )

    except Exception as error:
        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": f"Supportive resources could not be retrieved: {error}",
        }

    if not ranked_resources:
        return {
            "resource_retrieval_complete": False,
            "retrieved_resources": [],
            "error": "No matching curated resources could be retrieved.",
        }

    # prefer resources that are unsaved and have not recently been shown
    preferred = _prioritise_saved_and_recent(ranked_resources, state)

    # explicit media requests keep the top three from that media group
    if media_group:
        retrieved_resources = preferred[:3]

    # general requests try to give a more varied set of resource formats
    else:
        retrieved_resources = _select_diverse_resources(preferred, top_k=3)

    return {
        "resource_retrieval_complete": True,
        "retrieved_resources": retrieved_resources,
        "error": "",
    }


# prevents grounded generation unless retrieval finished successfully
def route_after_resource_retrieval(state):
    if state.get("error"):
        return "stop"

    if not state.get("resource_retrieval_complete", False):
        return "stop"

    if not state.get("retrieved_resources", []):
        return "stop"

    return "generate"


# adds a hidden instruction telling the llm that resource cards are already provided
def _grounded_resource_user_message(state, retrieved_resources):
    original = str(state.get("user_message", "") or "").strip()
    media_group = _requested_media_group(original)

    # this keeps the llm response short and avoids repeating resource cards in text
    if media_group == "listening":
        note = (
            "The application has already retrieved matching curated podcast "
            "or listening resource cards. Briefly introduce the cards. Do not "
            "say that you cannot provide podcasts and do not list the cards."
        )

    elif media_group == "video":
        note = (
            "The application has already retrieved matching curated video "
            "resource cards. Briefly introduce them without listing them."
        )

    elif media_group == "reading":
        note = (
            "The application has already retrieved matching curated reading "
            "resource cards. Briefly introduce them without listing them."
        )

    else:
        # for general requests, mention when the returned set includes listening content
        families = {
            _resource_family(resource)
            for resource in retrieved_resources
        }

        if "listening" in families:
            note = (
                "The application has already retrieved a mixed set of curated "
                "resource cards, including a listening resource. Briefly "
                "introduce the cards without listing them."
            )

        else:
            note = (
                "The application has already retrieved matching curated "
                "resource cards. Briefly introduce them without listing them."
            )

    return f"{original}\n\nApplication instruction: {note}"


# generates a response grounded only with the resources already retrieved
def generate_grounded_resource_response(state):
    retrieved_resources = list(state.get("retrieved_resources", []) or [])

    if not retrieved_resources:
        return {
            "assistant_result": {},
            "error": (
                "No curated resources were available "
                "for grounded response generation."
            ),
        }

    try:
        # retrieved resource content is supplied directly to the local llm
        result = generate_llm_response(
            application_state=state.get("application_state", {}),
            user_message=_grounded_resource_user_message(
                state,
                retrieved_resources,
            ),
            task="conversation",
            retrieved_resources=retrieved_resources,
            chat_history=state.get("assistant_chat_history", []),
        )

    except Exception as error:
        return {
            "assistant_result": {},
            "error": (
                "The grounded Assistant response could not "
                f"be generated: {error}"
            ),
        }

    return {
        "assistant_result": result,
        "error": "",
    }


# asks the local llm to draft a personal summary from allowed application state
def run_summary_draft(state):
    result = generate_llm_response(
        application_state=state.get("application_state", {}),
        user_message=state.get("user_message", ""),
        task="summary",
        chat_history=state.get("assistant_chat_history", []),
    )

    return {
        "assistant_result": result,
        "retrieved_resources": [],
    }


# asks the local llm to draft appointment discussion points
def run_appointment_points(state):
    result = generate_llm_response(
        application_state=state.get("application_state", {}),
        user_message=state.get("user_message", ""),
        task="appointment_points",
        chat_history=state.get("assistant_chat_history", []),
    )

    return {
        "assistant_result": result,
        "retrieved_resources": [],
    }


# builds the complete bounded assistant graph and connects each routing branch
def build_assistant_graph():
    graph = StateGraph(AssistantState)

    # each node performs one controlled part of the assistant workflow
    nodes = {
        "validate_request": validate_assistant_request,
        "safety_check": check_urgent_safety,
        "urgent_safety_response": run_urgent_safety_response,
        "interpret_intent": interpret_user_intent,
        "conversation": run_conversation,
        "summary": run_summary_draft,
        "appointment_points": run_appointment_points,
        "prepare_resource_query": prepare_resource_query,
        "minilm_resource_retrieval": run_minilm_resource_retrieval,
        "grounded_resource_response": generate_grounded_resource_response,
    }

    # register all graph nodes
    for name, node in nodes.items():
        graph.add_node(name, node)

    # every assistant request begins with basic validation
    graph.add_edge(START, "validate_request")

    # invalid requests stop before any safety, retrieval or llm processing
    graph.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {
            "continue": "safety_check",
            "stop": END,
        },
    )

    # urgent safety handling takes priority over ordinary assistant tasks
    graph.add_conditional_edges(
        "safety_check",
        route_after_safety_check,
        {
            "urgent": "urgent_safety_response",
            "normal": "interpret_intent",
        },
    )

    # urgent responses end immediately and do not return to normal generation
    graph.add_edge("urgent_safety_response", END)

    # normal requests are sent to one of the four allowed assistant branches
    graph.add_conditional_edges(
        "interpret_intent",
        route_detected_intent,
        {
            "conversation": "conversation",
            "summary": "summary",
            "appointment_points": "appointment_points",
            "resources": "prepare_resource_query",
        },
    )

    # resource requests first prepare a personalised query and then run retrieval
    graph.add_edge(
        "prepare_resource_query",
        "minilm_resource_retrieval",
    )

    # grounded generation is allowed only after successful resource retrieval
    graph.add_conditional_edges(
        "minilm_resource_retrieval",
        route_after_resource_retrieval,
        {
            "generate": "grounded_resource_response",
            "stop": END,
        },
    )

    # each completed assistant task finishes the graph
    for node_name in (
        "grounded_resource_response",
        "conversation",
        "summary",
        "appointment_points",
    ):
        graph.add_edge(node_name, END)

    return graph.compile()


# compile the assistant graph once so Flask can reuse it for later requests
assistant_graph = build_assistant_graph()