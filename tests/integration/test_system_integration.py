
# imports active resources from the real resource database
from database.resources import get_active_resources
# imports the main assistant graph for end-to-end assistant testing
from graphs.assistant_graph import assistant_graph
# imports the basic and personalised resource retrieval functions
from rag.retriever import retrieve_resources
from rag.personalisation import retrieve_from_relevance
# imports the real embedding model and local llm service
from services.assistant_service import get_embedding_model
from services.llm_service import generate_llm_response


# combines important resource fields into searchable lowercase text
def resource_text(resource):
    topics = resource.get("topics", [])
    # convert topic lists into one text string
    if isinstance(topics, (list, tuple)):
        topics = " ".join(str(item) for item in topics)
    return " ".join([
        str(resource.get("title", "")),
        str(topics),
        str(resource.get("content", "")),
    ]).lower()


# checks whether any retrieved resource contains one of the expected terms
def has_term(resources, terms):
    for resource in resources:
        text = resource_text(resource)
        if any(term.lower() in text for term in terms):
            return True
    return False


# checks semantic retrieval using the real resource database and embedding model
def test_rag():
    resources = get_active_resources()
    assert len(resources) > 0
    model = get_embedding_model()

    # test retrieval for sleep, support and feeding queries
    cases = [
        (
            "I have hardly been sleeping because I keep waking with my baby.",
            ["sleep", "rest"],
        ),
        (
            "I feel overwhelmed and need more practical support.",
            ["support", "help", "wellbeing"],
        ),
        (
            "Feeding my baby has been difficult recently.",
            ["feeding", "breastfeeding", "infant feeding"],
        ),
    ]
    for query, terms in cases:
        result = retrieve_resources(
            query=query,
            resources=resources,
            embedding_model=model,
            top_k=5,
        )
        # each query should return relevant scored resources
        assert len(result) > 0
        assert has_term(result, terms) is True
        assert all("retrieval_score" in item for item in result)


# checks retrieval when personal pattern information is added to the query
def test_personal_rag():
    resources = get_active_resources()
    model = get_embedding_model()
    # example relevance result containing a repeated sleep pattern
    relevance = {
        "surface_patterns": [
            {
                "category": "repeated_pattern",
                "pattern": "sleep difficulty",
                "reason": "Repeated across recent check-ins.",
            }
        ],
        "supporting_patterns": [],
        "surface_count": 1,
        "supporting_count": 0,
    }
    result = retrieve_from_relevance(
        user_query="What information might be useful for me?",
        relevance_result=relevance,
        resources=resources,
        embedding_model=model,
        top_k=5,
    )
    # sleep-related personal context should retrieve sleep or rest resources
    assert len(result) > 0
    assert has_term(result, ["sleep", "rest"]) is True

# checks a real conversation response from the local llm
def test_llm_chat():
    result = generate_llm_response(
        application_state={
            "status": "Experiencing some strain today",
            "approved_themes": ["Sleep difficulty"],
        },
        user_message="Please give me a short supportive response.",
        task="conversation",
    )
    # conversation mode should return only the normal assistant reply
    assert isinstance(result, dict)
    assert result["assistant_reply"].strip() != ""
    assert result["summary_draft"] == ""
    assert result["appointment_points"] == []
    assert result["proposed_dashboard_updates"] == []
    assert result["needs_user_confirmation"] is False


# checks summary generation using approved check-in information
def test_llm_summary():
    result = generate_llm_response(
        application_state={
            "status": "Experiencing some strain today",
            "approved_text_emotions": ["Tiredness"],
            "approved_supportive_factors": ["Some practical support"],
            "approved_strain_factors": ["Sleep difficulty"],
        },
        user_message="Create a short summary from my approved check-in.",
        task="summary",
    )
    # summary mode should not return unrelated task outputs
    assert result["summary_draft"].strip() != ""
    assert result["appointment_points"] == []
    assert result["proposed_dashboard_updates"] == []
    assert result["needs_user_confirmation"] is False


# checks generation of appointment discussion points
def test_llm_points():
    result = generate_llm_response(
        application_state={
            "status": "Experiencing some strain today",
            "approved_strain_factors": [
                "Sleep difficulty",
                "Limited practical support",
            ],
        },
        user_message="Create possible appointment discussion points.",
        task="appointment_points",
    )

    points = result["appointment_points"]
    # the llm should return a small list of usable discussion points
    assert isinstance(points, list)
    assert 1 <= len(points) <= 4
    assert all(isinstance(point, str) and point.strip() for point in points)
    assert result["summary_draft"] == ""
    assert result["proposed_dashboard_updates"] == []

# checks the full assistant graph including intent detection rag and llm response
def test_assistant():
    result = assistant_graph.invoke({
        "user_message": "Find me a useful resource about postpartum sleep.",

        "application_state": {
            "status": "Experiencing some strain today",
            "approved_themes": ["Sleep difficulty"],
        },

        "relevance_result": {
            "surface_patterns": [
                {
                    "category": "repeated_pattern",
                    "pattern": "sleep difficulty",
                    "reason": "Repeated across recent check-ins.",
                }
            ],
            "supporting_patterns": [],
            "surface_count": 1,
            "supporting_count": 0,
        },

        "assistant_chat_history": [],
        "user_id": None,
    })

    # the message should be recognised as a resource request
    assert result["error"] == ""
    assert result["detected_intent"] == "resources"
    assert result["resource_retrieval_complete"] is True

    resources = result["retrieved_resources"]
    # retrieved results should contain relevant sleep information
    assert len(resources) > 0
    assert has_term(resources, ["sleep", "rest"]) is True
    assistant = result["assistant_result"]
    # the assistant should return a grounded reply without unrelated outputs
    assert assistant["assistant_reply"].strip() != ""
    assert assistant["summary_draft"] == ""
    assert assistant["appointment_points"] == []
    assert assistant["proposed_dashboard_updates"] == []