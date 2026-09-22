# json is used to build fake llm responses for testing
import json
# numpy is used to create predictable fake embedding vectors
import numpy as np
# requests is used to simulate connection errors from the local llm service
import requests
# imports the assistant graph module so its functions can be replaced during tests
import graphs.assistant_graph as agent_mod
# imports the compiled assistant graph and llm service being tested
from graphs.assistant_graph import assistant_graph
from services import llm_service


# replaces urgent safety detection with a safe result for normal assistant tests
def safe(monkeypatch):
    monkeypatch.setattr(
        agent_mod,
        "classify_urgent_safety",
        lambda user_message, chat_history=None: {
            "urgent_safety_detected": False,
            "safety_category": "none",
        },
    )

# provides predictable llm output without calling the real local model
def fake_llm(
    application_state,
    user_message,
    task="conversation",
    retrieved_resources=None,
    chat_history=None,
):
    return {
        "assistant_reply": "Test reply.",
        "summary_draft": "Test summary." if task == "summary" else "",
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


# fake embedding model used for resource retrieval tests
class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        self.calls.append(texts)
        values = []
        # assign simple vectors based on keywords
        for text in texts:
            text = str(text).lower()

            if "sleep" in text:
                vector = [1.0, 0.0]
            elif "support" in text:
                vector = [0.0, 1.0]
            else:
                vector = [0.1, 0.1]

            values.append(vector)

        values = np.array(values, dtype=float)

        # normalise vectors in the same way as the real embedding model
        if normalize_embeddings:
            lengths = np.linalg.norm(values, axis=1, keepdims=True)
            lengths[lengths == 0] = 1
            values = values / lengths

        return values


# fake http response used to test llm response parsing
class FakeResponse:
    def __init__(self, data):
        self.data = data
    # fake response does not raise an http error
    def raise_for_status(self):
        return None
    # returns the response in the structure expected from ollama
    def json(self):
        if isinstance(self.data, str):
            content = self.data
        else:
            content = json.dumps(self.data)

        return {
            "message": {
                "content": content
            }
        }


# provides a complete fixed llm result for tests that need standard output
def llm_result():
    return {
        "assistant_reply": "A supportive reply.",
        "summary_draft": "A summary.",
        "appointment_points": ["An appointment point."],
        "proposed_dashboard_updates": ["A dashboard update."],
        "resource_categories": ["A resource category."],
        "needs_user_confirmation": True,
        "urgent_safety_response": False,
    }


# checks that assistant intents are routed to the correct task
def test_routes(monkeypatch):
    # disable urgent safety routing for this test
    safe(monkeypatch)

    # simple intent classifier used to control routing
    def intent(user_message, chat_history=None):
        text = user_message.lower()

        if "summary" in text:
            return "summary"

        if "appointment" in text:
            return "appointment_points"

        if "diagnose" in text:
            return "diagnose_user"

        return "conversation"

    monkeypatch.setattr(
        agent_mod,
        "classify_assistant_intent",
        intent,
    )
    monkeypatch.setattr(
        agent_mod,
        "generate_llm_response",
        fake_llm,
    )

    # normal conversation should use the conversation route
    result = assistant_graph.invoke({
        "user_message": "Explain my check-in.",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })
    assert result["detected_intent"] == "conversation"
    assert result["assistant_result"]["assistant_reply"] == "Test reply."

    # summary requests should use the summary task
    result = assistant_graph.invoke({
        "user_message": "Create my summary.",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })

    assert result["detected_intent"] == "summary"
    assert result["assistant_result"]["summary_draft"] == "Test summary."
    # appointment requests should produce appointment points
    result = assistant_graph.invoke({
        "user_message": "Create appointment points.",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })

    assert result["detected_intent"] == "appointment_points"
    assert result["assistant_result"]["appointment_points"] == [
        "Test appointment point."
    ]

    # diagnosis requests should be redirected to normal conversation
    result = assistant_graph.invoke({
        "user_message": "Can you diagnose me?",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })

    assert result["detected_intent"] == "conversation"
    # an empty message should stop before normal assistant processing
    empty = assistant_graph.invoke({
        "user_message": "",
        "application_state": {},
        "relevance_result": {},
    })

    assert empty["error"] != ""
    assert "assistant_result" not in empty


# checks that urgent safety messages bypass normal assistant routing
def test_safety(monkeypatch):
    # returns a safety category based on the test message
    def check(user_message, chat_history=None):
        text = user_message.lower()

        if "myself" in text:
            return {
                "urgent_safety_detected": True,
                "safety_category": "self",
            }

        if "baby" in text:
            return {
                "urgent_safety_detected": True,
                "safety_category": "baby",
            }

        if "someone" in text:
            return {
                "urgent_safety_detected": True,
                "safety_category": "other_person",
            }

        return {
            "urgent_safety_detected": False,
            "safety_category": "none",
        }

    monkeypatch.setattr(
        agent_mod,
        "classify_urgent_safety",
        check,
    )
    # normal intent routing must not run for urgent messages
    def intent(user_message, chat_history=None):
        if any(
            word in user_message.lower()
            for word in ["myself", "baby", "someone"]
        ):
            raise AssertionError(
                "Normal routing should not run."
            )

        return "conversation"

    monkeypatch.setattr(
        agent_mod,
        "classify_assistant_intent",
        intent,
    )
    monkeypatch.setattr(
        agent_mod,
        "generate_llm_response",
        fake_llm,
    )
    cases = [
        ("I might hurt myself.", "self"),
        ("I might hurt my baby.", "baby"),
        ("I might hurt someone.", "other_person"),
    ]

    # check all three urgent safety categories
    for message, category in cases:
        result = assistant_graph.invoke({
            "user_message": message,
            "application_state": {},
            "relevance_result": {},
            "assistant_chat_history": [],
        })

        assert result["urgent_safety_detected"] is True
        assert result["safety_category"] == category
        assistant = result["assistant_result"]
        assert assistant["urgent_safety_response"] is True
        assert assistant["summary_draft"] == ""
        assert assistant["appointment_points"] == []
        assert result["retrieved_resources"] == []

    # a non-urgent message should continue through normal routing
    normal = assistant_graph.invoke({
        "user_message": "I feel very stressed today.",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })
    assert normal["urgent_safety_detected"] is False
    assert normal["detected_intent"] == "conversation"

# checks resource retrieval and grounding before the assistant reply is generated
def test_rag(monkeypatch):
    safe(monkeypatch)
    # force the assistant into the resource-retrieval route
    monkeypatch.setattr(
        agent_mod,
        "classify_assistant_intent",
        lambda user_message, chat_history=None: "resources",
    )
    model = FakeModel()
    # small approved resource set used by the retrieval test
    resources = [
        {
            "id": "r1",
            "title": "Postpartum Sleep",
            "resource_type": "article",
            "source": "Test source",
            "topics": ["sleep difficulty"],
            "content": "Information about sleep and rest after birth.",
            "url": "[https://example.com/sleep](https://example.com/sleep)",
        },
        {
            "id": "r2",
            "title": "Practical Support",
            "resource_type": "article",
            "source": "Test source",
            "topics": ["support"],
            "content": "Information about practical support after birth.",
            "url": "[https://example.com/support](https://example.com/support)",
        },
    ]

    monkeypatch.setattr(
        agent_mod,
        "get_active_resources",
        lambda: resources,
    )

    monkeypatch.setattr(
        agent_mod,
        "get_embedding_model",
        lambda: model,
    )
    calls = []

    # records the resources passed into the llm
    def grounded(
        application_state,
        user_message,
        task="conversation",
        retrieved_resources=None,
        chat_history=None,
    ):
        calls.append({
            "task": task,
            "resources": retrieved_resources,
        })

        return {
            "assistant_reply": "Here is a sleep resource.",
            "summary_draft": "",
            "appointment_points": [],
            "proposed_dashboard_updates": [],
            "resource_categories": [],
            "needs_user_confirmation": False,
            "urgent_safety_response": False,
        }

    monkeypatch.setattr(
        agent_mod,
        "generate_llm_response",
        grounded,
    )
    result = assistant_graph.invoke({
        "user_message": "Find something about sleep.",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })

    # sleep resource should be retrieved and passed to the llm
    assert result["resource_retrieval_complete"] is True
    assert len(model.calls) == 2
    assert result["retrieved_resources"][0]["id"] == "r1"
    assert len(calls) == 1
    assert calls[0]["task"] == "conversation"
    assert calls[0]["resources"][0]["id"] == "r1"
    assert (
        result["assistant_result"]["assistant_reply"]
        == "Here is a sleep resource."
    )


# checks that retrieval errors stop the llm from running
def test_rag_error(monkeypatch):
    safe(monkeypatch)

    monkeypatch.setattr(
        agent_mod,
        "classify_assistant_intent",
        lambda user_message, chat_history=None: "resources",
    )

    monkeypatch.setattr(
        agent_mod,
        "get_active_resources",
        lambda: [
            {
                "id": "r1",
                "title": "Sleep Resource",
                "resource_type": "article",
                "source": "Test source",
                "topics": ["sleep"],
                "content": "Information about sleep.",
                "url": "[https://example.com/sleep](https://example.com/sleep)",
            }
        ],
    )

    # simulate an unavailable embedding model
    def fail_model():
        raise RuntimeError("MiniLM unavailable")

    monkeypatch.setattr(
        agent_mod,
        "get_embedding_model",
        fail_model,
    )
    # llm should never run after retrieval failure
    def fail_llm(*args, **kwargs):
        raise AssertionError("LLM must not run.")
    monkeypatch.setattr(
        agent_mod,
        "generate_llm_response",
        fail_llm,
    )
    result = assistant_graph.invoke({
        "user_message": "Find a sleep resource.",
        "application_state": {},
        "relevance_result": {},
        "assistant_chat_history": [],
    })
    assert result["resource_retrieval_complete"] is False
    assert "MiniLM unavailable" in result["error"]
    assert result["retrieved_resources"] == []


# checks task-specific llm output rules
def test_llm_rules(monkeypatch):
    # normal conversation may return only an assistant reply
    monkeypatch.setattr(
        llm_service.HTTP_SESSION,
        "post",
        lambda *args, **kwargs: FakeResponse(
            "A supportive reply."
        ),
    )
    conversation = llm_service.generate_llm_response(
        application_state={},
        user_message="Hello",
        task="conversation",
    )
    assert conversation["assistant_reply"] == "A supportive reply."
    assert conversation["summary_draft"] == ""
    assert conversation["appointment_points"] == []
    assert conversation["proposed_dashboard_updates"] == []
    assert conversation["resource_categories"] == []
    assert conversation["needs_user_confirmation"] is False

    # dashboard updates should require confirmation before being applied
    update_data = {
        "proposed_dashboard_updates": [
            "A dashboard update."
        ]
    }
    monkeypatch.setattr(
        llm_service.HTTP_SESSION,
        "post",
        lambda *args, **kwargs: FakeResponse(update_data),
    )
    update = llm_service.generate_llm_response(
        application_state={},
        user_message="Add this to my dashboard.",
        task="dashboard_update",
    )
    assert update["proposed_dashboard_updates"] == [
        "A dashboard update."
    ]
    assert update["needs_user_confirmation"] is True
    # urgent safety output should remove normal generated content
    urgent_data = {
        "summary_draft": "A summary.",
        "urgent_safety_response": True,
    }

    monkeypatch.setattr(
        llm_service.HTTP_SESSION,
        "post",
        lambda *args, **kwargs: FakeResponse(urgent_data),
    )
    urgent = llm_service.generate_llm_response(
        application_state={},
        user_message="Urgent test message.",
        task="summary",
    )
    assert urgent["urgent_safety_response"] is True
    assert urgent["summary_draft"] == ""
    assert urgent["appointment_points"] == []
    assert urgent["proposed_dashboard_updates"] == []
    assert urgent["needs_user_confirmation"] is False


# checks empty input, unsupported tasks and local llm connection errors
def test_llm_errors(monkeypatch):
    calls = []
    # records whether an http request is made
    def track(*args, **kwargs):
        calls.append(True)
        return FakeResponse("Test reply.")
    monkeypatch.setattr(
        llm_service.HTTP_SESSION,
        "post",
        track,
    )
    # blank messages should stop before contacting the llm
    empty = llm_service.generate_llm_response(
        application_state={},
        user_message="   ",
        task="conversation",
    )
    assert calls == []
    assert empty["assistant_reply"] == (
        "Please enter a message before continuing."
    )
    # unsupported tasks should also stop before contacting the llm
    unsupported = llm_service.generate_llm_response(
        application_state={},
        user_message="Hello",
        task="delete_database",
    )
    assert calls == []
    assert "not supported" in unsupported["assistant_reply"].lower()
    # simulate the local ollama service being unavailable
    def offline(*args, **kwargs):
        raise requests.ConnectionError(
            "Ollama unavailable"
        )
    monkeypatch.setattr(
        llm_service.HTTP_SESSION,
        "post",
        offline,
    )
    result = llm_service.generate_llm_response(
        application_state={},
        user_message="Hello",
        task="conversation",
    )
    assert "unavailable" in result["assistant_reply"].lower()
    assert result["urgent_safety_response"] is False