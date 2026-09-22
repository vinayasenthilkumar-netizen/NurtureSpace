# used to convert resource topics between python and json
import json
# used to provide a fixed calendar date for daily-resource tests
from datetime import date
# used by the fake embedding model to create and normalise vectors
import numpy as np
# used for fixtures and test behaviour
import pytest
# imports the database connection module so test paths can be replaced
import database.connection as db_conn
# imports the function that creates the database tables
from database.schema import initialise_database
# imports user functions needed for bookmark ownership and cascade tests
from database.users import create_user, delete_user
# imports the resource and bookmark database functions being tested
from database.resources import (
    get_active_resources,
    get_resource_by_id,
    count_resources,
    get_daily_resources,
    bookmark_resource,
    remove_resource_bookmark,
    is_resource_bookmarked,
    get_bookmarked_resource_ids,
    get_bookmarked_resources,
)

# imports the basic semantic resource-retrieval functions
from rag.retriever import build_resource_text, retrieve_resources

# imports the personalisation helpers used for pattern-based retrieval
from rag.personalisation import (
    build_personalised_query,
    get_relevance_context,
    retrieve_personalised_resources,
    retrieve_from_relevance,
)


# provides simple predictable embeddings without loading the real embedding model
class FakeModel:

    # converts test text into small fixed vectors
    def encode(self, texts, normalize_embeddings=True):
        # stores the vector created for each input text
        vectors = []

        # process every resource or query text
        for text in texts:
            # convert the text to lowercase for simple keyword matching
            text = str(text).lower()

            # sleep-related text is placed on the first vector dimension
            if "sleep" in text:
                vector = [1.0, 0.0, 0.0]

            # feeding-related text is placed on the third vector dimension
            elif "feeding" in text:
                vector = [0.0, 0.0, 1.0]

            # support or help text is placed on the second vector dimension
            elif "support" in text or "help" in text:
                vector = [0.0, 1.0, 0.0]

            # unrelated text receives a small general vector
            else:
                vector = [0.1, 0.1, 0.1]

            # add the created vector to the result list
            vectors.append(vector)

        # convert the python list into a numpy array
        vectors = np.array(vectors, dtype=float)

        # normalise vectors when requested by the retrieval code
        if normalize_embeddings:
            # calculate the length of each vector
            sizes = np.linalg.norm(vectors, axis=1, keepdims=True)

            # avoid division by zero for empty vectors
            sizes[sizes == 0] = 1

            # scale each vector to unit length
            vectors = vectors / sizes

        # return the final embedding matrix
        return vectors

# creates a separate temporary sqlite database for each database test
@pytest.fixture()
def db(tmp_path, monkeypatch):
    # create the temporary database path
    path = tmp_path / "test.db"

    # make the connection helper use the pytest temporary directory
    monkeypatch.setattr(db_conn, "DATABASE_DIR", tmp_path)

    # make database connections use the temporary test file
    monkeypatch.setattr(db_conn, "DATABASE_PATH", path)

    # create all required tables in the temporary database
    initialise_database()

    # return the database path to tests that request this fixture
    return path


# inserts one test resource into the temporary resource database
def add_resource(
    resource_id,
    title,
    topics,
    content,
    resource_type="article",
    active=1,
    daily=1,
):
    # open the temporary sqlite database
    with db_conn.open_database() as connection:

        # insert the supplied resource values
        connection.execute(
            """
            INSERT INTO resources (
                id,
                title,
                resource_type,
                source,
                topics_json,
                content,
                url,
                is_active,
                daily_eligible
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                # save the test resource id
                resource_id,
                # save the resource title
                title,
                # save the selected resource type
                resource_type,
                # all test resources use the same source name
                "Test source",
                # convert the topic list to json for database storage
                json.dumps(topics),
                # save the resource content
                content,
                # create a simple test url using the resource id
                "https://example.com/" + resource_id,
                # save whether the resource is active
                active,
                # save whether it can appear as a daily resource
                daily,
            ),
        )


# creates a simple registered user for bookmark tests
def make_user(name="maya"):
    # use the normal user creation function with fixed test credentials
    return create_user(
        username=name,
        email=f"{name}@example.com",
        password_hash="test-hash",
        display_name=name.title(),
    )


# returns a small in-memory resource collection for retrieval tests
def resources():
    # each resource focuses on a different retrieval topic
    return [
        {
            "id": "r1",
            "title": "Sleep After Birth",
            "resource_type": "article",
            "source": "Test source",
            "topics": ["sleep difficulty"],
            "content": "Information about sleep and rest after birth.",
        },
        {
            "id": "r2",
            "title": "Finding Support",
            "resource_type": "article",
            "source": "Test source",
            "topics": ["practical support"],
            "content": "Information about getting help and support.",
        },
        {
            "id": "r3",
            "title": "Feeding Help",
            "resource_type": "article",
            "source": "Test source",
            "topics": ["feeding concerns"],
            "content": "Information about feeding.",
        },
    ]


# checks resource text building and basic semantic retrieval behaviour
def test_retrieval():
    # create the fixed test resource collection
    items = resources()

    # build the searchable text for the first resource
    text = build_resource_text(items[0])
    # the resource title should be included
    assert "Sleep After Birth" in text
    # the resource topic should be included
    assert "sleep difficulty" in text
    # the resource content should also be included
    assert "Information about sleep" in text
    # use the predictable fake model instead of the real embedding model
    model = FakeModel()
    # retrieve the two resources most related to sleep
    sleep = retrieve_resources(
        query="sleep difficulty",
        resources=items,
        embedding_model=model,
        top_k=2,
    )

    # two resources should be returned
    assert len(sleep) == 2
    # the sleep resource should be ranked first
    assert sleep[0]["id"] == "r1"
    # retrieved resources should include their similarity score
    assert "retrieval_score" in sleep[0]
    # run a separate retrieval query about support
    support = retrieve_resources(
        query="I need more support",
        resources=items,
        embedding_model=model,
        top_k=1,
    )

    # the support resource should be ranked first
    assert support[0]["id"] == "r2"
    # an empty query should return no resources
    assert retrieve_resources(
        query="",
        resources=items,
        embedding_model=model,
        top_k=3,
    ) == []
    # an empty resource collection should return no results
    assert retrieve_resources(
        query="sleep",
        resources=[],
        embedding_model=model,
        top_k=3,
    ) == []

    # asking for zero results should return an empty list
    assert retrieve_resources(
        query="sleep",
        resources=items,
        embedding_model=model,
        top_k=0,
    ) == []

# checks how user queries are combined with personal pattern context
def test_personal_query():
    # build a query using the user's request and relevant patterns
    query = build_personalised_query(
        user_query="What might help me?",
        relevant_patterns=[
            "Recurring sleep difficulty",
            "",
            None,
            "Limited support",
        ],
    )

    # the original user request should remain in the query
    assert "What might help me?" in query
    # valid sleep pattern context should be included
    assert "Recurring sleep difficulty" in query
    # valid support context should also be included
    assert "Limited support" in query
    # create more patterns than the maximum allowed
    patterns = [
        "Pattern one",
        "Pattern two",
        "Pattern three",
        "Pattern four",
        "Pattern five",
        "Pattern six",
    ]

    # limit personal context to five patterns
    limited = build_personalised_query(
        user_query="Help me",
        relevant_patterns=patterns,
        max_patterns=5,
    )

    # the fifth allowed pattern should remain
    assert "Pattern five" in limited
    # the sixth pattern should be excluded by the limit
    assert "Pattern six" not in limited
    # no query and no patterns should produce an empty query
    assert build_personalised_query(
        user_query="",
        relevant_patterns=[],
    ) == ""
    # create example personal pattern relevance output
    relevance = {
        "surface_patterns": [
            {
                "category": "recent_change",
                "pattern": {"field": "personal_care"},
            },
            {
                "category": "variable_pattern",
                "pattern": {"field": "stress"},
            },
        ],
        "supporting_patterns": [
            {
                "category": "reflection_alignment",
                "pattern": {"theme": "sleep difficulty"},
            }
        ],
    }

    # convert the structured relevance result into retrieval context text
    context = get_relevance_context(relevance)
    # recent personal-care change should be described
    assert "personal care has changed" in context[0]
    # stress variability should be described
    assert "stress has varied" in context[1]
    # reflection alignment should include the sleep theme
    assert "sleep difficulty" in context[2]

# checks retrieval when personal pattern information is added to the query
def test_personal_rag():
    # create the fixed resource collection
    items = resources()
    # use the predictable fake embedding model
    model = FakeModel()
    # retrieve using both the user query and a sleep-related personal pattern
    result = retrieve_personalised_resources(
        user_query="What resources might help?",
        relevant_patterns=["Recurring sleep difficulty"],
        resources=items,
        embedding_model=model,
        top_k=1,
    )

    # only the top result should be returned
    assert len(result) == 1
    # sleep context should lead to the sleep resource
    assert result[0]["id"] == "r1"
    # create a relevance structure containing a repeated sleep pattern
    relevance = {
        "surface_patterns": [
            {
                "category": "repeated_pattern",
                "pattern": "sleep difficulty",
            }
        ],
        "supporting_patterns": [],
    }

    # retrieve directly from the relevance result
    direct = retrieve_from_relevance(
        user_query="What may be useful?",
        relevance_result=relevance,
        resources=items,
        embedding_model=model,
        top_k=1,
    )

    # the sleep resource should again be the top result
    assert direct[0]["id"] == "r1"


# checks resource storage, active-resource filtering and id lookup
def test_resource_db(db):
    # add an active sleep resource
    add_resource(
        "1",
        "Sleep Resource",
        ["sleep"],
        "Sleep information.",
    )
    # add an active support resource
    add_resource(
        "2",
        "Support Resource",
        ["support"],
        "Support information.",
    )

    # add an inactive resource that should not appear in active results
    add_resource(
        "3",
        "Inactive Resource",
        ["feeding"],
        "Hidden information.",
        active=0,
    )

    # all three rows should still exist in the database
    assert count_resources() == 3
    # retrieve only active resources
    active = get_active_resources()
    # only the two active resources should be returned
    assert len(active) == 2
    # collect their ids for easier comparison
    ids = {item["id"] for item in active}
    # the inactive resource should not be included
    assert ids == {"1", "2"}
    # retrieve the sleep resource directly by id
    sleep = get_resource_by_id("1")
    # check the stored title
    assert sleep["title"] == "Sleep Resource"
    # json topics should be returned as a normal python list
    assert sleep["topics"] == ["sleep"]
    # the sqlite active flag should be converted to a boolean
    assert sleep["is_active"] is True
    # an empty id should not return a resource
    assert get_resource_by_id("") is None
    # an unknown id should also return none
    assert get_resource_by_id("missing") is None


# checks daily-resource filtering, rotation and two-resource limit
def test_daily(db):
    # add a daily article
    add_resource(
        "1",
        "Sleep",
        ["sleep"],
        "Sleep information.",
        resource_type="article",
        daily=1,
    )

    # add a daily blog
    add_resource(
        "2",
        "Support",
        ["support"],
        "Support information.",
        resource_type="blog",
        daily=1,
    )

    # add a daily video
    add_resource(
        "3",
        "Video",
        ["recovery"],
        "Recovery information.",
        resource_type="video",
        daily=1,
    )

    # add a podcast which is not an allowed daily display type
    add_resource(
        "4",
        "Podcast",
        ["feeding"],
        "Podcast information.",
        resource_type="podcast",
        daily=1,
    )

    # add an active resource that is not daily eligible
    add_resource(
        "5",
        "Not Daily",
        ["mood"],
        "Mood information.",
        daily=0,
    )

    # use a fixed date so daily rotation is predictable
    today = date(2026, 8, 31)
    # request the daily resources for the fixed date
    first = get_daily_resources(
        maximum_resources=5,
        current_date=today,
    )
    # run the same request again
    second = get_daily_resources(
        maximum_resources=5,
        current_date=today,
    )
    # the same calendar date should produce the same rotation
    assert first == second
    # daily display is capped at two
    assert len(first) == 2
    # every returned resource must be marked daily eligible
    assert all(item["daily_eligible"] for item in first)
    # only articles, blogs and videos are allowed in daily resources
    assert all(
        item["resource_type"].lower() in {"article", "blog", "video"}
        for item in first
    )
    # the podcast and non-daily resource must not be returned
    assert all(item["id"] not in {"4", "5"} for item in first)

# checks bookmark creation, duplicate handling, ownership and deletion
def test_bookmarks(db):
    # add the first resource that users can bookmark
    add_resource(
        "1",
        "Sleep Resource",
        ["sleep"],
        "Sleep information.",
    )

    # add a second resource used to test an unsaved bookmark
    add_resource(
        "2",
        "Support Resource",
        ["support"],
        "Support information.",
    )

    # create the first test user
    maya = make_user("maya")
    # create a second user to check bookmark independence
    sara = make_user("sara")
    # save the sleep resource for maya
    bookmark_resource(maya["id"], "1")
    # duplicate bookmark is ignored
    bookmark_resource(maya["id"], "1")
    # save the same resource independently for sara
    bookmark_resource(sara["id"], "1")
    # maya should have resource one bookmarked
    assert is_resource_bookmarked(maya["id"], "1") is True
    # maya should not have resource two bookmarked
    assert is_resource_bookmarked(maya["id"], "2") is False
    # maya's bookmarked id set should contain only resource one
    assert get_bookmarked_resource_ids(maya["id"]) == {"1"}
    # retrieve maya's full bookmarked resources
    saved = get_bookmarked_resources(maya["id"])
    # the duplicate save should still produce only one bookmark
    assert len(saved) == 1
    # the returned bookmark should be resource one
    assert saved[0]["id"] == "1"
    # bookmarked resources should include the saved timestamp
    assert "bookmarked_at" in saved[0]
    # remove maya's bookmark
    remove_resource_bookmark(maya["id"], "1")
    # maya should no longer have the resource bookmarked
    assert is_resource_bookmarked(maya["id"], "1") is False
    # sara's bookmark is independent
    assert is_resource_bookmarked(sara["id"], "1") is True
    # account deletion cascades to bookmarks
    delete_user(sara["id"])
    # sara's bookmark should disappear when her account is deleted
    assert is_resource_bookmarked(sara["id"], "1") is False