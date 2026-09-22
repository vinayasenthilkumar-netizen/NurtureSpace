# pytest is used for fixtures and exception checks
import pytest
# imports the database connection module so test paths can be replaced
import database.connection as db_conn
# imports the database setup and user creation functions used by the tests
from database.schema import initialise_database
from database.users import create_user
# imports check-in database functions for create, read, update and delete tests
from database.checkins import create_checkin,delete_checkin,get_checkin_by_id,get_checkins_for_user,update_checkin
# imports the scoring and context interpretation functions being tested
from modules import scoring
from modules.context_interpretation import interpret_context_answers


# creates a separate temporary database for each database test
@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    # use the pytest temporary folder instead of the real application database
    monkeypatch.setattr(db_conn, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(db_conn, "DATABASE_PATH", path)

    initialise_database()
    return path

# creates a simple test user
def make_user(
    name="maya",
    email="maya@example.com",
):
    return create_user(
        username=name,
        email=email,
        password_hash="test-hash",
        display_name="Maya",
    )

# provides fixed core answers used across the tests
def core():
    return {
        "sleep": 2,
        "mood": 3,
        "stress": 4,
        "support": 2,
    }
# provides fixed context answers used across the tests
def context():
    return {
        "food": 2,
        "medication": 0,
        "physical_recovery": 2,
        "baby_care": 4,
        "other_responsibilities": 1,
        "personal_care": 2,
    }


# creates a complete approved check-in for database tests
def save_checkin(
    user_id,
    summary="Test summary",
):
    return create_checkin(
        user_id=user_id,
        question_set_version="2.0",
        checkin_status="Experiencing some strain today",
        core_answers=core(),
        context_answers=context(),
        approved_text_emotions=[
            "Tiredness"
        ],
        approved_supportive_factors=[
            "Some practical support"
        ],
        approved_strain_factors=[
            "Sleep difficulty"
        ],
        approved_vocal_observation=(
            "Tired-sounding vocal pattern"
        ),
        approved_visible_observation=(
            "Downturned-looking expression"
        ),
        confirmed_summary=summary,
        confirmed_appointment_points=[
            "I would like to discuss my sleep."
        ],
    )


# checks stress reversal, question averages and status calculation
def test_status():
    # higher stress answers are reversed before the average is calculated
    assert scoring.reverse_stress_response(1) == 5
    assert scoring.reverse_stress_response(3) == 3
    assert scoring.reverse_stress_response(5) == 1

    average = scoring.calculate_checkin_average(
        sleep=4,
        mood=3,
        stress=5,
        support=4,
    )

    assert average == 3.0

    # changing one slider by one point changes the four-question mean by 0.25
    base = scoring.calculate_checkin_average(
        sleep=3,
        mood=3,
        stress=3,
        support=3,
    )

    better = scoring.calculate_checkin_average(
        sleep=4,
        mood=3,
        stress=3,
        support=3,
    )

    assert better - base == 0.25
    # check examples from each of the three status ranges
    assert scoring.calculate_checkin_status(
        sleep=5,
        mood=5,
        stress=1,
        support=5,
    ) == scoring.STEADY_STATUS

    assert scoring.calculate_checkin_status(
        sleep=3,
        mood=3,
        stress=3,
        support=3,
    ) == scoring.SOME_STRAIN_STATUS

    assert scoring.calculate_checkin_status(
        sleep=1,
        mood=1,
        stress=5,
        support=1,
    ) == scoring.SIGNIFICANT_STRAIN_STATUS


# checks the exact boundaries between the three status bands
def test_status_edges():
    cases = [
        (3.67, scoring.STEADY_STATUS),
        (3.66, scoring.SOME_STRAIN_STATUS),
        (2.34, scoring.SOME_STRAIN_STATUS),
        (2.33, scoring.SIGNIFICANT_STRAIN_STATUS),
    ]

    # every threshold value should return the expected status
    for value, expected in cases:
        assert scoring.get_status_from_average(value) == expected
    # averages outside the 1 to 5 scale should be rejected
    for value in [0.99, 5.01]:
        with pytest.raises(ValueError):
            scoring.get_status_from_average(value)


# checks that missing, wrong-type and out-of-range core answers are rejected
def test_bad_answers():
    cases = [
        {
            "sleep": None,
            "mood": 3,
            "stress": 2,
            "support": 4,
        },
        {
            "sleep": 0,
            "mood": 3,
            "stress": 2,
            "support": 4,
        },
        {
            "sleep": 3,
            "mood": "3",
            "stress": 2,
            "support": 4,
        },
        {
            "sleep": 3,
            "mood": 3,
            "stress": True,
            "support": 4,
        },
        {
            "sleep": 3,
            "mood": 3,
            "stress": 6,
            "support": 4,
        },
    ]
    # each invalid answer set should raise a validation error
    for answers in cases:
        with pytest.raises(ValueError):
            scoring.calculate_checkin_status(
                **answers
            )


# checks interpretation of the six context questions
def test_context():
    # low context answers should produce difficulty and discussion points
    difficult = interpret_context_answers({
        "food": 1,
        "medication": 1,
        "physical_recovery": 1,
        "baby_care": 1,
        "other_responsibilities": 1,
        "personal_care": 1,
    })
    assert len(difficult["difficulty_points"]) == 6
    assert len(difficult["appointment_suggestions"]) == 6
    # zero values are treated as not applicable for optional context areas
    optional = interpret_context_answers({
        "food": 5,
        "medication": 0,
        "physical_recovery": 5,
        "baby_care": 0,
        "other_responsibilities": 0,
        "personal_care": 5,
    })
    assert optional["not_applicable"] == [
        "medication",
        "baby_care",
        "other_responsibilities",
    ]
    assert len(optional["summary_points"]) == 3
    # context answers should not calculate a score or status
    assert "score" not in difficult
    assert "status" not in difficult
    assert "average" not in difficult

# checks that context answers cannot change the core-question status
def test_context_no_status():
    answers = {
        "sleep": 3,
        "mood": 3,
        "stress": 3,
        "support": 3,
    }

    status = scoring.calculate_checkin_status(
        **answers
    )
    # compare very low and very high context responses
    low = interpret_context_answers({
        "food": 1,
        "medication": 1,
        "physical_recovery": 1,
        "baby_care": 1,
        "other_responsibilities": 1,
        "personal_care": 1,
    })
    high = interpret_context_answers({
        "food": 5,
        "medication": 5,
        "physical_recovery": 5,
        "baby_care": 5,
        "other_responsibilities": 5,
        "personal_care": 5,
    })

    assert scoring.calculate_checkin_status(
        **answers
    ) == status
    # context interpretation should never contain a status
    assert "status" not in low
    assert "status" not in high


# checks saving, reading and validation of approved check-ins
def test_save(db):
    user = make_user()
    # save two entries so history ordering can also be checked
    first = save_checkin(
        user["id"],
        "First summary",
    )

    second = save_checkin(
        user["id"],
        "Second summary",
    )

    assert first["id"] > 0
    assert first["core_answers"]["sleep"] == 2
    assert first["context_answers"]["medication"] == 0
    assert first["approved_text_emotions"] == [
        "Tiredness"
    ]
    assert first["confirmed_summary"] == (
        "First summary"
    )

    # raw reflection data should not be stored in the approved check-in
    for key in [
        "transcript",
        "typed_reflection",
        "audio_path",
        "video_path",
        "frame_predictions",
    ]:
        assert key not in first

    history = get_checkins_for_user(
        user["id"]
    )

    # newest saved check-in should appear first
    assert len(history) == 2
    assert history[0]["id"] == second["id"]
    bad_core = core()
    bad_core.pop("support")
    # saving should fail when a required core answer is missing
    with pytest.raises(
        ValueError,
        match="Core answer is missing",
    ):
        create_checkin(
            user_id=user["id"],
            question_set_version="2.0",
            checkin_status="Some status",
            core_answers=bad_core,
            context_answers=context(),
        )


# checks that an existing saved check-in can be updated
def test_update(db):
    user = make_user()
    checkin = save_checkin(user["id"])
    # replace the saved status, answers and approved summary content
    updated = update_checkin(
        checkin_id=checkin["id"],
        user_id=user["id"],
        question_set_version="2.0",
        checkin_status="Feeling relatively steady today",
        core_answers={
            "sleep": 4,
            "mood": 4,
            "stress": 2,
            "support": 4,
        },
        context_answers=context(),
        approved_text_emotions=[],
        approved_supportive_factors=[],
        approved_strain_factors=[],
        approved_vocal_observation="",
        approved_visible_observation="",
        confirmed_summary="Updated summary",
        confirmed_appointment_points=[],
    )
    # updating should keep the same database record id
    assert updated["id"] == checkin["id"]
    assert updated["checkin_status"] == (
        "Feeling relatively steady today"
    )
    assert updated["confirmed_summary"] == (
        "Updated summary"
    )


# checks that users cannot read or delete another user's check-ins
def test_access(db):
    first_user = make_user(
        "maya",
        "maya@example.com",
    )
    second_user = make_user(
        "sara",
        "sara@example.com",
    )
    checkin = save_checkin(
        first_user["id"]
    )
    # another user should not be able to retrieve the saved check-in
    other_read = get_checkin_by_id(
        checkin["id"],
        second_user["id"],
    )
    assert other_read is None
    # another user should also be unable to delete it
    other_delete = delete_checkin(
        checkin["id"],
        second_user["id"],
    )

    assert other_delete is False
    assert get_checkin_by_id(
        checkin["id"],
        first_user["id"],
    ) is not None

    # the owner should be able to delete their own check-in
    own_delete = delete_checkin(
        checkin["id"],
        first_user["id"],
    )

    assert own_delete is True
    assert get_checkin_by_id(
        checkin["id"],
        first_user["id"],
    ) is None