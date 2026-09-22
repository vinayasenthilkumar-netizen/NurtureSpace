# imports sqlite so database errors can be simulated during saving tests
import sqlite3
# imports the service that controls saving approved check-ins and reflections
from services import checkin_saving


# checks the basic rules that decide whether an approved check-in may be saved
def test_checkin_save_rules(monkeypatch):
    # keep track of whether the privacy check is called
    calls = []
    # fake privacy check that always blocks saving
    def privacy(user_id):
        calls.append(user_id)
        return False

    # replace the real privacy function with the controlled test version
    monkeypatch.setattr(checkin_saving, "can_save_approved_checkins", privacy)
    # saving should fail when no user id is supplied
    result = checkin_saving.save_or_update_approved_checkin(None, {}, True)
    # the operation should report failure
    assert result["success"] is False
    # privacy should not be checked when there is no user
    assert calls == []
    # saving should also fail when persistence was not requested
    result = checkin_saving.save_or_update_approved_checkin(7, {}, False)
    # the operation should report failure
    assert result["success"] is False
    # privacy should still not be checked when saving was not requested
    assert calls == []
    # saving should reach the privacy check when a user exists and saving is requested
    result = checkin_saving.save_or_update_approved_checkin(7, {}, True)
    # saving should fail because the mocked privacy preference blocks it
    assert result["success"] is False
    # the privacy check should receive the correct user id
    assert calls == [7]


# checks whether approved check-ins are created or updated correctly
def test_checkin_create_update(monkeypatch):
    # allow saving for this test by forcing the privacy check to return true
    monkeypatch.setattr(checkin_saving, "can_save_approved_checkins", lambda user_id: True)
    # record which database operation is called
    calls = []

    # fake check-in creation without writing to the real database
    def create(user_id, **data):
        # store the call so the test can check the supplied values
        calls.append(("create", user_id, data))
        # return a predictable created record
        return {"id": 11, **data}

    # fake check-in update without writing to the real database
    def update(checkin_id, user_id, **data):
        # store the update call and its values
        calls.append(("update", checkin_id, user_id, data))
        # return a predictable updated record
        return {"id": checkin_id, **data}

    # replace the real create function with the fake version
    monkeypatch.setattr(checkin_saving, "create_checkin", create)
    # replace the real update function with the fake version
    monkeypatch.setattr(checkin_saving, "update_checkin", update)
    # create approved check-in data for the test
    data = {
        "confirmed_summary": "Approved summary",
        "confirmed_appointment_points": ["Discuss sleep."],
        "approved_text_emotions": ["Low mood"],
    }

    # save without an existing id so a new check-in should be created
    created = checkin_saving.save_or_update_approved_checkin(7, data, True)
    # save with an existing id so the same check-in should be updated
    updated = checkin_saving.save_or_update_approved_checkin(7, data, True, 11)
    # the first operation should be reported as a successful creation
    assert created["success"] is True and created["action"] == "created"
    # the second operation should be reported as a successful update
    assert updated["success"] is True and updated["action"] == "updated"
    # check that the correct create and update functions were called
    assert calls == [("create", 7, data), ("update", 11, 7, data)]


# checks that invalid data and database failures are handled safely
def test_checkin_save_errors(monkeypatch):
    # allow saving so the test can reach later validation and database logic
    monkeypatch.setattr(checkin_saving, "can_save_approved_checkins", lambda user_id: True)
    # supply invalid check-in data that is not a dictionary
    result = checkin_saving.save_or_update_approved_checkin(7, "bad", True)
    # invalid input should return a failed save result
    assert result["success"] is False

    # fake a database failure during check-in creation
    def db_error(**kwargs):
        raise sqlite3.Error("database unavailable")
    # replace the real create function with the failing version
    monkeypatch.setattr(checkin_saving, "create_checkin", db_error)
    # try to save valid data while the database function raises an error
    result = checkin_saving.save_or_update_approved_checkin(7, {"confirmed_summary": "OK"}, True)
    # the service should return failure instead of allowing the database exception to escape
    assert result["success"] is False
    # the returned message should explain that the information could not be saved
    assert "could not be saved" in result["message"].lower()

    # fake a database failure while reading privacy preferences
    def privacy_error(user_id):
        raise sqlite3.Error("privacy unavailable")
    # replace the privacy check with the failing version
    monkeypatch.setattr(checkin_saving, "can_save_approved_checkins", privacy_error)
    # attempt to save while privacy preferences cannot be read
    result = checkin_saving.save_or_update_approved_checkin(7, {}, True)
    # the operation should fail safely
    assert result["success"] is False
    # the message should identify the privacy-preference problem
    assert "privacy preferences" in result["message"].lower()


# checks whether approved reflection-only entries are created or updated correctly
def test_reflection_save(monkeypatch):
    # allow approved reflection saving for this test
    monkeypatch.setattr(checkin_saving, "can_save_approved_checkins", lambda user_id: True)
    # record calls made to the reflection database functions
    calls = []

    # fake creation of a reflection-only record
    def create(user_id, **data):
        # store the supplied values for later checking
        calls.append(("create", user_id, data))
        # return a predictable created reflection
        return {"id": 21, **data}

    # fake update of an existing reflection-only record
    def update(reflection_id, user_id, **data):
        # store the supplied update values
        calls.append(("update", reflection_id, user_id, data))
        # return a predictable updated reflection
        return {"id": reflection_id, **data}
    # replace the real reflection create function
    monkeypatch.setattr(checkin_saving, "create_saved_reflection", create)
    # replace the real reflection update function
    monkeypatch.setattr(checkin_saving, "update_saved_reflection", update)
    # create approved reflection data for the test
    data = {
        "confirmed_summary": "Reflection summary",
        "confirmed_appointment_points": ["Discuss recovery."],
    }

    # save without an existing reflection id so a new record should be created
    created = checkin_saving.save_or_update_approved_reflection(7, data, True)
    # save with an existing reflection id so the record should be updated
    updated = checkin_saving.save_or_update_approved_reflection(7, data, True, 21)
    # the first operation should report a successful creation
    assert created["success"] is True and created["action"] == "created"
    # the second operation should report a successful update
    assert updated["success"] is True and updated["action"] == "updated"
    # check that the expected create and update functions were called
    assert calls == [("create", 7, data), ("update", 21, 7, data)]