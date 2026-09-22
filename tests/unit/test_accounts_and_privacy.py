# pytest is used for the temporary database fixture
import pytest
# imports the database connection module so test paths can be replaced
import database.connection as db_conn
# imports the current privacy version and database setup function
from config.settings import CURRENT_PRIVACY_VERSION
from database.schema import initialise_database
# imports user functions used for account checks
from database.users import get_user_by_id, set_user_active
# imports privacy functions used to test saving permissions
from database.privacy import (
    get_privacy_preferences,
    save_privacy_preferences,
    privacy_is_current,
    can_save_approved_checkins,
)
# imports check-in functions used for account deletion tests
from database.checkins import create_checkin, get_checkins_for_user
# imports registration and login services
from services.auth_service import register_user, authenticate_user
# imports password-change and account-deletion services
from services.account_service import (
    change_password_with_current_password,
    delete_account_with_password,
)


# fixed passwords used across the account tests
PASS = "Password123!"
NEW_PASS = "NewPassword456!"


# creates a separate temporary database for each test
@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    # use the temporary pytest database instead of the real application database
    monkeypatch.setattr(db_conn, "DATABASE_DIR", tmp_path)
    monkeypatch.setattr(db_conn, "DATABASE_PATH", path)
    initialise_database()
    return path


# creates a standard registered user used by several tests
def make_user():
    result = register_user(
        username="maya",
        email="maya@example.com",
        password=PASS,
        confirm_password=PASS,
        display_name="Maya"
    )
    assert result["success"] is True
    return result["user"]


# checks successful registration, normalisation and duplicate handling
def test_register(db):
    result = register_user(
        username="  Maya_01  ",
        email="  Maya@Example.COM  ",
        password=PASS,
        confirm_password=PASS,
        display_name="  Maya  "
    )
    assert result["success"] is True
    user = result["user"]
    # username and email should be cleaned before being returned
    assert user["username"] == "maya_01"
    assert user["email"] == "maya@example.com"
    assert user["display_name"] == "Maya"
    assert "password_hash" not in user
    # the stored password should be hashed rather than saved as plain text
    stored = get_user_by_id(user["id"])
    assert stored["password_hash"] != PASS
    same_name = register_user(
        username="MAYA_01",
        email="other@example.com",
        password=PASS,
        confirm_password=PASS
    )
    assert same_name["success"] is False
    assert "username" in same_name["message"].lower()
    same_email = register_user(
        username="maya_02",
        email="MAYA@EXAMPLE.COM",
        password=PASS,
        confirm_password=PASS
    )
    assert same_email["success"] is False
    assert "email" in same_email["message"].lower()


# checks common invalid registration inputs
def test_bad_register(db):
    blank_name = register_user(
        username="",
        email="one@example.com",
        password=PASS,
        confirm_password=PASS
    )
    short_name = register_user(
        username="ab",
        email="two@example.com",
        password=PASS,
        confirm_password=PASS
    )
    bad_name = register_user(
        username="maya-user",
        email="three@example.com",
        password=PASS,
        confirm_password=PASS
    )
    bad_email = register_user(
        username="maya1",
        email="not-an-email",
        password=PASS,
        confirm_password=PASS
    )
    short_pass = register_user(
        username="maya2",
        email="four@example.com",
        password="short",
        confirm_password="short"
    )
    mismatch = register_user(
        username="maya3",
        email="five@example.com",
        password=PASS,
        confirm_password="Different123!"
    )

    # every invalid registration should be rejected
    assert blank_name["success"] is False
    assert short_name["success"] is False
    assert bad_name["success"] is False
    assert bad_email["success"] is False
    assert short_pass["success"] is False
    assert mismatch["success"] is False


# checks successful login and common login failures
def test_login(db):
    user = make_user()
    # login should ignore surrounding spaces and username case
    good = authenticate_user(
        username=" MAYA ",
        password=PASS
    )
    assert good["success"] is True
    assert good["user"]["id"] == user["id"]
    assert "password_hash" not in good["user"]

    wrong = authenticate_user(
        username="maya",
        password="WrongPassword!"
    )
    assert wrong["success"] is False

    missing = authenticate_user(
        username="missing",
        password=PASS
    )
    assert missing["success"] is False

    # inactive accounts should not be allowed to log in
    set_user_active(user["id"], False)

    inactive = authenticate_user(
        username="maya",
        password=PASS
    )

    assert inactive["success"] is False
    assert "inactive" in inactive["message"].lower()


# checks privacy preferences and whether approved check-ins can be saved
def test_privacy(db):
    user = make_user()
    user_id = user["id"]
    # a new user should not have accepted privacy preferences yet
    assert get_privacy_preferences(user_id) is None
    assert privacy_is_current(user_id) is False
    assert can_save_approved_checkins(user_id) is False

    saved = save_privacy_preferences(
        user_id=user_id,
        save_approved_checkins=True,
        use_saved_history_for_personalisation=True,
        save_bookmarked_resources=True
    )

    assert saved["save_approved_checkins"] is True
    assert saved["use_saved_history_for_personalisation"] is True
    assert saved["save_bookmarked_resources"] is True
    # current privacy consent should allow approved check-ins to be saved
    assert privacy_is_current(user_id) is True
    assert can_save_approved_checkins(user_id) is True
    # an old privacy version should no longer count as current
    save_privacy_preferences(
        user_id=user_id,
        save_approved_checkins=True,
        privacy_version="old-version"
    )
    assert privacy_is_current(user_id) is False
    assert can_save_approved_checkins(user_id) is False
    # current consent can still explicitly disable check-in saving
    save_privacy_preferences(
        user_id=user_id,
        save_approved_checkins=False,
        privacy_version=CURRENT_PRIVACY_VERSION
    )
    assert privacy_is_current(user_id) is True
    assert can_save_approved_checkins(user_id) is False


# checks password-change validation and successful password replacement
def test_password_change(db):
    user = make_user()
    wrong = change_password_with_current_password(
        user_id=user["id"],
        username="maya",
        current_password="WrongPassword!",
        new_password=NEW_PASS,
        confirm_password=NEW_PASS
    )
    assert wrong["success"] is False
    mismatch = change_password_with_current_password(
        user_id=user["id"],
        username="maya",
        current_password=PASS,
        new_password=NEW_PASS,
        confirm_password="DifferentPassword!"
    )
    assert mismatch["success"] is False
    same = change_password_with_current_password(
        user_id=user["id"],
        username="maya",
        current_password=PASS,
        new_password=PASS,
        confirm_password=PASS
    )
    assert same["success"] is False
    wrong_user = change_password_with_current_password(
        user_id=9999,
        username="maya",
        current_password=PASS,
        new_password=NEW_PASS,
        confirm_password=NEW_PASS
    )
    assert wrong_user["success"] is False
    # a valid request should replace the old password
    changed = change_password_with_current_password(
        user_id=user["id"],
        username="maya",
        current_password=PASS,
        new_password=NEW_PASS,
        confirm_password=NEW_PASS
    )

    assert changed["success"] is True
    old_login = authenticate_user(username="maya", password=PASS)
    new_login = authenticate_user(username="maya", password=NEW_PASS)

    assert old_login["success"] is False
    assert new_login["success"] is True


# checks password-protected account deletion and cascading saved data removal
def test_delete_account(db):
    user = make_user()
    user_id = user["id"]
    # allow this user to save approved check-ins
    save_privacy_preferences(
        user_id=user_id,
        save_approved_checkins=True
    )
    # create one saved check-in linked to the account
    create_checkin(
        user_id=user_id,
        question_set_version="2.0",
        checkin_status="Experiencing some strain today",
        core_answers={
            "sleep": 2,
            "mood": 3,
            "stress": 4,
            "support": 2,
        },
        context_answers={
            "food": 3,
            "medication": 0,
            "physical_recovery": 3,
            "baby_care": 4,
            "other_responsibilities": 2,
            "personal_care": 2,
        }
    )

    assert len(get_checkins_for_user(user_id)) == 1
    # incorrect password should not delete the account or its data
    wrong = delete_account_with_password(
        user_id=user_id,
        username="maya",
        password="WrongPassword!"
    )
    assert wrong["success"] is False
    assert get_user_by_id(user_id) is not None
    assert len(get_checkins_for_user(user_id)) == 1
    # correct password should delete the account
    deleted = delete_account_with_password(
        user_id=user_id,
        username="maya",
        password=PASS
    )
    assert deleted["success"] is True
    # related privacy preferences and check-ins should also be removed
    assert get_user_by_id(user_id) is None
    assert get_privacy_preferences(user_id) is None
    assert get_checkins_for_user(user_id) == []