
# imports the current privacy notice version used by the application
from config.settings import CURRENT_PRIVACY_VERSION
# opens a connection to the local sqlite database
from database.connection import open_database
# used to check that the privacy preferences belong to an existing user
from database.users import get_user_by_id


# checks and cleans a user id before it is used in a database query
def clean_user_id(user_id):
    # Return a valid positive user ID or None.
    try:
        clean_id = int(user_id)
    except (TypeError, ValueError):
        return None
    # only positive numbers are accepted as valid user ids
    return clean_id if clean_id > 0 else None


# converts python boolean values into values that can be stored in sqlite
def to_database_boolean(value):
    if value is True or value == 1:
        return 1

    if value is False or value == 0:
        return 0

    # reject values that are not clear privacy choices
    raise ValueError("Privacy choices must be True or False.")


# converts a database row into privacy preferences used by the application
def row_to_privacy_preferences(row):
    if row is None:
        return None
    preferences = dict(row)
    # these values are stored as 1 or 0 in sqlite
    boolean_fields = [
        "save_approved_checkins",
        "use_saved_history_for_personalisation",
        "save_bookmarked_resources",
    ]
    # convert stored integer values back into python booleans
    for field in boolean_fields:
        preferences[field] = bool(preferences[field])
    return preferences

# gets the saved privacy choices for one user
def get_privacy_preferences(user_id):
    clean_id = clean_user_id(user_id)

    if clean_id is None:
        return None

    # only retrieve the privacy record linked to this user
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM privacy_preferences
            WHERE user_id = ?
            """,
            (clean_id,),
        ).fetchone()

    return row_to_privacy_preferences(row)

# creates or updates the privacy choices for a registered user
def save_privacy_preferences(
    user_id,
    save_approved_checkins=False,
    use_saved_history_for_personalisation=False,
    save_bookmarked_resources=False,
    privacy_version=CURRENT_PRIVACY_VERSION,
):
    clean_id = clean_user_id(user_id)

    if clean_id is None:
        raise ValueError("A valid user ID is required.")

    # make sure privacy preferences are only saved for an existing account
    if get_user_by_id(clean_id) is None:
        raise ValueError("The user account could not be found.")
    # convert privacy choices into sqlite friendly values
    save_checkins = to_database_boolean(save_approved_checkins)
    personalisation = to_database_boolean(
        use_saved_history_for_personalisation
    )
    save_bookmarks = to_database_boolean(save_bookmarked_resources)

    clean_version = str(privacy_version or "").strip()

    if not clean_version:
        raise ValueError("A privacy version is required.")

    # insert a new preference record or update the existing one for the user
    with open_database() as connection:
        connection.execute(
            """
            INSERT INTO privacy_preferences (
                user_id,
                save_approved_checkins,
                use_saved_history_for_personalisation,
                save_bookmarked_resources,
                privacy_version
            )
            VALUES (?, ?, ?, ?, ?)

            ON CONFLICT(user_id)
            DO UPDATE SET
                save_approved_checkins =
                    excluded.save_approved_checkins,
                use_saved_history_for_personalisation =
                    excluded.use_saved_history_for_personalisation,
                save_bookmarked_resources =
                    excluded.save_bookmarked_resources,
                privacy_version =
                    excluded.privacy_version,
                accepted_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                clean_id,
                save_checkins,
                personalisation,
                save_bookmarks,
                clean_version,
            ),
        )

        # read the saved record again so the latest values can be returned
        row = connection.execute(
            """
            SELECT *
            FROM privacy_preferences
            WHERE user_id = ?
            """,
            (clean_id,),
        ).fetchone()

    return row_to_privacy_preferences(row)

# checks whether the user has accepted the current privacy notice version
def privacy_is_current(user_id):
    preferences = get_privacy_preferences(user_id)
    if preferences is None:
        return False
    # an older privacy version means the current notice has not been accepted
    return preferences["privacy_version"] == CURRENT_PRIVACY_VERSION

# checks whether the users current privacy settings allow check  ins to be saved
def can_save_approved_checkins(user_id):
    preferences = get_privacy_preferences(user_id)
    if preferences is None:
        return False
    # saving is blocked until the current privacy version has been accepted
    if preferences["privacy_version"] != CURRENT_PRIVACY_VERSION:
        return False
    # use the users saved choice to decide whether approved check  ins may persist
    return bool(preferences["save_approved_checkins"])