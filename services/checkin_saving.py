
# sqlite is used to catch database errors during saving
import sqlite3
# imports the create and update functions for structured check-ins
from database.checkins import create_checkin, update_checkin
# imports the create and update functions for reflection-only entries
from database.reflections import create_saved_reflection, update_saved_reflection
# checks whether the user's current privacy settings allow approved data to be saved
from database.privacy import can_save_approved_checkins

# creates a consistent result dictionary for structured check-in saving
def create_save_result(success, message, checkin=None, action=""):

    return {
        "success": success,
        "message": message,
        "checkin": checkin,
        "action": action,
    }


# creates a consistent result dictionary for reflection-only saving
def create_reflection_save_result(
    success,
    message,
    reflection=None,
    action="",
):
    # return result for the reflection saving

    return {
        "success": success,
        "message": message,
        "reflection": reflection,
        "action": action,
    }


# structured check in
# creates a new approved check-in or updates an existing saved one
def save_or_update_approved_checkin(
    user_id,
    checkin_data,
    summary_confirmed,
    existing_checkin_id=None,
):
    # create or update an approved structured check in
    # saving is only available for logged-in users
    if not user_id:
        return create_save_result(
            False,
            "Please log in before saving a check-in.",
        )

    # the reviewed summary and appointment points must be confirmed first
    if not summary_confirmed:
        return create_save_result(
            False,
            "Please review and confirm the personal summary and "
            "appointment discussion points before saving.",
        )

    try:
        saving_allowed = can_save_approved_checkins(user_id)
    except sqlite3.Error:
        return create_save_result(
            False,
            "Your privacy preferences could not be checked.",
        )

    if not saving_allowed:
        return create_save_result(
            False,
            "Saving approved information is currently turned off "
            "in your privacy preferences.",
        )

    if not isinstance(checkin_data, dict):
        return create_save_result(
            False,
            "The approved check-in information is incomplete.",
        )

    try:
        # update the existing record when a saved check-in id is provided
        if existing_checkin_id:
            checkin = update_checkin(
                checkin_id=existing_checkin_id,
                user_id=user_id,
                **checkin_data,
            )

            return create_save_result(
                True,
                "Your saved check-in was updated.",
                checkin,
                "updated",
            )

        # otherwise create a new approved check-in
        checkin = create_checkin(
            user_id=user_id,
            **checkin_data,
        )

        return create_save_result(
            True,
            "Your approved check-in was saved.",
            checkin,
            "created",
        )

    except ValueError as error:
        return create_save_result(False, str(error))

    except sqlite3.Error:
        return create_save_result(
            False,
            "The check-in could not be saved. Please try again.",
        )


# only reflection
# creates or updates a saved reflection when no structured check-in is required
def save_or_update_approved_reflection(
    user_id,
    reflection_data,
    summary_confirmed,
    existing_reflection_id=None,
):
    # create or update a reflection only entry without structured questions or check-in status.

    # reflection saving also requires a logged-in user
    if not user_id:
        return create_reflection_save_result(
            False,
            "Please log in before saving a reflection.",
        )

    # only confirmed summary content can be saved
    if not summary_confirmed:
        return create_reflection_save_result(
            False,
            "Please review and confirm the personal summary and "
            "appointment discussion points before saving.",
        )

    try:
        saving_allowed = can_save_approved_checkins(user_id)
    except sqlite3.Error:
        return create_reflection_save_result(
            False,
            "Your privacy preferences could not be checked.",
        )

    if not saving_allowed:
        return create_reflection_save_result(
            False,
            "Saving approved information is currently turned off "
            "in your privacy preferences.",
        )

    if not isinstance(reflection_data, dict):
        return create_reflection_save_result(
            False,
            "The approved reflection information is incomplete.",
        )

    try:
        # update the saved reflection when an existing id is provided
        if existing_reflection_id:
            reflection = update_saved_reflection(
                reflection_id=existing_reflection_id,
                user_id=user_id,
                **reflection_data,
            )

            return create_reflection_save_result(
                True,
                "Your saved reflection was updated.",
                reflection,
                "updated",
            )

        # otherwise create a new reflection-only record
        reflection = create_saved_reflection(
            user_id=user_id,
            **reflection_data,
        )

        return create_reflection_save_result(
            True,
            "Your approved reflection was saved.",
            reflection,
            "created",
        )

    except ValueError as error:
        return create_reflection_save_result(False, str(error))

    except sqlite3.Error:
        return create_reflection_save_result(
            False,
            "The reflection could not be saved. Please try again.",
        )