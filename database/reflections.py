

# imports json so lists can be stored as text in sqlite and converted back later
import json
# three reflection modes accepted by the application
from core.constants import TEXT_REFLECTION, VIDEO_REFLECTION, VOICE_REFLECTION
# used to open and manage sqlite connections
from database.connection import open_database
# lookup function to confirm that a user exists before saving
from database.users import get_user_by_id


# convrts a python value into json text for database storage
def encode_json(value):
    # keep normal unicode characters instead of converting them into escape codes
    return json.dumps(value, ensure_ascii=False)


# converts stored json text back into its original python value
def decode_json(value, default_value):
    # try to decode the stored json value
    try:
        return json.loads(value)

    # return the supplied default if the stored value cannot be decoded
    except (TypeError, ValueError, json.JSONDecodeError):
        return default_value


# cleans approved text items before they are stored
def clean_text_list(items):
    # treat a missing list as an empty list
    if items is None:
        return []

    # approved items must be passed as a list
    if not isinstance(items, list):
        raise ValueError("Approved items must be supplied as a list.")

    # create a new list for cleaned values
    result = []

    # process every supplied item
    for item in items:
        # convert the value to text and remove surrounding spaces
        item = str(item or "").strip()

        # keep only not empty items that have not already been added
        if item and item not in result:
            result.append(item)

    # return the cleaned list
    return result


# checks and cleans an optional reflection score
def clean_optional_score(score):
    # a reflection score can be absent
    if score is None:
        return None

    # try to convert the supplied score into a number
    try:
        score = float(score)

    # reject values that cannot be converted into a number
    except (TypeError, ValueError):
        raise ValueError("Reflection Score must be a number.") from None

    # reflection scores must remain within the applications 1 to 5 scale
    if not 1.0 <= score <= 5.0:
        raise ValueError("Reflection Score must be between 1 and 5.")

    # store scores to two decimal places
    return round(score, 2)


# validates and prepares approved reflection values before database storage
def prepare_saved_reflection_values(
    # selected reflection type
    reflection_mode,
    # calculated reflection score if available
    reflection_score=None,
    # text label linked to the reflection score
    reflection_indicator="",
    # approved text emotion labels
    approved_text_emotions=None,
    # approved supportive factors
    approved_supportive_factors=None,
    # approved strain factors
    approved_strain_factors=None,
    # approved observation from the voice signal
    approved_vocal_observation="",
    # approved observation from visible-expression analysis
    approved_visible_observation="",
    # confirmed personal summary
    confirmed_summary="",
    # confirmed appointment discussion points
    confirmed_appointment_points=None,
):
    # convert the reflection mode into clean text
    mode = str(reflection_mode or "").strip()

    # only the three supported reflection modes are accepted
    if mode not in {TEXT_REFLECTION, VOICE_REFLECTION, VIDEO_REFLECTION}:
        raise ValueError("A valid reflection mode is required.")

    # clean the confirmed personal summary
    summary = str(confirmed_summary or "").strip()

    # a reflection cannot be saved without a confirmed personal summary
    if not summary:
        raise ValueError(
            "A confirmed personal summary is required before saving."
        )

    # clean the approved vocal observation
    vocal = str(approved_vocal_observation or "").strip()

    # clean the approved visible-expression observation
    visible = str(approved_visible_observation or "").strip()

    # return the cleaned values in the format expected by the database
    return {
        # save the selected reflection mode
        "reflection_mode": mode,

        # validate and save the reflection score
        "reflection_score": clean_optional_score(reflection_score),

        # store an empty indicator as null
        "reflection_indicator": (
            str(reflection_indicator or "").strip() or None
        ),

        # clean the approved text emotions and encode them as json
        "approved_text_emotions_json": encode_json(
            clean_text_list(approved_text_emotions)
        ),

        # clean the supportive factors and encode them as json
        "approved_supportive_factors_json": encode_json(
            clean_text_list(approved_supportive_factors)
        ),

        # clean the strain factors and encode them as json
        "approved_strain_factors_json": encode_json(
            clean_text_list(approved_strain_factors)
        ),

        # save the vocal observation or null when it is empty
        "approved_vocal_observation": vocal or None,

        # save the visible observation or null when it is empty
        "approved_visible_observation": visible or None,

        # save the confirmed personal summary
        "confirmed_summary": summary,

        # clean the appointment points and encode them as json
        "confirmed_appointment_points_json": encode_json(
            clean_text_list(confirmed_appointment_points)
        ),
    }


# converts a saved reflection database row into a dictionary used by the application
def row_to_saved_reflection(row):
    # return nothing when no database row was found
    if row is None:
        return None

    # convert the sqlite row into a normal dictionary
    reflection = dict(row)

    # decode the stored text-emotion json into a python list
    reflection["approved_text_emotions"] = decode_json(
        reflection.pop("approved_text_emotions_json"), []
    )

    # decode the stored supportive-factor json into a python list
    reflection["approved_supportive_factors"] = decode_json(
        reflection.pop("approved_supportive_factors_json"), []
    )

    # decode the stored strain-factor json into a python list
    reflection["approved_strain_factors"] = decode_json(
        reflection.pop("approved_strain_factors_json"), []
    )

    # decode the stored appointment-point json into a python list
    reflection["confirmed_appointment_points"] = decode_json(
        reflection.pop("confirmed_appointment_points_json"), []
    )

    # return the converted reflection record
    return reflection


# creates a new saved reflection-only record for a registered user
def create_saved_reflection(
    # user who owns the reflection
    user_id,
    # text, voice or video reflection mode
    reflection_mode,
    # calculated reflection score
    reflection_score=None,
    # score interpretation label
    reflection_indicator="",
    # approved text emotions
    approved_text_emotions=None,
    # approved supportive factors
    approved_supportive_factors=None,
    # approved strain factors
    approved_strain_factors=None,
    # approved voice observation
    approved_vocal_observation="",
    # approved visible-expression observation
    approved_visible_observation="",
    # confirmed personal summary
    confirmed_summary="",
    # confirmed appointment discussion points
    confirmed_appointment_points=None,
):
    # retrieve the user account before saving the reflection
    user = get_user_by_id(user_id)

    # do not save a reflection for an account that does not exist
    if user is None:
        raise ValueError("The user account could not be found.")

    # validate and prepare all approved reflection values
    values = prepare_saved_reflection_values(
        # pass the selected reflection mode
        reflection_mode=reflection_mode,
        # pass the calculated reflection score
        reflection_score=reflection_score,
        # pass the reflection indicator
        reflection_indicator=reflection_indicator,
        # pass the approved text emotions
        approved_text_emotions=approved_text_emotions,
        # pass the approved supportive factors
        approved_supportive_factors=approved_supportive_factors,
        # pass the approved strain factors
        approved_strain_factors=approved_strain_factors,
        # pass the approved vocal observation
        approved_vocal_observation=approved_vocal_observation,
        # pass the approved visible observation
        approved_visible_observation=approved_visible_observation,
        # pass the confirmed summary
        confirmed_summary=confirmed_summary,
        # pass the confirmed appointment points
        confirmed_appointment_points=confirmed_appointment_points,
    )

    # open the database connection for the insert
    with open_database() as connection:
        # insert only the approved reflection information
        cursor = connection.execute(
            """
            INSERT INTO saved_reflections (
                user_id,
                reflection_mode,
                reflection_score,
                reflection_indicator,
                approved_text_emotions_json,
                approved_supportive_factors_json,
                approved_strain_factors_json,
                approved_vocal_observation,
                approved_visible_observation,
                confirmed_summary,
                confirmed_appointment_points_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                # save the confirmed database user id
                user["id"],

                # save the reflection mode
                values["reflection_mode"],

                # save the reflection score
                values["reflection_score"],

                # save the reflection indicator
                values["reflection_indicator"],

                # save approved text emotions as json
                values["approved_text_emotions_json"],

                # save supportive factors as json
                values["approved_supportive_factors_json"],

                # save strain factors as json
                values["approved_strain_factors_json"],

                # save the approved vocal observation
                values["approved_vocal_observation"],

                # save the approved visible observation
                values["approved_visible_observation"],

                # save the confirmed personal summary
                values["confirmed_summary"],

                # save the appointment points as json
                values["confirmed_appointment_points_json"],
            ),
        )

        # get the id automatically created for the new reflection
        reflection_id = cursor.lastrowid

        # retrieve the new reflection so the saved record can be returned
        row = connection.execute(
            "SELECT * FROM saved_reflections WHERE id = ?",
            (reflection_id,),
        ).fetchone()

    # convert the database row into the format used by the application
    return row_to_saved_reflection(row)


# retrieves one saved reflection by its id
def get_saved_reflection_by_id(reflection_id, user_id=None):
    # make sure the reflection id is a valid integer
    try:
        reflection_id = int(reflection_id)

    # return nothing when an invalid id is supplied
    except (TypeError, ValueError):
        return None

    # open the database connection for the lookup
    with open_database() as connection:

        # allow an id-only lookup when no user id is supplied
        if user_id is None:
            row = connection.execute(
                "SELECT * FROM saved_reflections WHERE id = ?",
                (reflection_id,),
            ).fetchone()

        # restrict the lookup to a specific user when a user id is supplied
        else:
            # make sure the user id is a valid integer
            try:
                user_id = int(user_id)

            # return nothing when the user id is invalid
            except (TypeError, ValueError):
                return None

            # retrieve the reflection only when both ids match
            row = connection.execute(
                """
                SELECT *
                FROM saved_reflections
                WHERE id = ? AND user_id = ?
                """,
                (reflection_id, user_id),
            ).fetchone()

    # convert the database row before returning it
    return row_to_saved_reflection(row)


# retrieves all saved reflection-only records belonging to a user
def get_saved_reflections_for_user(user_id):
    # make sure the user id can be used safely in the query
    try:
        user_id = int(user_id)

    # return an empty list when the user id is invalid
    except (TypeError, ValueError):
        return []

    # open the database and retrieve the user's reflections
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM saved_reflections
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()

    # convert every database row into the format used by the application
    return [row_to_saved_reflection(row) for row in rows]


# updates all approved information in an existing saved reflection
def update_saved_reflection(reflection_id, user_id, **reflection_data):
    # retrieve the existing reflection and confirm that it belongs to the user
    existing = get_saved_reflection_by_id(reflection_id, user_id)

    # stop if the requested saved reflection does not exist
    if existing is None:
        raise ValueError("The saved reflection could not be found.")

    # validate and prepare the replacement reflection values
    values = prepare_saved_reflection_values(**reflection_data)

    # open the database connection for the update
    with open_database() as connection:
        # update the approved reflection fields
        connection.execute(
            """
            UPDATE saved_reflections
            SET
                reflection_mode = ?,
                reflection_score = ?,
                reflection_indicator = ?,
                approved_text_emotions_json = ?,
                approved_supportive_factors_json = ?,
                approved_strain_factors_json = ?,
                approved_vocal_observation = ?,
                approved_visible_observation = ?,
                confirmed_summary = ?,
                confirmed_appointment_points_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                # update the reflection mode
                values["reflection_mode"],

                # update the reflection score
                values["reflection_score"],

                # update the reflection indicator
                values["reflection_indicator"],

                # update the approved text emotions
                values["approved_text_emotions_json"],

                # update the supportive factors
                values["approved_supportive_factors_json"],

                # update the strain factors
                values["approved_strain_factors_json"],

                # update the approved vocal observation
                values["approved_vocal_observation"],

                # update the approved visible observation
                values["approved_visible_observation"],

                # update the confirmed personal summary
                values["confirmed_summary"],

                # update the confirmed appointment points
                values["confirmed_appointment_points_json"],

                # identify the reflection being updated
                int(reflection_id),

                # make sure it belongs to the correct user
                int(user_id),
            ),
        )

    # return the latest saved version after the update
    return get_saved_reflection_by_id(reflection_id, user_id)


# updates only the saved personal summary and appointment discussion points
def update_reflection_summary_content(
    # reflection that should be changed
    reflection_id,
    # user who owns the reflection
    user_id,
    # optional replacement personal summary
    confirmed_summary=None,
    # optional replacement appointment discussion points
    confirmed_appointment_points=None,
):
    # retrieve the existing saved reflection
    existing = get_saved_reflection_by_id(reflection_id, user_id)

    # stop when the reflection cannot be found for this user
    if existing is None:
        raise ValueError("The saved reflection could not be found.")

    # keep the existing summary when no replacement was supplied
    if confirmed_summary is None:
        summary = str(existing.get("confirmed_summary", "") or "").strip()

    # otherwise clean the replacement summary
    else:
        summary = str(confirmed_summary or "").strip()

    # the saved personal summary cannot be left empty
    if not summary:
        raise ValueError("The personal summary cannot be empty.")

    # keep the existing appointment points when no replacement is supplied
    if confirmed_appointment_points is None:
        appointment_points = existing.get(
            "confirmed_appointment_points", []
        ) or []

    # otherwise clean the replacement appointment points
    else:
        appointment_points = clean_text_list(
            confirmed_appointment_points
        )

    # open the database connection to update only these two fields
    with open_database() as connection:
        connection.execute(
            """
            UPDATE saved_reflections
            SET
                confirmed_summary = ?,
                confirmed_appointment_points_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                # save the confirmed summary
                summary,

                # convert appointment points to json before saving
                encode_json(appointment_points),

                # identify the reflection being updated
                int(reflection_id),

                # make sure the reflection belongs to this user
                int(user_id),
            ),
        )

    # return the updated reflection
    return get_saved_reflection_by_id(reflection_id, user_id)


# deletes one saved reflection belonging to a specific user
def delete_saved_reflection(reflection_id, user_id):
    # make sure both ids are valid integers
    try:
        reflection_id = int(reflection_id)
        user_id = int(user_id)

    # return false when either id is invalid
    except (TypeError, ValueError):
        return False

    # open the database connection for the delete operation
    with open_database() as connection:
        # delete only when both the reflection id and user id match
        cursor = connection.execute(
            """
            DELETE FROM saved_reflections
            WHERE id = ? AND user_id = ?
            """,
            (reflection_id, user_id),
        )

        # return true only if a database row was actually deleted
        return cursor.rowcount > 0