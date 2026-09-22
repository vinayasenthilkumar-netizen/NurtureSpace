# very important note for understanding
# database functions for approved check in
# only reviewed results are stored
# raw media transcripts and raw model outputs are not saved

# convert python values to and and or from json for database storage
import json
# question keys used to validate structured check in answers
from core.constants import CORE_STATUS_QUESTION_KEYS, CONTEXT_QUESTION_KEYS
# opens a connection to the local application database
from database.connection import open_database
# used to check that a user exists before saving a check in
from database.users import get_user_by_id


# converts the python values intoto json text before storing in the database
def encode_json(value):
    return json.dumps(value, ensure_ascii=False)

# converts the stored json text back to python values
def decode_json(value, default_value):
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        # return safe default if stored value cannot be decoded
        return default_value


# cleans approved lists before they are saved
def clean_text_list(items):
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("Approved items must be supplied as a list.")
    result = []
    # remove empty values and repeated items
    for item in items:
        item = str(item or "").strip()
        if item and item not in result:
            result.append(item)
    return result


# checks that optional wellbeing score is valid before saving it
def clean_optional_score(score, score_name):
    if score is None:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        raise ValueError(f"{score_name} must be a number.") from None
    # all saved wellbeing scores use the same 1 to 5 range
    if not 1.0 <= score <= 5.0:
        raise ValueError(f"{score_name} must be between 1 and 5.")
    return round(score, 2)

# validates the four core questions used for the question score and the status idicator
def validate_core_answers(core_answers):
    if not isinstance(core_answers, dict):
        raise ValueError("Core answers must be supplied as a dictionary.")
    for key in CORE_STATUS_QUESTION_KEYS:
        if key not in core_answers:
            raise ValueError(f"Core answer is missing: {key}.")
        value = core_answers[key]

        # core answers have to be whole numbers from 1 to 5
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"Core answer is invalid: {key}.")


# validates the context questions that give extra check in information
def validate_context_answers(context_answers):
    if not isinstance(context_answers, dict):
        raise ValueError("Context answers must be supplied as a dictionary.")
    for key in CONTEXT_QUESTION_KEYS:
        if key not in context_answers:
            raise ValueError(f"Context answer is missing: {key}.")
        value = context_answers[key]
        # context questions allow values from 0 to 5
        if not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"Context answer is invalid: {key}.")


# converts the  database row into a check in dictionary used by the application
def row_to_checkin(row):
    if row is None:
        return None

    checkin = dict(row)

    # convert the json fields into regular python dictionary and list
    checkin["core_answers"] = decode_json(
        checkin.pop("core_answers_json"), {}
    )
    checkin["context_answers"] = decode_json(
        checkin.pop("context_answers_json"), {}
    )
    checkin["approved_text_emotions"] = decode_json(
        checkin.pop("approved_text_emotions_json"), []
    )
    checkin["approved_supportive_factors"] = decode_json(
        checkin.pop("approved_supportive_factors_json"), []
    )
    checkin["approved_strain_factors"] = decode_json(
        checkin.pop("approved_strain_factors_json"), []
    )
    checkin["confirmed_appointment_points"] = decode_json(
        checkin.pop("confirmed_appointment_points_json"), []
    )

    return checkin


# validate and prepare the reviewed check in values before database storage
def prepare_checkin_values(
    question_set_version,
    checkin_status,
    core_answers,
    context_answers,
    reflection_mode="",
    question_score=None,
    reflection_score=None,
    reflection_indicator="",
    combined_wellbeing_score=None,
    combined_wellbeing_indicator="",
    approved_text_emotions=None,
    approved_supportive_factors=None,
    approved_strain_factors=None,
    approved_vocal_observation="",
    approved_visible_observation="",
    confirmed_summary="",
    confirmed_appointment_points=None,
):
    # clean the text values before validation and storage
    version = str(question_set_version or "").strip()
    status = str(checkin_status or "").strip()
    mode = str(reflection_mode or "").strip()

    if not version:
        raise ValueError("A question set version is required.")

    if not status:
        raise ValueError("A check in status is required.")

    # validate both the core and context structured question sections before saving
    validate_core_answers(core_answers)
    validate_context_answers(context_answers)

    vocal = str(approved_vocal_observation or "").strip()
    visible = str(approved_visible_observation or "").strip()

    # prepare the final approved values in the database formate needed
    return {
        "question_set_version": version,
        "checkin_status": status,
        "reflection_mode": mode or None,
        "core_answers_json": encode_json(core_answers),
        "context_answers_json": encode_json(context_answers),
        "question_score": clean_optional_score(
            question_score, "Question Score"
        ),
        "reflection_score": clean_optional_score(
            reflection_score, "Reflection Score"
        ),
        "reflection_indicator": (
            str(reflection_indicator or "").strip() or None
        ),
        "combined_wellbeing_score": clean_optional_score(
            combined_wellbeing_score,
            "Combined Wellbeing Score",
        ),
        "combined_wellbeing_indicator": (
            str(combined_wellbeing_indicator or "").strip() or None
        ),
        "approved_text_emotions_json": encode_json(
            clean_text_list(approved_text_emotions)
        ),
        "approved_supportive_factors_json": encode_json(
            clean_text_list(approved_supportive_factors)
        ),
        "approved_strain_factors_json": encode_json(
            clean_text_list(approved_strain_factors)
        ),
        "approved_vocal_observation": vocal or None,
        "approved_visible_observation": visible or None,
        "confirmed_summary": str(confirmed_summary or "").strip(),
        "confirmed_appointment_points_json": encode_json(
            clean_text_list(confirmed_appointment_points)
        ),
    }


# creates a new approved check in for a registered user
def create_checkin(
    user_id,
    question_set_version,
    checkin_status,
    core_answers,
    context_answers,
    reflection_mode="",
    question_score=None,
    reflection_score=None,
    reflection_indicator="",
    combined_wellbeing_score=None,
    combined_wellbeing_indicator="",
    approved_text_emotions=None,
    approved_supportive_factors=None,
    approved_strain_factors=None,
    approved_vocal_observation="",
    approved_visible_observation="",
    confirmed_summary="",
    confirmed_appointment_points=None,
):
    # check if the  check in belongs to a valid user account
    user = get_user_by_id(user_id)

    if user is None:
        raise ValueError("The user account could not be found.")

    # validate and prepare all approved values before inserting them
    values = prepare_checkin_values(
        question_set_version=question_set_version,
        checkin_status=checkin_status,
        core_answers=core_answers,
        context_answers=context_answers,
        reflection_mode=reflection_mode,
        question_score=question_score,
        reflection_score=reflection_score,
        reflection_indicator=reflection_indicator,
        combined_wellbeing_score=combined_wellbeing_score,
        combined_wellbeing_indicator=combined_wellbeing_indicator,
        approved_text_emotions=approved_text_emotions,
        approved_supportive_factors=approved_supportive_factors,
        approved_strain_factors=approved_strain_factors,
        approved_vocal_observation=approved_vocal_observation,
        approved_visible_observation=approved_visible_observation,
        confirmed_summary=confirmed_summary,
        confirmed_appointment_points=confirmed_appointment_points,
    )

    # save the approved check in using parameterised sql values
    with open_database() as connection:
        cursor = connection.execute(
            """
            INSERT INTO checkins (
                user_id,
                question_set_version,
                checkin_status,
                reflection_mode,
                core_answers_json,
                context_answers_json,
                question_score,
                reflection_score,
                reflection_indicator,
                combined_wellbeing_score,
                combined_wellbeing_indicator,
                approved_text_emotions_json,
                approved_supportive_factors_json,
                approved_strain_factors_json,
                approved_vocal_observation,
                approved_visible_observation,
                confirmed_summary,
                confirmed_appointment_points_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                values["question_set_version"],
                values["checkin_status"],
                values["reflection_mode"],
                values["core_answers_json"],
                values["context_answers_json"],
                values["question_score"],
                values["reflection_score"],
                values["reflection_indicator"],
                values["combined_wellbeing_score"],
                values["combined_wellbeing_indicator"],
                values["approved_text_emotions_json"],
                values["approved_supportive_factors_json"],
                values["approved_strain_factors_json"],
                values["approved_vocal_observation"],
                values["approved_visible_observation"],
                values["confirmed_summary"],
                values["confirmed_appointment_points_json"],
            ),
        )

        # generated database id to return the new saved check in
        checkin_id = cursor.lastrowid

        row = connection.execute(
            "SELECT * FROM checkins WHERE id = ?",
            (checkin_id,),
        ).fetchone()

    return row_to_checkin(row)


# updates an existing chec belonging to the specified user
def update_checkin(checkin_id, user_id, **checkin_data):
    existing = get_checkin_by_id(checkin_id, user_id)

    if existing is None:
        raise ValueError("The saved check-in could not be found.")

    # run the same validation used when a check in is first created
    values = prepare_checkin_values(**checkin_data)

    with open_database() as connection:
        connection.execute(
            """
            UPDATE checkins
            SET
                question_set_version = ?,
                checkin_status = ?,
                reflection_mode = ?,
                core_answers_json = ?,
                context_answers_json = ?,
                question_score = ?,
                reflection_score = ?,
                reflection_indicator = ?,
                combined_wellbeing_score = ?,
                combined_wellbeing_indicator = ?,
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
                values["question_set_version"],
                values["checkin_status"],
                values["reflection_mode"],
                values["core_answers_json"],
                values["context_answers_json"],
                values["question_score"],
                values["reflection_score"],
                values["reflection_indicator"],
                values["combined_wellbeing_score"],
                values["combined_wellbeing_indicator"],
                values["approved_text_emotions_json"],
                values["approved_supportive_factors_json"],
                values["approved_strain_factors_json"],
                values["approved_vocal_observation"],
                values["approved_visible_observation"],
                values["confirmed_summary"],
                values["confirmed_appointment_points_json"],
                int(checkin_id),
                int(user_id),
            ),
        )

    # return the latest saved version after the update
    return get_checkin_by_id(checkin_id, user_id)


# updates only the saved personal summary and appointment discussion points
def update_checkin_summary_content(
    checkin_id,
    user_id,
    confirmed_summary=None,
    confirmed_appointment_points=None,
):
    # update only the saved Personal Summary and AppointmentDiscussion Points.

    existing = get_checkin_by_id(checkin_id, user_id)

    if existing is None:
        raise ValueError("The saved check-in could not be found.")

    # keep the existing summary when no replacement is supplied
    if confirmed_summary is None:
        summary = str(existing.get("confirmed_summary", "") or "").strip()
    else:
        summary = str(confirmed_summary or "").strip()

    if not summary:
        raise ValueError("The personal summary cannot be empty.")

    # keep the existing appointment points unless new ones are provided
    if confirmed_appointment_points is None:
        appointment_points = existing.get(
            "confirmed_appointment_points", []
        ) or []
    else:
        appointment_points = clean_text_list(
            confirmed_appointment_points
        )

    # only these two user editable summary fields are changed here
    with open_database() as connection:
        connection.execute(
            """
            UPDATE checkins
            SET
                confirmed_summary = ?,
                confirmed_appointment_points_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                summary,
                encode_json(appointment_points),
                int(checkin_id),
                int(user_id),
            ),
        )

    return get_checkin_by_id(checkin_id, user_id)


# gets one saved check in and can restrict the lookup to a specific user
def get_checkin_by_id(checkin_id, user_id=None):
    try:
        checkin_id = int(checkin_id)
    except (TypeError, ValueError):
        return None

    with open_database() as connection:
        # allow an id only lookup when no user id is supplied
        if user_id is None:
            row = connection.execute(
                "SELECT * FROM checkins WHERE id = ?",
                (checkin_id,),
            ).fetchone()
        else:
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                return None

            # when a user id is suppliedmake sure the check in belongs to that user
            row = connection.execute(
                """
                SELECT *
                FROM checkins
                WHERE id = ? AND user_id = ?
                """,
                (checkin_id, user_id),
            ).fetchone()

    return row_to_checkin(row)


# gets all saved check ins for a user with the newest records first
def get_checkins_for_user(user_id):
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return []

    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM checkins
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
        ).fetchall()
    # convert every database row into the format used by the application
    return [row_to_checkin(row) for row in rows]


# deletes one saved check in only when it belongs to the specified user
def delete_checkin(checkin_id, user_id):
    try:
        checkin_id = int(checkin_id)
        user_id = int(user_id)
    except (TypeError, ValueError):
        return False
    with open_database() as connection:
        cursor = connection.execute(
            """
            DELETE FROM checkins
            WHERE id = ? AND user_id = ?
            """,
            (checkin_id, user_id),
        )

        # return true only when a database row was actually deleted
        return cursor.rowcount > 0