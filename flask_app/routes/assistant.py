
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from core.constants import (
    BABY_CARE_LABELS,
    FOOD_LABELS,
    MEDICATION_LABELS,
    MOOD_LABELS,
    OTHER_RESPONSIBILITIES_LABELS,
    PERSONAL_CARE_LABELS,
    PHYSICAL_RECOVERY_LABELS,
    SLEEP_LABELS,
    STRESS_LABELS,
    SUPPORT_LABELS,
)
from database.checkins import (
    get_checkins_for_user,
    update_checkin_summary_content,
)
from database.privacy import get_privacy_preferences, privacy_is_current
from database.reflections import (
    get_saved_reflections_for_user,
    update_reflection_summary_content,
)
from database.resources import (
    bookmark_resource,
    get_bookmarked_resource_ids,
    get_resource_by_id,
    remove_resource_bookmark,
)
from flask_app.routes import account_required, can_save_bookmarks
from graphs.assistant_graph import assistant_graph
from patterns.history import get_all_patterns_for_user
from patterns.relevance import select_relevant_patterns


assistant_blueprint = Blueprint(
    "assistant",
    __name__,
    url_prefix="/assistant",
)

login_required = account_required(
    "Please log in to use the Wellbeing Assistant.",
    "warning",
)

SINGAPORE_TIMEZONE = ZoneInfo("Asia/Singapore")

SUMMARY_UNDO_KEY = "assistant_summary_undo"
APPOINTMENT_UNDO_KEY = "assistant_appointment_undo"

ASSISTANT_EDIT_TERMS = (
    "add",
    "include",
    "mention",
    "remove",
    "delete",
    "take out",
    "leave out",
    "drop",
    "get rid of",
    "rewrite",
    "reword",
    "refine",
    "shorten",
    "shorter",
    "simplify",
    "more concise",
    "change",
    "edit",
    "update",
)

SUMMARY_VIEW_TERMS = (
    "my personal summary",
    "my wellbeing summary",
    "my summary",
    "show my summary",
    "show me my summary",
    "give me my summary",
    "give my summary",
    "show my checkin",
    "show my check-in",
    "show me my checkin",
    "show me my check-in",
    "give me my checkin",
    "give me my check-in",
)

APPOINTMENT_VIEW_TERMS = (
    "my appointment points",
    "my appointment point",
    "my discussion points",
    "my discussion point",
    "show my appointment points",
    "show me my appointment points",
    "give me my appointment points",
    "give my appointment points",
)

CORE_RESPONSE_LABELS = {
    "sleep": SLEEP_LABELS,
    "mood": MOOD_LABELS,
    "stress": STRESS_LABELS,
    "support": SUPPORT_LABELS,
}

CONTEXT_RESPONSE_LABELS = {
    "food": FOOD_LABELS,
    "medication": MEDICATION_LABELS,
    "physical_recovery": PHYSICAL_RECOVERY_LABELS,
    "baby_care": BABY_CARE_LABELS,
    "other_responsibilities": OTHER_RESPONSIBILITIES_LABELS,
    "personal_care": PERSONAL_CARE_LABELS,
}

CONTEXT_MESSAGES = {
    "current_checkin": (
        "Using the approved information from your current check-in."
    ),
    "current_reflection": (
        "Using the approved information from your current reflection."
    ),
    "latest_saved_checkin": (
        "Using your latest saved check-in because saved-history "
        "personalisation is enabled."
    ),
    "latest_saved_reflection": (
        "Using your latest saved reflection because saved-history "
        "personalisation is enabled."
    ),
}

NO_CONTEXT_MESSAGE = (
    "No approved check-in context is being used right now. "
    "You can still ask general postpartum wellbeing questions."
)


def clean_text(value):
    return str(value or "").strip()


def clean_text_list(values):
    if not values:
        return []

    if isinstance(values, str):
        return [
            line.strip()
            for line in values.splitlines()
            if line.strip()
        ]

    if not isinstance(values, (list, tuple, set)):
        values = [values]

    cleaned = []

    for value in values:
        text = clean_text(value)

        if text and text not in cleaned:
            cleaned.append(text)

    return cleaned


def is_edit_request(user_message):
    text = clean_text(user_message).lower()

    return any(
        term in text
        for term in ASSISTANT_EDIT_TERMS
    )


def detect_saved_content_view_request(user_message):
    text = clean_text(user_message).lower()

    if not text or is_edit_request(text):
        return ""

    if any(term in text for term in APPOINTMENT_VIEW_TERMS):
        return "appointment_points"

    if any(term in text for term in SUMMARY_VIEW_TERMS):
        return "summary"

    return ""


def get_label(value, labels):
    if value is None:
        return ""

    if value in labels:
        return clean_text(labels[value])

    try:
        value = int(value)
    except (TypeError, ValueError):
        return ""

    return clean_text(labels.get(value, ""))


def build_labelled_responses(answers, label_sets):
    return {
        field: get_label(answers.get(field), labels)
        for field, labels in label_sets.items()
    }


def assistant_redirect():
    return redirect(url_for("assistant.assistant_page"))


def parse_saved_datetime(value):
    text = clean_text(value)

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        try:
            parsed = datetime.strptime(
                text,
                "%Y-%m-%d %H:%M:%S",
            )
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(SINGAPORE_TIMEZONE)


def get_current_core_answers():
    answers = session.get("core_answers", {}) or {}

    if answers:
        return dict(answers)

    return {
        "sleep": session.get("sleep_response"),
        "mood": session.get("mood_response"),
        "stress": session.get("stress_response"),
        "support": session.get("support_response"),
    }


def get_current_context_answers():
    answers = session.get("context_answers", {}) or {}

    if answers:
        return dict(answers)

    return {
        "food": session.get("food_response"),
        "medication": session.get("medication_response"),
        "physical_recovery": session.get("physical_recovery_response"),
        "baby_care": session.get("baby_care_response"),
        "other_responsibilities": session.get(
            "other_responsibilities_response"
        ),
        "personal_care": session.get("personal_care_response"),
    }


def get_approved_review_state():
    return {
        "approved_text_emotions": clean_text_list(
            session.get("approved_text_emotions", [])
        ),
        "approved_supportive_factors": clean_text_list(
            session.get("approved_supportive_factors", [])
        ),
        "approved_strain_factors": clean_text_list(
            session.get("approved_strain_factors", [])
        ),
        "approved_vocal_observation": clean_text(
            session.get("approved_audio_observation", "")
        ),
        "approved_visible_observation": clean_text(
            session.get("approved_visible_expression", "")
        ),
    }


def get_confirmed_dashboard_state():
    summary_confirmed = bool(
        session.get("summary_review_saved", False)
        or session.get("summary_confirmed", False)
        or session.get("dashboard_summary_confirmed", False)
    )

    if not summary_confirmed:
        return {
            "confirmed_summary": "",
            "confirmed_appointment_points": [],
        }

    return {
        "confirmed_summary": clean_text(
            session.get("personal_summary", "")
        ),
        "confirmed_appointment_points": clean_text_list(
            session.get("appointment_points", [])
        ),
    }


def build_current_assistant_state():
    status = clean_text(session.get("checkin_status", ""))
    review_complete = bool(session.get("review_complete", False))

    if status:
        context_source = "current_checkin"
    elif review_complete:
        context_source = "current_reflection"
    else:
        return {}

    state = {
        "context_source": context_source,
        "status": status,
        "core_responses": build_labelled_responses(
            get_current_core_answers(),
            CORE_RESPONSE_LABELS,
        ),
        "context_responses": build_labelled_responses(
            get_current_context_answers(),
            CONTEXT_RESPONSE_LABELS,
        ),
    }

    if context_source == "current_checkin":
        saved_id = session.get("saved_checkin_id")
        if saved_id:
            state["saved_entry_type"] = "checkin"
            state["saved_entry_id"] = saved_id
    else:
        saved_id = session.get("saved_reflection_id")
        if saved_id:
            state["saved_entry_type"] = "reflection"
            state["saved_entry_id"] = saved_id

    state.update(get_approved_review_state())
    state.update(get_confirmed_dashboard_state())

    return state


def build_saved_checkin_assistant_state(checkin):
    if not isinstance(checkin, dict):
        return {}

    core_answers = checkin.get("core_answers", {}) or {}
    context_answers = checkin.get("context_answers", {}) or {}

    return {
        "context_source": "latest_saved_checkin",
        "saved_entry_type": "checkin",
        "saved_entry_id": checkin.get("id"),
        "saved_entry_created_at": clean_text(
            checkin.get("created_at", "")
        ),
        "status": clean_text(
            checkin.get("checkin_status", "")
        ),
        "core_responses": build_labelled_responses(
            core_answers,
            CORE_RESPONSE_LABELS,
        ),
        "context_responses": build_labelled_responses(
            context_answers,
            CONTEXT_RESPONSE_LABELS,
        ),
        "approved_text_emotions": clean_text_list(
            checkin.get("approved_text_emotions", [])
        ),
        "approved_supportive_factors": clean_text_list(
            checkin.get("approved_supportive_factors", [])
        ),
        "approved_strain_factors": clean_text_list(
            checkin.get("approved_strain_factors", [])
        ),
        "approved_vocal_observation": clean_text(
            checkin.get("approved_vocal_observation", "")
        ),
        "approved_visible_observation": clean_text(
            checkin.get("approved_visible_observation", "")
        ),
        "confirmed_summary": clean_text(
            checkin.get("confirmed_summary", "")
        ),
        "confirmed_appointment_points": clean_text_list(
            checkin.get("confirmed_appointment_points", [])
        ),
    }


def build_saved_reflection_assistant_state(reflection):
    if not isinstance(reflection, dict):
        return {}

    return {
        "context_source": "latest_saved_reflection",
        "saved_entry_type": "reflection",
        "saved_entry_id": reflection.get("id"),
        "saved_entry_created_at": clean_text(
            reflection.get("created_at", "")
        ),
        "status": "",
        "core_responses": {},
        "context_responses": {},
        "approved_text_emotions": clean_text_list(
            reflection.get("approved_text_emotions", [])
        ),
        "approved_supportive_factors": clean_text_list(
            reflection.get("approved_supportive_factors", [])
        ),
        "approved_strain_factors": clean_text_list(
            reflection.get("approved_strain_factors", [])
        ),
        "approved_vocal_observation": clean_text(
            reflection.get("approved_vocal_observation", "")
        ),
        "approved_visible_observation": clean_text(
            reflection.get("approved_visible_observation", "")
        ),
        "confirmed_summary": clean_text(
            reflection.get("confirmed_summary", "")
        ),
        "confirmed_appointment_points": clean_text_list(
            reflection.get("confirmed_appointment_points", [])
        ),
    }


def history_personalisation_allowed(user_id):
    if not user_id:
        return False

    preferences = get_privacy_preferences(user_id)

    return bool(
        preferences
        and privacy_is_current(user_id)
        and preferences.get(
            "use_saved_history_for_personalisation",
            False,
        )
    )


def get_latest_saved_entry(user_id):
    if not user_id:
        return None

    checkins = get_checkins_for_user(user_id) or []
    reflections = get_saved_reflections_for_user(user_id) or []
    entries = []

    if checkins:
        created_at = parse_saved_datetime(
            checkins[0].get("created_at")
        )

        if created_at:
            entries.append({
                "entry_type": "checkin",
                "record": checkins[0],
                "created_at": created_at,
            })

    if reflections:
        created_at = parse_saved_datetime(
            reflections[0].get("created_at")
        )

        if created_at:
            entries.append({
                "entry_type": "reflection",
                "record": reflections[0],
                "created_at": created_at,
            })

    if not entries:
        return None

    return max(
        entries,
        key=lambda item: item["created_at"],
    )


def get_editable_saved_entry(user_id):
    latest = get_latest_saved_entry(user_id)

    if not latest:
        return None

    today = datetime.now(SINGAPORE_TIMEZONE).date()

    if latest["created_at"].date() != today:
        return None

    return latest


def get_saved_entry_for_direct_view(user_id):
    if not user_id:
        return None

    latest = get_latest_saved_entry(user_id)

    if not latest:
        return None

    try:
        latest_id = int(latest["record"].get("id"))
    except (TypeError, ValueError):
        return None

    if latest["entry_type"] == "checkin":
        current_saved_id = session.get("saved_checkin_id")
    else:
        current_saved_id = session.get("saved_reflection_id")

    try:
        current_saved_id = (
            int(current_saved_id)
            if current_saved_id is not None
            else None
        )
    except (TypeError, ValueError):
        current_saved_id = None

    if current_saved_id == latest_id:
        return latest

    try:
        if history_personalisation_allowed(user_id):
            return latest
    except Exception as error:
        print("Assistant direct-view permission error:", error)

    return None


def build_saved_content_view_message(view_type, user_id):
    saved_entry = get_saved_entry_for_direct_view(user_id)

    if not saved_entry:
        return None

    record = saved_entry["record"]
    entry_type = saved_entry["entry_type"]
    context_source = (
        "latest_saved_checkin"
        if entry_type == "checkin"
        else "latest_saved_reflection"
    )

    message = {
        "role": "assistant",
        "assistant_reply": "",
        "summary_draft": "",
        "appointment_points": [],
        "retrieved_resources": [],
        "detected_intent": view_type,
        "context_source": context_source,
        "saved_entry_type": entry_type,
        "saved_entry_id": record.get("id"),
        "is_edit_response": False,
        "is_saved_view": True,
        "saved_update_state": "",
    }

    if view_type == "summary":
        summary = clean_text(record.get("confirmed_summary", ""))

        if not summary:
            message["assistant_reply"] = (
                "There is no saved Personal Summary for this entry yet."
            )
            message["detected_intent"] = "conversation"
            message["is_saved_view"] = False
        else:
            message["summary_draft"] = summary

        return message

    points = clean_text_list(
        record.get("confirmed_appointment_points", [])
    )

    if not points:
        message["assistant_reply"] = (
            "There are no saved Appointment Discussion Points "
            "for this entry yet."
        )
        message["detected_intent"] = "conversation"
        message["is_saved_view"] = False
    else:
        message["appointment_points"] = points

    return message


def get_assistant_application_state():
    current_state = build_current_assistant_state()

    if current_state:
        return current_state

    user_id = session.get("user_id")

    if not user_id:
        return {}

    try:
        if not history_personalisation_allowed(user_id):
            return {}

        latest = get_latest_saved_entry(user_id)

        if not latest:
            return {}

        if latest["entry_type"] == "checkin":
            return build_saved_checkin_assistant_state(
                latest["record"]
            )

        return build_saved_reflection_assistant_state(
            latest["record"]
        )

    except Exception as error:
        print("Assistant historical context error:", error)
        return {}


def get_assistant_relevance_result():
    user_id = session.get("user_id")

    if not user_id:
        return {}

    try:
        if not history_personalisation_allowed(user_id):
            return {}

        all_patterns = get_all_patterns_for_user(user_id)
        return select_relevant_patterns(all_patterns)

    except Exception as error:
        print("Assistant relevance error:", error)
        return {}


def initialise_assistant_state():
    session.setdefault("assistant_messages", [])


def build_assistant_chat_history():
    history = []

    for message in session.get("assistant_messages", []) or []:
        if not isinstance(message, dict):
            continue

        role = clean_text(message.get("role", ""))

        if role == "user":
            content = clean_text(message.get("content", ""))

            if content:
                history.append({
                    "role": "user",
                    "content": content,
                })

            continue

        if role != "assistant":
            continue

        parts = []

        assistant_reply = clean_text(
            message.get("assistant_reply", "")
        )
        summary_draft = clean_text(
            message.get("summary_draft", "")
        )
        appointment_points = clean_text_list(
            message.get("appointment_points", [])
        )

        if assistant_reply:
            parts.append(assistant_reply)

        if summary_draft:
            parts.append(
                f"Summary draft: {summary_draft}"
            )

        if appointment_points:
            points = "\n".join(
                f"- {point}" for point in appointment_points
            )
            parts.append(
                f"Appointment points:\n{points}"
            )

        resource_titles = []

        for resource in message.get("retrieved_resources", []) or []:
            if isinstance(resource, dict):
                title = clean_text(resource.get("title", ""))

                if title:
                    resource_titles.append(title)

        if resource_titles:
            parts.append(
                "Resources shown: " + ", ".join(resource_titles)
            )

        content = "\n\n".join(parts).strip()

        if content:
            history.append({
                "role": "assistant",
                "content": content,
            })

    return history[-6:]


def prepare_resource_for_session(resource):
    if not isinstance(resource, dict):
        return {}

    return {
        "id": clean_text(resource.get("id", "")),
        "title": (
            clean_text(resource.get("title", "Resource"))
            or "Resource"
        ),
        "resource_type": clean_text(
            resource.get("resource_type", "")
        ),
        "source": clean_text(resource.get("source", "")),
        "content": clean_text(resource.get("content", "")),
        "url": clean_text(resource.get("url", "")),
    }


def extract_appointment_points(assistant_reply):
    text = clean_text(assistant_reply)

    if not text:
        return []

    points = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        cleaned = line

        for prefix in ("- ", "• ", "* "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        if len(cleaned) >= 3 and cleaned[0].isdigit():
            dot_position = cleaned.find(".")

            if 0 < dot_position <= 2:
                cleaned = cleaned[dot_position + 1:].strip()

        if cleaned and cleaned != line:
            points.append(cleaned)

    return points


def build_assistant_message(graph_result):
    error_message = clean_text(graph_result.get("error", ""))

    if error_message:
        return {
            "role": "assistant",
            "assistant_reply": error_message,
            "summary_draft": "",
            "appointment_points": [],
            "retrieved_resources": [],
            "detected_intent": "",
        }

    assistant_result = graph_result.get("assistant_result", {}) or {}

    detected_intent = clean_text(
        graph_result.get("detected_intent", "conversation")
    )
    assistant_reply = clean_text(
        assistant_result.get("assistant_reply", "")
    )
    summary_draft = clean_text(
        assistant_result.get("summary_draft", "")
    )
    appointment_points = clean_text_list(
        assistant_result.get("appointment_points", [])
    )

    if detected_intent == "appointment_points" and appointment_points:
        assistant_reply = ""

    if (
        detected_intent == "summary"
        and not summary_draft
        and assistant_reply
    ):
        summary_draft = assistant_reply
        assistant_reply = ""

    if (
        detected_intent == "appointment_points"
        and not appointment_points
        and assistant_reply
    ):
        extracted = extract_appointment_points(assistant_reply)

        if extracted:
            appointment_points = extracted
            assistant_reply = ""

    retrieved_resources = []

    for resource in graph_result.get("retrieved_resources", []) or []:
        prepared = prepare_resource_for_session(resource)

        if prepared:
            retrieved_resources.append(prepared)

    return {
        "role": "assistant",
        "assistant_reply": assistant_reply,
        "summary_draft": summary_draft,
        "appointment_points": appointment_points,
        "retrieved_resources": retrieved_resources,
        "detected_intent": detected_intent,
    }


def get_assistant_message(message_index):
    messages = list(session.get("assistant_messages", []) or [])

    try:
        index = int(message_index)
    except (TypeError, ValueError):
        return None

    if not 0 <= index < len(messages):
        return None

    message = messages[index]

    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
    ):
        return None

    return message


def current_dashboard_context_available():
    current_state = build_current_assistant_state()
    context_source = clean_text(
        current_state.get("context_source", "")
    )

    if context_source not in {
        "current_checkin",
        "current_reflection",
    }:
        return False

    if context_source == "current_checkin":
        return not bool(session.get("saved_checkin_id"))

    return not bool(session.get("saved_reflection_id"))


def saved_message_target_is_editable(message, user_id):
    if not user_id:
        return False

    editable = get_editable_saved_entry(user_id)

    if not editable:
        return False

    message_type = clean_text(
        message.get("saved_entry_type", "")
    )

    try:
        message_id = int(message.get("saved_entry_id"))
        record_id = int(editable["record"].get("id"))
    except (TypeError, ValueError):
        return False

    return (
        message_type == editable["entry_type"]
        and message_id == record_id
    )


def mark_dashboard_draft_for_review():
    session.update({
        "summary_draft_generated": True,
        "summary_review_saved": False,
        "summary_edit_mode": True,
        "summary_generation_error": "",
        "summary_source_signature": None,
    })
    session.modified = True


def sync_saved_content_to_session(
    summary=None,
    appointment_points=None,
):
    if summary is not None:
        session["personal_summary"] = clean_text(summary)

    if appointment_points is not None:
        session["appointment_points"] = clean_text_list(
            appointment_points
        )

    session["summary_review_saved"] = True
    session["summary_confirmed"] = True
    session["dashboard_summary_confirmed"] = True
    session["summary_edit_mode"] = False
    session.modified = True


def _store_saved_undo(key, editable, value, message_index):
    try:
        message_index = int(message_index)
    except (TypeError, ValueError):
        message_index = None

    session[key] = {
        "entry_type": editable["entry_type"],
        "entry_id": editable["record"].get("id"),
        "message_index": message_index,
        "value": value,
    }
    session.modified = True


def _undo_matches_editable(undo_state, editable):
    if not isinstance(undo_state, dict) or not editable:
        return False

    try:
        undo_id = int(undo_state.get("entry_id"))
        editable_id = int(editable["record"].get("id"))
    except (TypeError, ValueError):
        return False

    return (
        clean_text(undo_state.get("entry_type", ""))
        == editable["entry_type"]
        and undo_id == editable_id
    )


def _undo_matches_message(undo_state, message_index):
    if not isinstance(undo_state, dict):
        return False

    try:
        return int(undo_state.get("message_index")) == int(message_index)
    except (TypeError, ValueError):
        return False


def _set_message_update_state(message_index, intent, state):
    messages = list(session.get("assistant_messages", []) or [])

    try:
        index = int(message_index)
    except (TypeError, ValueError):
        return False

    if not 0 <= index < len(messages):
        return False

    message = messages[index]

    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or clean_text(message.get("detected_intent", "")) != intent
    ):
        return False

    if state == "applied":
        for item in messages:
            if (
                isinstance(item, dict)
                and item.get("role") == "assistant"
                and clean_text(item.get("detected_intent", "")) == intent
                and item.get("saved_update_state") == "applied"
            ):
                item["saved_update_state"] = ""

    message["saved_update_state"] = state
    messages[index] = message
    session["assistant_messages"] = messages
    session.modified = True
    return True


def update_saved_summary(message, user_id, summary_draft, message_index):
    if not saved_message_target_is_editable(message, user_id):
        return False

    editable = get_editable_saved_entry(user_id)
    previous_summary = clean_text(
        editable["record"].get("confirmed_summary", "")
    )

    if editable["entry_type"] == "checkin":
        update_checkin_summary_content(
            checkin_id=editable["record"]["id"],
            user_id=user_id,
            confirmed_summary=summary_draft,
        )
    else:
        update_reflection_summary_content(
            reflection_id=editable["record"]["id"],
            user_id=user_id,
            confirmed_summary=summary_draft,
        )

    _store_saved_undo(
        SUMMARY_UNDO_KEY,
        editable,
        previous_summary,
        message_index,
    )
    sync_saved_content_to_session(summary=summary_draft)
    return True

def update_saved_appointment_points(
    message,
    user_id,
    appointment_points,
    message_index,
):
    if not saved_message_target_is_editable(message, user_id):
        return False

    editable = get_editable_saved_entry(user_id)
    previous_points = clean_text_list(
        editable["record"].get("confirmed_appointment_points", [])
    )

    if editable["entry_type"] == "checkin":
        update_checkin_summary_content(
            checkin_id=editable["record"]["id"],
            user_id=user_id,
            confirmed_appointment_points=appointment_points,
        )
    else:
        update_reflection_summary_content(
            reflection_id=editable["record"]["id"],
            user_id=user_id,
            confirmed_appointment_points=appointment_points,
        )

    _store_saved_undo(
        APPOINTMENT_UNDO_KEY,
        editable,
        previous_points,
        message_index,
    )
    sync_saved_content_to_session(
        appointment_points=appointment_points
    )
    return True

@assistant_blueprint.route("/")
@login_required
def assistant_page():
    initialise_assistant_state()

    application_state = get_assistant_application_state()
    context_source = clean_text(
        application_state.get("context_source", "")
    )
    context_message = CONTEXT_MESSAGES.get(
        context_source,
        NO_CONTEXT_MESSAGE,
    )

    user_id = session.get("user_id")

    try:
        bookmarked_resource_ids = get_bookmarked_resource_ids(
            user_id
        )
    except Exception as error:
        print("Assistant bookmark lookup error:", error)
        bookmarked_resource_ids = set()

    bookmarked_resource_ids = {
        str(resource_id)
        for resource_id in bookmarked_resource_ids
    }

    editable_saved_entry = None

    try:
        editable_saved_entry = get_editable_saved_entry(user_id)
    except Exception as error:
        print("Assistant editable-entry error:", error)

    return render_template(
        "assistant.html",
        messages=session.get("assistant_messages", []) or [],
        context_source=context_source,
        context_message=context_message,
        historical_context=(
            context_source in {
                "latest_saved_checkin",
                "latest_saved_reflection",
            }
        ),
        current_dashboard_editing_allowed=(
            current_dashboard_context_available()
        ),
        saved_editing_allowed=bool(editable_saved_entry),
        editable_saved_entry_type=(
            editable_saved_entry["entry_type"]
            if editable_saved_entry
            else ""
        ),
        editable_saved_entry_id=(
            editable_saved_entry["record"].get("id")
            if editable_saved_entry
            else None
        ),
        bookmarked_resource_ids=bookmarked_resource_ids,
        bookmark_saving_allowed=can_save_bookmarks(user_id),
    )


@assistant_blueprint.route("/message", methods=["POST"])
@login_required
def send_message():
    initialise_assistant_state()

    user_message = clean_text(
        request.form.get("message", "")
    )

    if not user_message:
        flash("Please enter a message.", "warning")
        return assistant_redirect()

    if len(user_message) > 3000:
        flash(
            "Please keep the message under 3,000 characters.",
            "warning",
        )
        return assistant_redirect()

    chat_history = build_assistant_chat_history()

    messages = list(
        session.get("assistant_messages", []) or []
    )
    messages.append({
        "role": "user",
        "content": user_message,
    })

    session["assistant_messages"] = messages
    session.modified = True

    view_type = detect_saved_content_view_request(user_message)

    if view_type:
        direct_message = build_saved_content_view_message(
            view_type=view_type,
            user_id=session.get("user_id"),
        )

        if direct_message:
            messages = list(
                session.get("assistant_messages", []) or []
            )
            messages.append(direct_message)

            session["assistant_messages"] = messages[-20:]
            session.modified = True
            return assistant_redirect()

    application_state = get_assistant_application_state()
    relevance_result = get_assistant_relevance_result()

    try:
        graph_result = assistant_graph.invoke({
            "user_message": user_message,
            "application_state": application_state,
            "relevance_result": relevance_result,
            "assistant_chat_history": chat_history,
            "user_id": session.get("user_id"),
        })
    except Exception as error:
        print("Flask Assistant error:", error)
        graph_result = {
            "error": "The Assistant could not complete this request."
        }

    assistant_message = build_assistant_message(graph_result)

    assistant_message["context_source"] = clean_text(
        application_state.get("context_source", "")
    )
    assistant_message["saved_entry_type"] = clean_text(
        application_state.get("saved_entry_type", "")
    )
    assistant_message["saved_entry_id"] = application_state.get(
        "saved_entry_id"
    )

    assistant_message["is_edit_response"] = (
        is_edit_request(user_message)
        and assistant_message.get("detected_intent")
        in {"summary", "appointment_points"}
    )
    assistant_message["saved_update_state"] = ""
    assistant_message["is_saved_view"] = False

    messages = list(
        session.get("assistant_messages", []) or []
    )
    messages.append(assistant_message)

    session["assistant_messages"] = messages[-20:]
    session.modified = True

    return assistant_redirect()


@assistant_blueprint.route("/use-summary", methods=["POST"])
@login_required
def use_summary_on_dashboard():
    message_index = request.form.get("message_index")
    message = get_assistant_message(message_index)

    if not message:
        flash(
            "That Assistant response is no longer available.",
            "warning",
        )
        return assistant_redirect()

    summary_draft = clean_text(
        message.get("summary_draft", "")
    )

    if not summary_draft:
        flash(
            "This Assistant response does not contain "
            "a personal summary draft.",
            "warning",
        )
        return assistant_redirect()

    user_id = session.get("user_id")

    try:
        if update_saved_summary(
            message,
            user_id,
            summary_draft,
            message_index,
        ):
            _set_message_update_state(
                message_index,
                "summary",
                "applied",
            )
            flash(
                "Your saved Personal Summary was updated. "
                "The updated version will also appear in Personal Patterns.",
                "success",
            )
            return assistant_redirect()
    except Exception as error:
        print("Assistant saved-summary update error:", error)
        flash(
            "The saved summary could not be updated.",
            "error",
        )
        return assistant_redirect()

    if current_dashboard_context_available():
        session["personal_summary"] = summary_draft
        session.setdefault("appointment_points", [])

        mark_dashboard_draft_for_review()

        flash(
            "The Assistant summary has been added as a Dashboard draft. "
            "This check-in has not been saved yet, so review and confirm it "
            "before saving.",
            "success",
        )

        return redirect(url_for("dashboard.dashboard_page"))

    flash(
        "That saved check-in is now read-only. "
        "Only today's latest saved check-in can be amended.",
        "warning",
    )
    return assistant_redirect()


@assistant_blueprint.route(
    "/use-appointment-points",
    methods=["POST"],
)
@login_required
def use_appointment_points_on_dashboard():
    message_index = request.form.get("message_index")
    message = get_assistant_message(message_index)

    if not message:
        flash(
            "That Assistant response is no longer available.",
            "warning",
        )
        return assistant_redirect()

    appointment_points = clean_text_list(
        message.get("appointment_points", [])
    )

    if not appointment_points:
        flash(
            "This Assistant response does not contain "
            "appointment discussion points.",
            "warning",
        )
        return assistant_redirect()

    user_id = session.get("user_id")

    try:
        if update_saved_appointment_points(
            message,
            user_id,
            appointment_points,
            message_index,
        ):
            _set_message_update_state(
                message_index,
                "appointment_points",
                "applied",
            )
            flash(
                "Your saved Appointment Discussion Points were updated. "
                "The updated version will also appear in Personal Patterns.",
                "success",
            )
            return assistant_redirect()
    except Exception as error:
        print("Assistant saved-points update error:", error)
        flash(
            "The saved appointment points could not be updated.",
            "error",
        )
        return assistant_redirect()

    if current_dashboard_context_available():
        session["appointment_points"] = appointment_points
        session.setdefault("personal_summary", "")

        mark_dashboard_draft_for_review()

        flash(
            "The appointment discussion points have been added to the "
            "Dashboard as a draft. This check-in has not been saved yet, "
            "so review them before saving.",
            "success",
        )

        return redirect(url_for("dashboard.dashboard_page"))

    flash(
        "That saved check-in is now read-only. "
        "Only today's latest saved check-in can be amended.",
        "warning",
    )
    return assistant_redirect()


@assistant_blueprint.route("/undo-summary", methods=["POST"])
@login_required
def undo_saved_summary():
    message_index = request.form.get("message_index")
    user_id = session.get("user_id")
    editable = get_editable_saved_entry(user_id)
    undo_state = session.get(SUMMARY_UNDO_KEY)

    if (
        not _undo_matches_editable(undo_state, editable)
        or not _undo_matches_message(undo_state, message_index)
    ):
        flash(
            "There is no saved summary update available to undo.",
            "warning",
        )
        return assistant_redirect()

    previous_summary = clean_text(undo_state.get("value", ""))

    try:
        if editable["entry_type"] == "checkin":
            update_checkin_summary_content(
                checkin_id=editable["record"]["id"],
                user_id=user_id,
                confirmed_summary=previous_summary,
            )
        else:
            update_reflection_summary_content(
                reflection_id=editable["record"]["id"],
                user_id=user_id,
                confirmed_summary=previous_summary,
            )
    except Exception as error:
        print("Assistant summary undo error:", error)
        flash("The previous summary could not be restored.", "error")
        return assistant_redirect()

    sync_saved_content_to_session(summary=previous_summary)
    session.pop(SUMMARY_UNDO_KEY, None)
    _set_message_update_state(
        message_index,
        "summary",
        "undone",
    )
    session.modified = True

    flash(
        "Your previous Personal Summary was restored. "
        "Personal Patterns will show the restored version.",
        "success",
    )
    return assistant_redirect()


@assistant_blueprint.route(
    "/undo-appointment-points",
    methods=["POST"],
)
@login_required
def undo_saved_appointment_points():
    message_index = request.form.get("message_index")
    user_id = session.get("user_id")
    editable = get_editable_saved_entry(user_id)
    undo_state = session.get(APPOINTMENT_UNDO_KEY)

    if (
        not _undo_matches_editable(undo_state, editable)
        or not _undo_matches_message(undo_state, message_index)
    ):
        flash(
            "There is no saved appointment-point update available to undo.",
            "warning",
        )
        return assistant_redirect()

    previous_points = clean_text_list(undo_state.get("value", []))

    try:
        if editable["entry_type"] == "checkin":
            update_checkin_summary_content(
                checkin_id=editable["record"]["id"],
                user_id=user_id,
                confirmed_appointment_points=previous_points,
            )
        else:
            update_reflection_summary_content(
                reflection_id=editable["record"]["id"],
                user_id=user_id,
                confirmed_appointment_points=previous_points,
            )
    except Exception as error:
        print("Assistant appointment undo error:", error)
        flash(
            "The previous appointment points could not be restored.",
            "error",
        )
        return assistant_redirect()

    sync_saved_content_to_session(
        appointment_points=previous_points
    )
    session.pop(APPOINTMENT_UNDO_KEY, None)
    _set_message_update_state(
        message_index,
        "appointment_points",
        "undone",
    )
    session.modified = True

    flash(
        "Your previous Appointment Discussion Points were restored. "
        "Personal Patterns will show the restored version.",
        "success",
    )
    return assistant_redirect()


@assistant_blueprint.route("/clear", methods=["POST"])
@login_required
def clear_conversation():
    session["assistant_messages"] = []
    session.pop(SUMMARY_UNDO_KEY, None)
    session.pop(APPOINTMENT_UNDO_KEY, None)
    session.modified = True

    flash(
        "Assistant conversation cleared.",
        "success",
    )
    return assistant_redirect()


@assistant_blueprint.route("/save-resource", methods=["POST"])
@login_required
def save_resource():
    user_id = session.get("user_id")
    resource_id = clean_text(
        request.form.get("resource_id", "")
    )

    if not resource_id:
        flash(
            "The resource could not be identified.",
            "warning",
        )
        return assistant_redirect()

    if not can_save_bookmarks(user_id):
        flash(
            "Saving resources is turned off in your privacy settings.",
            "warning",
        )
        return redirect(url_for("privacy.privacy_settings"))

    try:
        resource = get_resource_by_id(resource_id)
    except Exception as error:
        print("Assistant resource lookup error:", error)
        resource = None

    if not resource:
        flash(
            "That resource is no longer available.",
            "warning",
        )
        return assistant_redirect()

    try:
        bookmark_resource(
            user_id=user_id,
            resource_id=resource_id,
        )
    except Exception as error:
        print("Assistant save-resource error:", error)
        flash(
            "The resource could not be saved.",
            "error",
        )
        return assistant_redirect()

    flash(
        "Resource saved to your account.",
        "success",
    )
    return assistant_redirect()


@assistant_blueprint.route("/remove-resource", methods=["POST"])
@login_required
def remove_saved_resource():
    user_id = session.get("user_id")
    resource_id = clean_text(
        request.form.get("resource_id", "")
    )

    if not resource_id:
        flash(
            "The resource could not be identified.",
            "warning",
        )
        return assistant_redirect()

    try:
        remove_resource_bookmark(
            user_id=user_id,
            resource_id=resource_id,
        )
    except Exception as error:
        print("Assistant remove-resource error:", error)
        flash(
            "The saved resource could not be removed.",
            "error",
        )
        return assistant_redirect()

    flash(
        "Resource removed from your saved resources.",
        "success",
    )
    return assistant_redirect()