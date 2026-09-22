
# used to create a stable signature of the evidence behind a generated summary
import hashlib
import json

# flask helpers for routes, forms, templates, sessions and redirects
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# question labels, reflection modes and wellbeing status constants
from core.constants import (
    BABY_CARE_LABELS,
    FOOD_LABELS,
    MEDICATION_LABELS,
    MOOD_LABELS,
    OTHER_RESPONSIBILITIES_LABELS,
    PERSONAL_CARE_LABELS,
    PHYSICAL_RECOVERY_LABELS,
    QUESTION_SET_VERSION,
    SIGNIFICANT_STRAIN_STATUS,
    SLEEP_LABELS,
    SOME_STRAIN_STATUS,
    STEADY_STATUS,
    STRESS_LABELS,
    SUPPORT_LABELS,
    TEXT_REFLECTION,
    VIDEO_REFLECTION,
    VOICE_REFLECTION,
)

# privacy permission controlling whether approved check-ins may be persisted
from database.privacy import can_save_approved_checkins

# shared authenticated-session check
from flask_app.routes import has_account_session as require_account

# Dashboard generation is handled through the parent guided workflow
from graphs.guided_workflow_graph import (
    GUIDED_STAGE_DASHBOARD,
    guided_workflow_graph,
)

# persistence services create new records or update existing approved records
from services.checkin_saving import (
    save_or_update_approved_checkin,
    save_or_update_approved_reflection,
)


# all Dashboard routes are grouped under /dashboard
dashboard_blueprint = Blueprint(
    "dashboard",
    __name__,
    url_prefix="/dashboard",
)


# four core responses contribute to the structured questionnaire result
CORE_RESPONSE_FIELDS = (
    ("sleep", "Sleep quality", SLEEP_LABELS),
    ("mood", "Current mood", MOOD_LABELS),
    ("stress", "Stress level", STRESS_LABELS),
    ("support", "Perceived support", SUPPORT_LABELS),
)


# six context responses are shown for interpretation but do not affect Question Score
CONTEXT_RESPONSE_FIELDS = (
    ("food", "Food needs", FOOD_LABELS),
    ("medication", "Medicines or supplements", MEDICATION_LABELS),
    ("physical_recovery", "Physical recovery", PHYSICAL_RECOVERY_LABELS),
    ("baby_care", "Caring for the baby", BABY_CARE_LABELS),
    (
        "other_responsibilities",
        "Other responsibilities",
        OTHER_RESPONSIBILITIES_LABELS,
    ),
    ("personal_care", "Personal care", PERSONAL_CARE_LABELS),
)


# simplified mappings are useful when preparing labelled state for the Dashboard graph
CORE_RESPONSE_LABELS = {
    field: labels
    for field, _, labels in CORE_RESPONSE_FIELDS
}

CONTEXT_RESPONSE_LABELS = {
    field: labels
    for field, _, labels in CONTEXT_RESPONSE_FIELDS
}


# status labels map to the CSS classes used by the Dashboard
STATUS_CLASSES = {
    STEADY_STATUS: "steady",
    SOME_STRAIN_STATUS: "some-strain",
    SIGNIFICANT_STRAIN_STATUS: "significant-strain",
}


# defaults are added only when the session does not already contain the key
SUMMARY_DEFAULTS = {
    "summary_draft_generated": False,
    "summary_review_saved": False,
    "summary_edit_mode": False,
    "summary_source_signature": None,
    "summary_generation_error": "",
    "personal_summary": "",
    "appointment_points": [],
    "saved_checkin_id": None,
    "saved_reflection_id": None,
}


# reset values clear temporary generated content without removing saved record ids
SUMMARY_RESET_VALUES = {
    "summary_draft_generated": False,
    "summary_review_saved": False,
    "summary_edit_mode": False,
    "summary_source_signature": None,
    "summary_generation_error": "",
    "personal_summary": "",
    "appointment_points": [],
}


def dashboard_redirect():
    return redirect(url_for("dashboard.dashboard_page"))


def login_redirect():
    return redirect(url_for("auth.login"))


def get_response_text(value, labels):
    if value is None:
        return "Not answered"
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "Unknown response"
    # return a safe fallback if the stored value is outside the expected label set
    return labels.get(value, "Unknown response")


def get_status_class(status):
    return STATUS_CLASSES.get(status, "neutral")


def parse_appointment_points(text):
    points = []

    for line in str(text or "").splitlines():
        # allow users to type plain lines or simple dash/bullet prefixes
        point = line.strip().lstrip("-• ").strip()

        if point and point not in points:
            points.append(point)

    return points


def build_response_display(answers, fields):
    return [
        {
            "label": label,
            "value": get_response_text(answers.get(field), labels),
        }
        for field, label, labels in fields
    ]


def build_labelled_responses(answers, label_sets):
    return {
        field: get_response_text(answers.get(field), labels)
        for field, labels in label_sets.items()
    }


def get_confirmed_reflection():
    mode = session.get("reflection_mode")

    # text reflections use the original submitted text
    if mode == TEXT_REFLECTION:
        return str(session.get("typed_reflection", "") or "").strip()

    # voice and video use the transcript after user review
    if mode in (VOICE_REFLECTION, VIDEO_REFLECTION):
        return str(session.get("confirmed_transcript", "") or "").strip()

    return ""


def get_core_display():
    answers = session.get("checkin_core_answers") or {}
    return build_response_display(answers, CORE_RESPONSE_FIELDS)


def get_context_display():
    answers = session.get("checkin_context_answers") or {}
    return build_response_display(answers, CONTEXT_RESPONSE_FIELDS)


def structured_checkin_complete():
    core = session.get("checkin_core_answers") or {}
    context = session.get("checkin_context_answers") or {}

    # a full structured check-in requires 4 core answers, 6 context answers and a status
    return (
        len(core) == 4
        and len(context) == 6
        and bool(session.get("checkin_status"))
    )


def get_approved_analysis_state():
    """Return only reflection observations approved during Review."""

    # suggestions ignored by the user are deliberately excluded here
    return {
        "approved_text_emotions": list(
            session.get("approved_text_emotions", []) or []
        ),
        "approved_supportive_factors": list(
            session.get("approved_supportive_factors", []) or []
        ),
        "approved_strain_factors": list(
            session.get("approved_strain_factors", []) or []
        ),
        "approved_vocal_observation": str(
            session.get("approved_audio_observation", "") or ""
        ).strip(),
        "approved_visible_observation": str(
            session.get("approved_visible_expression", "") or ""
        ).strip(),
    }


def get_wellbeing_result():
    combined_value = session.get("combined_wellbeing_score")
    combined_indicator = session.get("combined_wellbeing_indicator")

    # combined result takes priority when both questionnaire and reflection exist
    if combined_value is not None and combined_indicator:
        return {
            "score": None,
            "indicator": combined_indicator,
            "source": "combined",
        }

    # reflection-only check-ins retain their reflection score and category
    return {
        "score": session.get("reflection_score"),
        "indicator": session.get("reflection_indicator"),
        "source": "reflection",
    }


def build_dashboard_application_state():
    core_answers = session.get("checkin_core_answers") or {}
    context_answers = session.get("checkin_context_answers") or {}
    context_result = session.get("context_result") or {}

    state = {
        # tells generation whether this is a full check-in or reflection-only flow
        "context_source": (
            "current_checkin"
            if structured_checkin_complete()
            else "current_reflection"
        ),

        # deterministic wellbeing results are supplied as context, not generated by the LLM
        "status": session.get("checkin_status"),
        "question_score": session.get("question_score"),
        "reflection_score": session.get("reflection_score"),
        "reflection_indicator": session.get("reflection_indicator"),

        # internal numerical value retained for deterministic processing and evaluation
        "combined_wellbeing_score": session.get("combined_wellbeing_score"),

        # categorical combined result that may be shown to the user
        "combined_wellbeing_indicator": session.get(
            "combined_wellbeing_indicator"
        ),

        # generation receives readable labels rather than unexplained numeric answers
        "core_responses": build_labelled_responses(
            core_answers,
            CORE_RESPONSE_LABELS,
        ),
        "context_responses": build_labelled_responses(
            context_answers,
            CONTEXT_RESPONSE_LABELS,
        ),

        # only the written reflection or user-confirmed transcript is used here
        "confirmed_reflection": get_confirmed_reflection(),

        # structured context interpretation may contribute to summary wording
        "context_summary_points": list(
            context_result.get("summary_points", []) or []
        ),
        "context_appointment_points": list(
            context_result.get("appointment_suggestions", []) or []
        ),
    }

    # add only the observations explicitly approved during Review
    state.update(get_approved_analysis_state())

    return state


def build_summary_source_signature():
    # sort keys so identical application state always creates the same JSON text
    source = json.dumps(
        build_dashboard_application_state(),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    # a hash allows stale generated drafts to be detected without storing a duplicate state
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def initialise_summary_state():
    for key, value in SUMMARY_DEFAULTS.items():
        session.setdefault(key, value)


def reset_summary():
    session.update(SUMMARY_RESET_VALUES)


def clear_outdated_summary():
    saved_signature = session.get("summary_source_signature")

    # changing approved evidence, questionnaire answers or reflection text means the previous generated summary should no longer be reused
    if (
        saved_signature
        and saved_signature != build_summary_source_signature()
    ):
        reset_summary()


def mark_generated_draft(personal_summary, appointment_points):
    session.update({
        "personal_summary": personal_summary,
        "appointment_points": appointment_points,
        "summary_draft_generated": True,

        # generation alone does not count as user confirmation
        "summary_review_saved": False,

        # generated content opens in editable mode before confirmation
        "summary_edit_mode": True,
        "summary_generation_error": "",

        # remember exactly which application evidence produced this draft
        "summary_source_signature": build_summary_source_signature(),
    })


def generate_dashboard_drafts():
    # build bounded context only from current questionnaire, reflection
    # and observations approved by the user
    application_state = build_dashboard_application_state()

    try:
        graph_result = guided_workflow_graph.invoke({
            "workflow_stage": GUIDED_STAGE_DASHBOARD,
            "dashboard_application_state": application_state,
        })

    except Exception as error:
        print("Dashboard generation error:", error)

        session["summary_generation_error"] = (
            "The summary and appointment drafts could not be generated."
        )
        return False

    # do not accept partial or incomplete Dashboard generation
    if not graph_result.get("dashboard_generation_complete", False):
        session["summary_generation_error"] = str(
            graph_result.get(
                "error",
                "The summary and appointment drafts could not be generated.",
            )
            or "The summary and appointment drafts could not be generated."
        )
        return False

    # extract the generated text from the completed graph result
    personal_summary = str(
        graph_result.get("dashboard_summary_draft", "") or ""
    ).strip()

    appointment_points = list(
        graph_result.get("dashboard_appointment_points", []) or []
    )

    # a Personal Summary is required before the generated result is usable
    if not personal_summary:
        session["summary_generation_error"] = (
            "The personal summary could not be generated."
        )
        return False

    # keep the generated content temporary until the user confirms and saves it
    mark_generated_draft(personal_summary, appointment_points)
    return True


def get_confirmed_summary_state():
    return {
        "confirmed_summary": str(
            session.get("personal_summary", "") or ""
        ).strip(),
        "confirmed_appointment_points": list(
            session.get("appointment_points", []) or []
        ),
    }


def build_approved_checkin_data():
    data = {
        # retain questionnaire version so saved answers remain interpretable
        "question_set_version": QUESTION_SET_VERSION,

        "checkin_status": session.get("checkin_status"),
        "reflection_mode": session.get("reflection_mode"),

        # copy dictionaries so persistence receives plain approved data
        "core_answers": dict(
            session.get("checkin_core_answers", {}) or {}
        ),
        "context_answers": dict(
            session.get("checkin_context_answers", {}) or {}
        ),

        # deterministic question and reflection results
        "question_score": session.get("question_score"),
        "reflection_score": session.get("reflection_score"),
        "reflection_indicator": session.get("reflection_indicator"),

        # retained internally for reproducibility, evaluation and later calculations
        # but not shown as the final user-facing wellbeing result
        "combined_wellbeing_score": session.get(
            "combined_wellbeing_score"
        ),

        # user-facing category derived from the internal combined value
        "combined_wellbeing_indicator": session.get(
            "combined_wellbeing_indicator"
        ),
    }

    # save only approved observations and confirmed summary content
    data.update(get_approved_analysis_state())
    data.update(get_confirmed_summary_state())

    return data


def build_approved_reflection_data():
    # reflection-only entries do not include questionnaire answers or Question Score
    data = {
        "reflection_mode": session.get("reflection_mode"),
        "reflection_score": session.get("reflection_score"),
        "reflection_indicator": session.get("reflection_indicator"),
    }

    # approved observations and confirmed summary content are still included
    data.update(get_approved_analysis_state())
    data.update(get_confirmed_summary_state())

    return data


@dashboard_blueprint.route("/")
def dashboard_page():
    if not require_account():
        flash("Please log in to continue.", "error")
        return login_redirect()

    # Dashboard is available only after the user has completed Review choices
    if not session.get("review_complete", False):
        flash("Please complete the reflection review first.", "error")
        return redirect(url_for("review.review_page"))

    # make sure expected summary keys exist and remove stale generated content
    initialise_summary_state()
    clear_outdated_summary()

    try:
        # privacy preference determines whether the confirmed entry can be persisted
        saving_allowed = can_save_approved_checkins(
            session.get("user_id")
        )
    except Exception:
        # fail safely by disabling persistence if privacy permission cannot be checked
        saving_allowed = False

    reflection_mode = session.get("reflection_mode")
    appointment_points = session.get("appointment_points", []) or []
    checkin_status = session.get("checkin_status")

    # choose combined or reflection-only presentation without exposing combined number
    wellbeing_result = get_wellbeing_result()

    return render_template(
        "dashboard.html",

        # structured questionnaire status and completion state
        checkin_status=checkin_status,
        status_class=get_status_class(checkin_status),
        structured_complete=structured_checkin_complete(),

        # individual deterministic component results
        question_score=session.get("question_score"),
        reflection_score=session.get("reflection_score"),
        reflection_indicator=session.get("reflection_indicator"),

        # expose only the categorical combined result to the UI
        overall_checkin_indicator=session.get(
            "combined_wellbeing_indicator"
        ),

        # combined results deliberately hide their internal numerical value
        # while reflection-only results may still display Reflection Score
        wellbeing_score=wellbeing_result["score"],
        wellbeing_indicator=wellbeing_result["indicator"],
        wellbeing_source=wellbeing_result["source"],

        # labelled questionnaire responses shown in the expandable section
        core_responses=get_core_display(),
        context_responses=get_context_display(),

        # reflection details and mode constants required by the template
        reflection_mode=reflection_mode,
        text_reflection=TEXT_REFLECTION,
        voice_reflection=VOICE_REFLECTION,
        video_reflection=VIDEO_REFLECTION,
        confirmed_reflection=get_confirmed_reflection(),

        # only observations kept by the user during Review are displayed
        approved_emotions=session.get(
            "approved_text_emotions", []
        ) or [],
        approved_supportive=session.get(
            "approved_supportive_factors", []
        ) or [],
        approved_strain=session.get(
            "approved_strain_factors", []
        ) or [],
        approved_audio=session.get("approved_audio_observation"),
        approved_expression=session.get(
            "approved_visible_expression"
        ),

        # current generated-summary state
        summary_generated=bool(
            session.get("summary_draft_generated", False)
        ),
        summary_confirmed=bool(
            session.get("summary_review_saved", False)
        ),
        summary_edit_mode=bool(
            session.get("summary_edit_mode", False)
        ),
        personal_summary=session.get("personal_summary", ""),
        appointment_points=appointment_points,

        # textarea uses one appointment point per line
        appointment_points_text="\n".join(appointment_points),

        generation_error=session.get(
            "summary_generation_error", ""
        ),

        # persistence controls depend on privacy and whether an entry already exists
        saving_allowed=saving_allowed,
        saved_checkin_id=session.get("saved_checkin_id"),
        saved_reflection_id=session.get("saved_reflection_id"),
    )


@dashboard_blueprint.route("/generate", methods=["POST"])
def generate():
    if not require_account():
        return login_redirect()

    # generation should not bypass the user's Review choices
    if not session.get("review_complete", False):
        return redirect(url_for("review.review_page"))

    if generate_dashboard_drafts():
        flash("Your drafts are ready to review.", "success")
    else:
        flash(
            session.get(
                "summary_generation_error",
                "The drafts could not be generated.",
            ),
            "error",
        )

    return dashboard_redirect()


@dashboard_blueprint.route("/regenerate", methods=["POST"])
def regenerate():
    if not require_account():
        return login_redirect()

    # regeneration uses the same bounded current application state
    if generate_dashboard_drafts():
        flash("A new draft has been generated.", "success")
    else:
        flash(
            session.get(
                "summary_generation_error",
                "The drafts could not be regenerated.",
            ),
            "error",
        )

    return dashboard_redirect()


@dashboard_blueprint.route("/confirm", methods=["POST"])
def confirm():
    if not require_account():
        return login_redirect()

    # use the user's edited textarea value rather than the original generated text
    personal_summary = str(
        request.form.get("personal_summary", "") or ""
    ).strip()

    # Dashboard requires a Personal Summary before it can be confirmed
    if not personal_summary:
        flash("The personal summary cannot be empty.", "error")
        return dashboard_redirect()

    # each non-empty textarea line becomes one appointment discussion point
    appointment_points = parse_appointment_points(
        request.form.get("appointment_points", "")
    )

    session.update({
        "personal_summary": personal_summary,
        "appointment_points": appointment_points,

        # content exists and has now been explicitly confirmed by the user
        "summary_draft_generated": True,
        "summary_review_saved": True,
        "summary_edit_mode": False,
        "summary_generation_error": "",

        # signature ties this confirmed version to the current evidence
        "summary_source_signature": build_summary_source_signature(),
    })

    flash(
        "Your summary and appointment points are confirmed.",
        "success",
    )

    return dashboard_redirect()


@dashboard_blueprint.route("/edit", methods=["POST"])
def edit():
    if not require_account():
        return login_redirect()

    # this changes only the interface state, not the underlying approved evidence
    session["summary_edit_mode"] = True
    return dashboard_redirect()


@dashboard_blueprint.route("/cancel-edit", methods=["POST"])
def cancel_edit():
    session["summary_edit_mode"] = False
    return dashboard_redirect()


@dashboard_blueprint.route("/clear", methods=["POST"])
def clear():

    # approved questionnaire/reflection evidence remains unchanged
    reset_summary()

    flash(
        "The temporary summary and appointment points were cleared.",
        "success",
    )

    return dashboard_redirect()


def apply_save_result(
    result,
    result_key,
    session_id_key,
    success_message,
    error_message,
):

    if result.get("success"):
        # service returns the created or updated database record
        saved_item = result.get(result_key) or {}

        # retaining the id allows later saves to update instead of duplicating the entry
        session[session_id_key] = saved_item.get("id")

        flash(
            result.get("message", success_message),
            "success",
        )
        return

    # persistence errors are returned to the user without clearing Dashboard state
    flash(
        result.get("message", error_message),
        "error",
    )


@dashboard_blueprint.route("/save", methods=["POST"])
def save_checkin():

    if not require_account():
        return login_redirect()

    # generated drafts cannot be persisted until the user explicitly confirms them
    if not session.get("summary_review_saved", False):
        flash(
            "Please confirm the personal summary before saving.",
            "error",
        )
        return dashboard_redirect()

    user_id = session.get("user_id")

    # full structured check-ins and reflection-only entries use separate persistence paths
    if structured_checkin_complete():

        # service handles privacy checks and create-versus-update behaviour
        result = save_or_update_approved_checkin(
            user_id=user_id,
            checkin_data=build_approved_checkin_data(),
            summary_confirmed=True,
            existing_checkin_id=session.get("saved_checkin_id"),
        )

        apply_save_result(
            result,
            "checkin",
            "saved_checkin_id",
            "Your approved check-in was saved.",
            "The check-in could not be saved.",
        )

    else:
        # reflection-only saves exclude questionnaire data
        result = save_or_update_approved_reflection(
            user_id=user_id,
            reflection_data=build_approved_reflection_data(),
            summary_confirmed=True,
            existing_reflection_id=session.get("saved_reflection_id"),
        )

        apply_save_result(
            result,
            "reflection",
            "saved_reflection_id",
            "Your approved reflection was saved.",
            "The reflection could not be saved.",
        )

    # remain on the Dashboard so the saved / updated state is immediately visible
    return dashboard_redirect()