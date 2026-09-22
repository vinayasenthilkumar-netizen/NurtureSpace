
# datetime is used to enforce the Monday weekly check-in rule
from datetime import datetime
from zoneinfo import ZoneInfo

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

# question text, answer labels and user-facing wellbeing status constants
from core.constants import (
    BABY_CARE_LABELS,
    BABY_CARE_QUESTION,
    FOOD_LABELS,
    FOOD_QUESTION,
    MEDICATION_LABELS,
    MEDICATION_QUESTION,
    MOOD_LABELS,
    MOOD_QUESTION,
    OTHER_RESPONSIBILITIES_LABELS,
    OTHER_RESPONSIBILITIES_QUESTION,
    PERSONAL_CARE_LABELS,
    PERSONAL_CARE_QUESTION,
    PHYSICAL_RECOVERY_LABELS,
    PHYSICAL_RECOVERY_QUESTION,
    SIGNIFICANT_STRAIN_STATUS,
    SLEEP_LABELS,
    SLEEP_QUESTION,
    SOME_STRAIN_STATUS,
    STEADY_STATUS,
    STRESS_LABELS,
    STRESS_QUESTION,
    SUPPORT_LABELS,
    SUPPORT_QUESTION,
)

# shared account check used before opening questionnaire routes
from flask_app.routes import has_account_session as require_account

# parent workflow graph and the structured-question stage
from graphs.guided_workflow_graph import (
    GUIDED_STAGE_STRUCTURED,
    guided_workflow_graph,
)


# all structured-question routes are grouped under /questions
questions_blueprint = Blueprint(
    "questions",
    __name__,
    url_prefix="/questions",
)


# Monday checks use Singapore local time rather than server-local time
SINGAPORE_TIMEZONE = ZoneInfo("Asia/Singapore")


# these four responses contribute to Today's Check-In Status and Question Score
CORE_QUESTIONS = [
    {
        "key": "sleep",
        "question": SLEEP_QUESTION,
        "labels": SLEEP_LABELS,
    },
    {
        "key": "mood",
        "question": MOOD_QUESTION,
        "labels": MOOD_LABELS,
    },
    {
        "key": "stress",
        "question": STRESS_QUESTION,
        "labels": STRESS_LABELS,
    },
    {
        "key": "support",
        "question": SUPPORT_QUESTION,
        "labels": SUPPORT_LABELS,
    },
]


# these six responses provide interpretation context only and do not affect the score
CONTEXT_QUESTIONS = [
    {
        "key": "food",
        "question": FOOD_QUESTION,
        "labels": FOOD_LABELS,
    },
    {
        "key": "medication",
        "question": MEDICATION_QUESTION,
        "labels": MEDICATION_LABELS,
    },
    {
        "key": "physical_recovery",
        "question": PHYSICAL_RECOVERY_QUESTION,
        "labels": PHYSICAL_RECOVERY_LABELS,
    },
    {
        "key": "baby_care",
        "question": BABY_CARE_QUESTION,
        "labels": BABY_CARE_LABELS,
    },
    {
        "key": "other_responsibilities",
        "question": OTHER_RESPONSIBILITIES_QUESTION,
        "labels": OTHER_RESPONSIBILITIES_LABELS,
    },
    {
        "key": "personal_care",
        "question": PERSONAL_CARE_QUESTION,
        "labels": PERSONAL_CARE_LABELS,
    },
]


# map wellbeing status labels to the CSS classes used by the result page
STATUS_CLASSES = {
    STEADY_STATUS: "steady",
    SOME_STRAIN_STATUS: "some-strain",
    SIGNIFICANT_STRAIN_STATUS: "significant-strain",
}


def login_redirect():
    return redirect(url_for("auth.login"))


def core_questions_redirect():
    return redirect(url_for("questions.core_questions"))


def render_questions(*, part, questions=None, existing_answers=None, **extra):
    # the same template is reused for both questionnaire sections and the result page
    return render_template(
        "questions.html",
        part=part,
        questions=questions,
        existing_answers=existing_answers,
        checkin_type=get_checkin_type(),
        **extra,
    )


def parse_answer(form_value, valid_labels):

    if form_value is None:
        return None

    try:
        answer = int(form_value)
    except (TypeError, ValueError):
        return None

    # reject values that are outside the choices defined for this question
    return answer if answer in valid_labels else None


def parse_question_answers(question_definitions):
    answers = {}
    missing = []

    for question in question_definitions:
        key = question["key"]

        # each question validates against its own configured answer labels
        answer = parse_answer(
            request.form.get(key),
            question["labels"],
        )

        if answer is None:
            missing.append(question["question"])
        else:
            answers[key] = answer

    return answers, missing


def get_checkin_type():
    # weekday() returns 0 for Monday
    is_monday = datetime.now(SINGAPORE_TIMEZONE).weekday() == 0

    return {
        "is_monday": is_monday,
        "label": "Weekly check-in" if is_monday else "Optional check-in",
    }


def get_status_class(status):
    return STATUS_CLASSES.get(status, "neutral")


def clear_summary_results():

    # changing questionnaire evidence means an older generated summary is now stale
    session.update({
        "summary_draft_generated": False,
        "summary_review_saved": False,
        "summary_edit_mode": False,
        "summary_source_signature": None,
        "summary_generation_error": "",
        "personal_summary": "",
        "appointment_points": [],
    })


def clear_previous_structured_results():
    # these values must be recalculated after the core answers change
    keys = (
        "checkin_status",
        "status_breakdown",
        "context_result",
        "checkin_context_answers",
        "question_score",
        "combined_wellbeing_score",
        "combined_wellbeing_indicator",
        "questions_completed",
        "questions_skipped_today",
    )

    for key in keys:
        session.pop(key, None)

    # any Dashboard draft based on the old answers must also be cleared
    clear_summary_results()


@questions_blueprint.route("/", methods=["GET", "POST"])
def core_questions():
    # structured check-ins are available only to signed-in users
    if not require_account():
        flash("Please log in to continue.", "error")
        return login_redirect()

    # keep previous answers available when returning to this page
    existing_answers = session.get("checkin_core_answers", {}) or {}

    # GET only displays the current four core questions
    if request.method == "GET":
        return render_questions(
            part="core",
            questions=CORE_QUESTIONS,
            existing_answers=existing_answers,
        )

    # POST validates all four submitted core answers
    answers, missing = parse_question_answers(CORE_QUESTIONS)

    if missing:
        flash(
            "Please answer all four questions before continuing.",
            "error",
        )

        # keep any valid answers the user already selected
        return render_questions(
            part="core",
            questions=CORE_QUESTIONS,
            existing_answers=answers,
        )

    # core responses are stored temporarily until the context section is completed
    session["checkin_core_answers"] = answers

    # older status, context and summary results no longer match these answers
    clear_previous_structured_results()

    # continue to the six context questions
    return redirect(url_for("questions.context_questions"))


@questions_blueprint.route("/context", methods=["GET", "POST"])
def context_questions():
    if not require_account():
        return login_redirect()

    # context questions cannot be completed before all four core answers exist
    core_answers = session.get("checkin_core_answers") or {}

    if len(core_answers) != 4:
        flash(
            "Please complete the first four questions first.",
            "error",
        )
        return core_questions_redirect()

    # preserve previously entered context responses when revisiting the page
    existing_answers = session.get("checkin_context_answers", {}) or {}

    if request.method == "GET":
        return render_questions(
            part="context",
            questions=CONTEXT_QUESTIONS,
            existing_answers=existing_answers,
        )

    # validate all six context responses
    context_answers, missing = parse_question_answers(CONTEXT_QUESTIONS)

    if missing:
        flash(
            "Please answer all six context questions before continuing.",
            "error",
        )

        return render_questions(
            part="context",
            questions=CONTEXT_QUESTIONS,
            existing_answers=context_answers,
        )

    try:
        # the structured graph calculates questionnaire status and deterministic scores
        graph_result = guided_workflow_graph.invoke({
            "workflow_stage": GUIDED_STAGE_STRUCTURED,
            "core_answers": core_answers,
            "context_answers": context_answers,

            # if Reflection was completed first, its score allows the graph
            # to derive the internal combined value and overall indicator
            "reflection_score": session.get("reflection_score"),
        })

    except Exception as error:
        print("Structured check-in graph error:", error)

        flash(
            "The check-in could not be processed. "
            "Please try again.",
            "error",
        )

        # keep the submitted context answers so the user does not have to re-enter them
        return render_questions(
            part="context",
            questions=CONTEXT_QUESTIONS,
            existing_answers=context_answers,
        )

    # graph-level validation errors are returned without saving incomplete results
    graph_error = str(graph_result.get("error", "") or "").strip()

    if graph_error:
        flash(graph_error, "error")

        return render_questions(
            part="context",
            questions=CONTEXT_QUESTIONS,
            existing_answers=context_answers,
        )

    # save the completed structured check-in state in the current session
    session.update({
        "checkin_context_answers": context_answers,

        # user-facing questionnaire status derived from the four core questions
        "checkin_status": graph_result.get("checkin_status"),

        # breakdown retains details used to explain or inspect the status result
        "status_breakdown": graph_result.get("status_breakdown") or {},

        # interpretation of the six context responses
        "context_result": graph_result.get("context_result") or {},

        # numerical score based only on the four core questions
        "question_score": graph_result.get("question_score"),

        # internal numerical combined value used for testing and indicator derivation
        "combined_wellbeing_score": graph_result.get(
            "combined_wellbeing_score"
        ),

        # user-facing combined category when both question and reflection scores exist
        "combined_wellbeing_indicator": graph_result.get(
            "combined_wellbeing_indicator"
        ),

        "questions_completed": True,
        "questions_skipped_today": False,
    })

    # a previously generated Dashboard summary may no longer match this check-in
    clear_summary_results()

    return redirect(url_for("questions.result"))


@questions_blueprint.route("/result")
def result():
    if not require_account():
        return login_redirect()

    # result page requires a completed questionnaire status in session
    status = str(
        session.get("checkin_status", "") or ""
    ).strip()

    if not status:
        return core_questions_redirect()

    # only the categorical status is presented to the user on this page
    return render_questions(
        part="result",
        status=status,
        status_class=get_status_class(status),
    )


@questions_blueprint.route("/skip", methods=["POST"])
def skip_today():
    if not require_account():
        return login_redirect()

    # Monday is the required weekly questionnaire day, so it cannot be skipped
    if datetime.now(SINGAPORE_TIMEZONE).weekday() == 0:
        flash(
            "Today is your weekly check-in day. "
            "You can return to it later.",
            "info",
        )
        return redirect(url_for("home.home"))

    # on optional days, remove any questionnaire state for the current check-in
    keys = (
        "checkin_core_answers",
        "checkin_context_answers",
        "checkin_status",
        "status_breakdown",
        "context_result",
        "question_score",
        "combined_wellbeing_score",
        "combined_wellbeing_indicator",
        "questions_completed",
        "saved_checkin_id",
    )

    for key in keys:
        session.pop(key, None)

    # remember that the questionnaire was intentionally skipped today
    session["questions_skipped_today"] = True

    # any summary based on structured-question data is now invalid
    clear_summary_results()

    # Reflection remains available separately after skipping Questions
    return redirect(url_for("home.home"))