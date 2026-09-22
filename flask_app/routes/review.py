
# flask helpers for routes, forms, templates, sessions and redirects
from flask import Blueprint, flash, redirect, render_template, request, session, url_for

# reflection modes and user-facing status labels
from core.constants import (
    SIGNIFICANT_STRAIN_STATUS,
    SOME_STRAIN_STATUS,
    STEADY_STATUS,
    TEXT_REFLECTION,
    VIDEO_REFLECTION,
    VOICE_REFLECTION,
)

# shared account-session check and the parent guided workflow graph
from flask_app.routes import has_account_session as require_account
from graphs.guided_workflow_graph import GUIDED_STAGE_REVIEW, guided_workflow_graph


# all review pages are grouped under /review
review_blueprint = Blueprint("review", __name__, url_prefix="/review")


def remove_duplicates(items):
    result = []

    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)

    return result


def suggestion_values(items, key):
    values = []

    for item in items or []:
        if isinstance(item, dict):
            value = str(item.get(key, "") or "").strip()
            if value:
                values.append(value)

    return remove_duplicates(values)


def get_status_class(status):
    if status == STEADY_STATUS:
        return "steady"
    if status == SOME_STRAIN_STATUS:
        return "some-strain"
    if status == SIGNIFICANT_STRAIN_STATUS:
        return "significant-strain"

    return "neutral"


def get_mode_slug(mode):
    if mode == VOICE_REFLECTION:
        return "voice"
    if mode == VIDEO_REFLECTION:
        return "video"

    return "text"


def clear_previous_results():
    # these fields always contain lists
    list_keys = (
        "approved_text_emotions",
        "approved_supportive_factors",
        "approved_strain_factors",
        "all_text_emotion_scores",
    )

    # these values are replaced whenever a new analysis is run
    result_keys = (
        "approved_audio_observation",
        "approved_visible_expression",
        "text_component_score",
        "audio_component_score",
        "video_component_score",
        "reflection_score",
        "reflection_indicator",

        # internal numerical value used to derive the Overall Check-In Indicator
        "combined_wellbeing_score",

        # retained session key containing the user-facing combined category
        "combined_wellbeing_indicator",
    )

    for key in list_keys:
        session[key] = []

    for key in result_keys:
        session[key] = None

    # a new analysis means the previous review decisions are no longer valid
    session.update({
        "text_review_saved": False,
        "theme_review_saved": False,
        "audio_review_saved": False,
        "expression_review_saved": False,
        "review_complete": False,
    })


def reset_summary_state():
    session.update({
        "summary_draft_generated": False,
        "summary_review_saved": False,
        "summary_edit_mode": False,
        "summary_source_signature": None,
        "summary_generation_error": "",
        "personal_summary": "",
        "appointment_points": [],
    })


def get_clear_observation(key):
    result = session.get(key)

    if not isinstance(result, dict) or not result.get("clear_result", False):
        return None

    observation = str(result.get("observation", "") or "").strip()
    return observation or None


def save_review_choices_from_form(form):
    mode = session.get("reflection_mode")

    approved_emotions = []
    approved_supportive = []
    approved_strain = []

    # process the text-emotion suggestions shown on the Review page
    emotions = session.get("suggested_text_emotions", []) or []

    for index, suggestion in enumerate(emotions):
        decision = str(
            form.get(f"emotion_decision_{index}", "ignore") or "ignore"
        ).strip().lower()

        if decision == "keep" and isinstance(suggestion, dict):
            category = str(suggestion.get("category", "") or "").strip()

            if category:
                approved_emotions.append(category)

    # process supportive factors using the same explicit Keep/Ignore approach
    supportive_factors = session.get("suggested_supportive_factors", []) or []

    for index, suggestion in enumerate(supportive_factors):
        decision = str(
            form.get(f"supportive_decision_{index}", "ignore") or "ignore"
        ).strip().lower()

        if decision == "keep" and isinstance(suggestion, dict):
            name = str(suggestion.get("name", "") or "").strip()

            if name:
                approved_supportive.append(name)

    # process strain factors separately so they remain distinct from supportive factors
    strain_factors = session.get("suggested_strain_factors", []) or []

    for index, suggestion in enumerate(strain_factors):
        decision = str(
            form.get(f"strain_decision_{index}", "ignore") or "ignore"
        ).strip().lower()

        if decision == "keep" and isinstance(suggestion, dict):
            name = str(suggestion.get("name", "") or "").strip()

            if name:
                approved_strain.append(name)

    # vocal observations are available only for voice and video reflections
    approved_audio = None

    if mode in (VOICE_REFLECTION, VIDEO_REFLECTION):
        audio_suggestion = session.get("suggested_audio_observation")
        audio_decision = str(
            form.get("audio_decision", "ignore") or "ignore"
        ).strip().lower()

        if (
            audio_decision == "keep"
            and isinstance(audio_suggestion, dict)
            and audio_suggestion.get("clear_result", False)
        ):
            observation = str(
                audio_suggestion.get("observation", "") or ""
            ).strip()

            # unclear observations should never become approved evidence
            if observation and observation != "Unclear":
                approved_audio = observation

    # visible-expression observations apply only to video reflections
    approved_expression = None

    if mode == VIDEO_REFLECTION:
        expression_suggestion = session.get("suggested_visible_expression")
        expression_decision = str(
            form.get("expression_decision", "ignore") or "ignore"
        ).strip().lower()

        if (
            expression_decision == "keep"
            and isinstance(expression_suggestion, dict)
            and expression_suggestion.get("clear_result", False)
        ):
            observation = str(
                expression_suggestion.get("observation", "") or ""
            ).strip()

            if observation:
                approved_expression = observation

    # store only the observations the user explicitly approved
    session["approved_text_emotions"] = remove_duplicates(approved_emotions)
    session["approved_supportive_factors"] = remove_duplicates(approved_supportive)
    session["approved_strain_factors"] = remove_duplicates(approved_strain)
    session["approved_audio_observation"] = approved_audio
    session["approved_visible_expression"] = approved_expression

    # mark the relevant parts of the review as completed
    session["text_review_saved"] = True
    session["theme_review_saved"] = True
    session["audio_review_saved"] = mode in (VOICE_REFLECTION, VIDEO_REFLECTION)
    session["expression_review_saved"] = mode == VIDEO_REFLECTION
    session["review_complete"] = True

    # approved evidence may have changed, so any previous summary must be rebuilt
    reset_summary_state()


def get_question_score_for_graph():
    if not session.get("questions_completed", False):
        return None

    return session.get("question_score")


def build_review_graph_input(mode):
    # common state required for text, voice and video review
    graph_input = {
        "workflow_stage": GUIDED_STAGE_REVIEW,
        "reflection_mode": mode,
        "question_score": get_question_score_for_graph(),
        "suggested_audio_observation": session.get("suggested_audio_observation"),
        "suggested_visible_expression": session.get("suggested_visible_expression"),
    }

    # media-specific processing results contain the temporary model evidence
    if mode == VOICE_REFLECTION:
        graph_input["voice_processing_result"] = (
            session.get("voice_processing_result") or {}
        )

    elif mode == VIDEO_REFLECTION:
        graph_input["video_processing_result"] = (
            session.get("video_processing_result") or {}
        )

    return graph_input


def store_graph_results(result):
    # store the full model suggestions so the Review page can display them
    session["suggested_text_emotions"] = list(
        result.get("suggested_text_emotions", []) or []
    )
    session["all_text_emotion_scores"] = list(
        result.get("all_text_emotion_scores", []) or []
    )
    session["suggested_supportive_factors"] = list(
        result.get("suggested_supportive_factors", []) or []
    )
    session["suggested_strain_factors"] = list(
        result.get("suggested_strain_factors", []) or []
    )

    # scoring fields are generated deterministically by the Review graph
    result_keys = (
        "text_component_score",
        "audio_component_score",
        "video_component_score",
        "reflection_score",
        "reflection_indicator",

        # internal numerical value
        "combined_wellbeing_score",

        # user-facing Overall Check-In Indicator
        "combined_wellbeing_indicator",
    )

    for key in result_keys:
        session[key] = result.get(key)


def run_review_analysis():
    mode = session.get("reflection_mode")
    graph_input = build_review_graph_input(mode)

    # written reflections can be analysed directly
    if mode == TEXT_REFLECTION:
        analysis_text = str(session.get("typed_reflection", "") or "").strip()

        if not analysis_text:
            return False, "There is no written reflection to analyse."

        graph_input["reflection_text"] = analysis_text

    # voice and video text analysis must use the user-confirmed transcript
    elif mode in (VOICE_REFLECTION, VIDEO_REFLECTION):
        analysis_text = str(
            session.get("confirmed_transcript", "") or ""
        ).strip()

        if not analysis_text:
            return False, "Please confirm the transcript before analysing it."

        graph_input["confirmed_transcript"] = analysis_text

    else:
        return False, "A reflection is required before review."

    # send the prepared state through the Review stage of the guided workflow
    try:
        result = guided_workflow_graph.invoke(graph_input)

    except Exception as error:
        print("Review analysis error:", error)

        return (
            False,
            "The reflection could not be analysed. "
            "Today's Check-In Status has not changed.",
        )

    # do not store partial graph results if the analysis did not complete
    if not result.get("analysis_complete", False):
        return False, str(
            result.get(
                "error",
                "The reflection could not be analysed. "
                "Today's Check-In Status has not changed.",
            )
        )

    # replace any older reflection analysis with this completed result
    clear_previous_results()
    store_graph_results(result)

    # remember exactly which text produced the current analysis
    session["last_analysed_text"] = str(
        result.get("combined_analysis_text", analysis_text) or analysis_text
    ).strip()

    session["reflection_analysis_complete"] = True

    return True, ""


@review_blueprint.route("/")
def review_page():
    # review is available only to authenticated users
    if not require_account():
        flash("Please log in to continue.", "error")
        return redirect(url_for("auth.login"))

    mode = session.get("reflection_mode")

    # a valid reflection must exist before the Review page can open
    if mode not in (TEXT_REFLECTION, VOICE_REFLECTION, VIDEO_REFLECTION):
        flash("Please add a reflection before continuing.", "error")
        return redirect(url_for("reflection.reflection_page"))

    # text reflections can be analysed immediately without transcript confirmation
    if mode == TEXT_REFLECTION:
        typed_text = str(session.get("typed_reflection", "") or "").strip()
        analysed_text = str(session.get("last_analysed_text", "") or "").strip()

        # rerun analysis when the written reflection has changed
        if typed_text and typed_text != analysed_text:
            success, message = run_review_analysis()

            if not success:
                flash(message, "error")

    # voice and video require transcript confirmation first
    transcript_required = mode in (VOICE_REFLECTION, VIDEO_REFLECTION)
    transcript_confirmed = bool(session.get("transcript_review_saved", False))

    # analysis is considered usable only when both the flag and analysed text exist
    analysis_complete = bool(
        session.get("reflection_analysis_complete", False)
        and session.get("last_analysed_text")
    )

    # obtain recording duration from the appropriate temporary processing result
    if mode == VOICE_REFLECTION:
        media_result = session.get("voice_processing_result") or {}
    elif mode == VIDEO_REFLECTION:
        media_result = session.get("video_processing_result") or {}
    else:
        media_result = {}

    duration_seconds = float(media_result.get("duration_seconds", 0) or 0)
    status = session.get("checkin_status")

    # pass review, scoring and approved evidence to the template
    return render_template(
        "review.html",
        reflection_mode=mode,
        text_reflection=TEXT_REFLECTION,
        voice_reflection=VOICE_REFLECTION,
        video_reflection=VIDEO_REFLECTION,
        mode_slug=get_mode_slug(mode),
        checkin_status=status,
        status_class=get_status_class(status),
        typed_reflection=session.get("typed_reflection", ""),
        transcript=session.get("transcript", ""),
        confirmed_transcript=session.get("confirmed_transcript", ""),
        transcript_required=transcript_required,
        transcript_confirmed=transcript_confirmed,
        analysis_complete=analysis_complete,
        emotions=session.get("suggested_text_emotions", []) or [],
        supportive_factors=session.get("suggested_supportive_factors", []) or [],
        strain_factors=session.get("suggested_strain_factors", []) or [],
        audio_observation=session.get("suggested_audio_observation"),
        visible_expression=session.get("suggested_visible_expression"),
        duration_seconds=duration_seconds,
        review_complete=bool(session.get("review_complete", False)),
        text_component_score=session.get("text_component_score"),
        audio_component_score=session.get("audio_component_score"),
        video_component_score=session.get("video_component_score"),
        reflection_score=session.get("reflection_score"),
        reflection_indicator=session.get("reflection_indicator"),
        question_score=session.get("question_score"),

        # expose only the categorical combined result, not the internal number
        overall_checkin_indicator=session.get("combined_wellbeing_indicator"),

        approved_emotions=session.get("approved_text_emotions", []) or [],
        approved_supportive=session.get("approved_supportive_factors", []) or [],
        approved_strain=session.get("approved_strain_factors", []) or [],
        approved_audio=session.get("approved_audio_observation"),
        approved_expression=session.get("approved_visible_expression"),
    )


@review_blueprint.route("/transcript", methods=["POST"])
def confirm_transcript():
    if not require_account():
        return redirect(url_for("auth.login"))

    mode = session.get("reflection_mode")

    # transcript confirmation has no role in text reflections
    if mode not in (VOICE_REFLECTION, VIDEO_REFLECTION):
        return redirect(url_for("review.review_page"))

    corrected = str(
        request.form.get("confirmed_transcript", "") or ""
    ).strip()

    if not corrected:
        flash("Please review the transcript before continuing.", "error")
        return redirect(url_for("review.review_page"))

    # save the corrected text and invalidate any analysis of an older transcript
    session["confirmed_transcript"] = corrected
    session["transcript_review_saved"] = True
    session["last_analysed_text"] = ""
    session["reflection_analysis_complete"] = False

    success, message = run_review_analysis()

    if success:
        flash("Transcript confirmed.", "success")
    else:
        flash(message, "error")

    return redirect(url_for("review.review_page"))


@review_blueprint.route("/analyse", methods=["POST"])
def analyse_again():
    if not require_account():
        return redirect(url_for("auth.login"))

    mode = session.get("reflection_mode")

    # voice and video cannot be analysed before transcript confirmation
    if (
        mode in (VOICE_REFLECTION, VIDEO_REFLECTION)
        and not session.get("transcript_review_saved", False)
    ):
        flash("Please confirm the transcript first.", "error")
        return redirect(url_for("review.review_page"))

    success, message = run_review_analysis()

    if success:
        flash("Reflection analysis refreshed.", "success")
    else:
        flash(message, "error")

    return redirect(url_for("review.review_page"))


@review_blueprint.route("/save", methods=["POST"])
def save_review():
    if not require_account():
        return redirect(url_for("auth.login"))

    # review choices should only be accepted after analysis has completed
    if not session.get("reflection_analysis_complete", False):
        flash("Analyse the reflection before continuing.", "error")
        return redirect(url_for("review.review_page"))

    # only explicitly kept descriptive observations become approved evidence
    save_review_choices_from_form(request.form)

    flash("Your reflection choices have been saved.", "success")

    # continue to the Dashboard for summary generation and final review
    return redirect(url_for("dashboard.dashboard_page"))