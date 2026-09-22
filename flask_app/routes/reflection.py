
# flask helpers used for routing, forms, templates, sessions and redirects
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# reflection feature switches and the maximum recording duration
from config.settings import (
    ENABLE_TEXT_REFLECTION,
    ENABLE_VIDEO_REFLECTION,
    ENABLE_VOICE_REFLECTION,
    MAX_RECORDING_SECONDS,
)

# internal constants used to identify each reflection mode
from core.constants import (
    TEXT_REFLECTION,
    VIDEO_REFLECTION,
    VOICE_REFLECTION,
)

# shared account-session check used by authenticated routes
from flask_app.routes import has_account_session as require_account

# parent workflow graph and the stage used for reflection processing
from graphs.guided_workflow_graph import (
    GUIDED_STAGE_REFLECTION,
    guided_workflow_graph,
)

# cleanup helpers remove temporary files created during media processing
from services.video_processing import cleanup_video_result
from services.voice_processing import cleanup_voice_result


# all reflection routes are grouped under /reflection
reflection_blueprint = Blueprint(
    "reflection",
    __name__,
    url_prefix="/reflection",
)


# maps the simple URL values to the internal reflection constants
MODE_MAP = {
    "text": TEXT_REFLECTION,
    "voice": VOICE_REFLECTION,
    "video": VIDEO_REFLECTION,
}


# these questionnaire values are removed when reflection is opened as a standalone check-in
STANDALONE_CHECKIN_KEYS = (
    "checkin_core_answers",
    "checkin_context_answers",
    "checkin_status",
    "status_breakdown",
    "context_result",
    "questions_completed",
    "question_score",
    "combined_wellbeing_score",
    "combined_wellbeing_indicator",
    "saved_checkin_id",
)


# temporary Reflection, Review and Dashboard values belonging to the current reflection
REFLECTION_RESULT_KEYS = (
    "reflection_mode",
    "typed_reflection",
    "transcript",
    "confirmed_transcript",

    # temporary non-file media processing results
    "voice_processing_result",
    "video_processing_result",
    "voice_processing_errors",
    "video_processing_errors",

    # model suggestions shown on the Review page
    "suggested_text_emotions",
    "all_text_emotion_scores",
    "suggested_supportive_factors",
    "suggested_strain_factors",
    "suggested_audio_observation",
    "suggested_visible_expression",

    # observations explicitly approved by the user during Review
    "approved_text_emotions",
    "approved_supportive_factors",
    "approved_strain_factors",
    "approved_audio_observation",
    "approved_visible_expression",

    # deterministic component and overall reflection results
    "text_component_score",
    "audio_component_score",
    "video_component_score",
    "reflection_score",
    "reflection_indicator",
    "combined_wellbeing_score",
    "combined_wellbeing_indicator",

    # review completion state
    "text_review_saved",
    "theme_review_saved",
    "transcript_review_saved",
    "audio_review_saved",
    "expression_review_saved",

    "reflection_analysis_complete",
    "review_complete",
    "last_analysed_text",
    "reflection_ready",

    # generated Dashboard summary state
    "summary_draft_generated",
    "summary_review_saved",
    "summary_edit_mode",
    "summary_source_signature",
    "summary_generation_error",
    "personal_summary",
    "appointment_points",
)


def login_redirect():
    return redirect(url_for("auth.login"))


def reflection_redirect(mode):
    return redirect(url_for("reflection.reflection_page", mode=mode))


def review_redirect():
    return redirect(url_for("review.review_page"))


def clean_text(value):
    return str(value or "").strip()


def clear_previous_reflection_results():
    # only reflection-related values are removed here
    # questionnaire state remains available for a combined check-in
    for key in REFLECTION_RESULT_KEYS:
        session.pop(key, None)


def initialise_new_reflection_state():
    # start all model suggestion lists empty until Review analysis runs
    session.update({
        "suggested_text_emotions": [],
        "all_text_emotion_scores": [],
        "suggested_supportive_factors": [],
        "suggested_strain_factors": [],
        "suggested_audio_observation": None,
        "suggested_visible_expression": None,

        # nothing has been approved by the user yet
        "approved_text_emotions": [],
        "approved_supportive_factors": [],
        "approved_strain_factors": [],
        "approved_audio_observation": None,
        "approved_visible_expression": None,

        # scoring is calculated later by the Review graph
        "text_component_score": None,
        "audio_component_score": None,
        "video_component_score": None,
        "reflection_score": None,
        "reflection_indicator": None,

        # combined values are available only when questionnaire data also exists
        "combined_wellbeing_score": None,
        "combined_wellbeing_indicator": None,

        # Review decisions have not yet been completed
        "text_review_saved": False,
        "theme_review_saved": False,
        "transcript_review_saved": False,
        "audio_review_saved": False,
        "expression_review_saved": False,

        "reflection_analysis_complete": False,
        "review_complete": False,
        "last_analysed_text": "",
        "reflection_ready": False,

        # a new reflection also invalidates any previous Dashboard draft
        "summary_draft_generated": False,
        "summary_review_saved": False,
        "summary_edit_mode": False,
        "summary_source_signature": None,
        "summary_generation_error": "",
        "personal_summary": "",
        "appointment_points": [],
    })


def prepare_new_reflection():
    # remove previous reflection data first, then restore a clean set of defaults
    clear_previous_reflection_results()
    initialise_new_reflection_state()


def run_reflection_graph(
    mode,
    reflection_text="",
    voice_input=None,
    video_input=None,
):
    # the parent graph is told explicitly which stage should run
    return guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_REFLECTION,
        "reflection_mode": mode,
        "reflection_text": reflection_text,
        "voice_input": voice_input,
        "video_input": video_input,
        "reflection_processing_complete": False,
        "reflection_processing_errors": [],
    })


def get_processing_error(graph_result, processing_result, fallback):
    # prefer errors attached to the mode-specific processing result
    # and fall back to errors recorded by the parent graph
    errors = list(
        processing_result.get("errors", [])
        or graph_result.get("reflection_processing_errors", [])
        or []
    )

    if errors:
        return str(errors[0])

    # use the graph-level error if no detailed processing error is available
    return str(graph_result.get("error", fallback))


def get_media_file(source_field, recorded_field, upload_field):
    # the hidden source field is controlled by the Record / Upload UI toggle
    source = clean_text(
        request.form.get(source_field, "upload")
    )

    if source == "record":
        return request.files.get(recorded_field)

    return request.files.get(upload_field)


def build_safe_media_result(
    processing_result,
    *,
    include_visible_expression=False,
):
    # transcript text may remain temporarily in the session for user review
    transcript = clean_text(
        processing_result.get("transcript", "")
    )

    # do not place file paths, raw media or temporary processing objects in session
    safe_result = {
        "processing_complete": bool(
            processing_result.get("processing_complete", False)
        ),
        "transcript": transcript,
        "transcription_success": bool(
            processing_result.get("transcription_success", False)
        ),
        "audio_observation": processing_result.get(
            "audio_observation"
        ),
        "duration_seconds": float(
            processing_result.get("duration_seconds", 0) or 0
        ),
        "errors": list(
            processing_result.get("errors", []) or []
        ),
    }

    # video has one additional model observation from sampled frames
    if include_visible_expression:
        safe_result["visible_expression"] = (
            processing_result.get("visible_expression")
        )

    return safe_result


def store_media_reflection(
    *,
    mode,
    processing_result,
    safe_result,
):
    transcript = safe_result["transcript"]

    # this media reflection replaces any earlier temporary reflection state
    prepare_new_reflection()

    # the raw file is not stored here, only information required for Review
    session.update({
        "reflection_mode": mode,
        "typed_reflection": "",
        "transcript": transcript,

        # initial confirmed value matches the transcript until the user reviews it
        "confirmed_transcript": transcript,

        # keep the tentative audio observation so it can be reviewed later
        "suggested_audio_observation": processing_result.get(
            "audio_observation"
        ),
        "reflection_ready": True,
    })

    if mode == VOICE_REFLECTION:
        # safe voice results contain no raw audio file
        session["voice_processing_result"] = safe_result
        session["voice_processing_errors"] = list(
            processing_result.get("errors", []) or []
        )

    elif mode == VIDEO_REFLECTION:
        # safe video results also exclude raw video, extracted audio and frames
        session["video_processing_result"] = safe_result
        session["video_processing_errors"] = list(
            processing_result.get("errors", []) or []
        )

        # visible-expression output remains tentative until Review approval
        session["suggested_visible_expression"] = (
            processing_result.get("visible_expression")
        )


@reflection_blueprint.route("/")
def reflection_page():
    # Reflection is available only to authenticated users
    if not require_account():
        flash("Please log in to continue.", "error")
        return login_redirect()

    # standalone mode starts a reflection without carrying questionnaire results
    standalone = request.args.get("standalone") == "1"

    if standalone:
        # remove questionnaire state so this remains a reflection-only entry
        for key in STANDALONE_CHECKIN_KEYS:
            session.pop(key, None)

        # a new standalone reflection should not point to an older saved reflection
        session["saved_reflection_id"] = None

    # select Text by default if no mode is supplied
    selected_mode = clean_text(
        request.args.get("mode", "text")
    ).lower()

    # unexpected URL values safely fall back to Text
    if selected_mode not in MODE_MAP:
        selected_mode = "text"

    return render_template(
        "reflection.html",
        selected_mode=selected_mode,
        ready=request.args.get("ready") == "1",
        reflection_mode=session.get("reflection_mode"),
        typed_reflection=session.get("typed_reflection", ""),
        transcript=session.get("transcript", ""),
        checkin_status=session.get("checkin_status"),

        # feature settings decide which tabs appear in the interface
        enable_text=ENABLE_TEXT_REFLECTION,
        enable_voice=ENABLE_VOICE_REFLECTION,
        enable_video=ENABLE_VIDEO_REFLECTION,

        # frontend recording controls use the same configured time limit
        max_seconds=MAX_RECORDING_SECONDS,
    )


@reflection_blueprint.route("/text", methods=["POST"])
def submit_text():
    if not require_account():
        return login_redirect()

    # remove surrounding whitespace from the submitted written reflection
    reflection_text = clean_text(
        request.form.get("reflection_text", "")
    )

    # empty text should never be sent to the graph
    if not reflection_text:
        flash(
            "Please write a reflection before continuing.",
            "error",
        )
        return reflection_redirect("text")

    try:
        # Text does not require media processing but still uses the Reflection graph
        graph_result = run_reflection_graph(
            mode=TEXT_REFLECTION,
            reflection_text=reflection_text,
        )

    except Exception as error:
        print(f"Text Reflection error: {error}")

        flash(
            "The written reflection could not be prepared. "
            "Please try again.",
            "error",
        )
        return reflection_redirect("text")

    # Review should open only when the graph confirms preparation succeeded
    if not graph_result.get("review_ready", False):
        flash(
            str(
                graph_result.get(
                    "error",
                    "The written reflection could not be prepared.",
                )
            ),
            "error",
        )
        return reflection_redirect("text")

    # discard any older temporary Reflection, Review and Dashboard state
    prepare_new_reflection()

    # store only the written text required for the next Review stage
    session.update({
        "reflection_mode": TEXT_REFLECTION,
        "typed_reflection": clean_text(
            graph_result.get(
                "reflection_text",
                reflection_text,
            )
            or reflection_text
        ),
        "transcript": "",
        "confirmed_transcript": "",
        "reflection_ready": True,
    })

    return review_redirect()


@reflection_blueprint.route("/voice", methods=["POST"])
def submit_voice():
    if not require_account():
        return login_redirect()

    # select the browser recording or uploaded file based on the form toggle
    voice_file = get_media_file(
        "voice_source",
        "voice_recorded_file",
        "voice_upload_file",
    )

    if voice_file is None or not voice_file.filename:
        flash(
            "Please record or upload a voice reflection.",
            "error",
        )
        return reflection_redirect("voice")

    # kept outside try so temporary files can still be cleaned in finally
    voice_result = {}

    try:
        # voice processing handles transcription and vocal-pattern analysis
        graph_result = run_reflection_graph(
            mode=VOICE_REFLECTION,
            voice_input=voice_file,
        )

        voice_result = (
            graph_result.get("voice_processing_result", {})
            or {}
        )

        # do not continue if the graph could not prepare the recording for Review
        if not graph_result.get("review_ready", False):
            flash(
                get_processing_error(
                    graph_result,
                    voice_result,
                    "The voice reflection could not be processed.",
                ),
                "error",
            )
            return reflection_redirect("voice")

        # strip out temporary file information before putting results in session
        safe_voice_result = build_safe_media_result(
            voice_result
        )

        # Review needs transcript and model observations, not the raw recording
        store_media_reflection(
            mode=VOICE_REFLECTION,
            processing_result=voice_result,
            safe_result=safe_voice_result,
        )

    except Exception as error:
        print(f"Voice Reflection error: {error}")

        flash(
            "The voice reflection could not be processed. "
            "Please try again.",
            "error",
        )
        return reflection_redirect("voice")

    finally:
        # cleanup runs whether processing succeeds, fails or returns early
        if isinstance(voice_result, dict):
            cleanup_voice_result(voice_result)

    # user confirms the transcript on the Review page before text analysis
    return review_redirect()


@reflection_blueprint.route("/video", methods=["POST"])
def submit_video():
    if not require_account():
        return login_redirect()

    # select either the newly recorded video or an uploaded file
    video_file = get_media_file(
        "video_source",
        "video_recorded_file",
        "video_upload_file",
    )

    if video_file is None or not video_file.filename:
        flash(
            "Please record or upload a video reflection.",
            "error",
        )
        return reflection_redirect("video")

    # defined before try so cleanup can always access the result
    video_result = {}

    try:
        # video processing coordinates expression, transcription and audio analysis
        graph_result = run_reflection_graph(
            mode=VIDEO_REFLECTION,
            video_input=video_file,
        )

        video_result = (
            graph_result.get("video_processing_result", {})
            or {}
        )

        # processing must reach review_ready before moving to Review
        if not graph_result.get("review_ready", False):
            flash(
                get_processing_error(
                    graph_result,
                    video_result,
                    "The video reflection could not be processed.",
                ),
                "error",
            )
            return reflection_redirect("video")

        # retain only safe non-file values required by later routes
        safe_video_result = build_safe_media_result(
            video_result,
            include_visible_expression=True,
        )

        # transcript and observations remain temporary until Review decisions
        store_media_reflection(
            mode=VIDEO_REFLECTION,
            processing_result=video_result,
            safe_result=safe_video_result,
        )

    except Exception as error:
        print(f"Video Reflection error: {error}")

        flash(
            "The video reflection could not be processed. "
            "Please try again.",
            "error",
        )
        return reflection_redirect("video")

    finally:
        # remove raw video, extracted audio, frames and other temporary files
        if isinstance(video_result, dict):
            cleanup_video_result(video_result)

    return review_redirect()