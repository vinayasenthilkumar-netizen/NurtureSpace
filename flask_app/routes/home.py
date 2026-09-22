
# local date handling is needed because questionnaire behaviour depends on Singapore time
from datetime import datetime
from zoneinfo import ZoneInfo
# flask helpers for routing, templates, sessions and redirects
from flask import Blueprint, redirect, render_template, session, url_for
# application name shown on the public and signed-in home page
from config.settings import APP_NAME
# privacy version check ensures users have accepted the current privacy notice
from database.privacy import privacy_is_current
# resource helpers provide today's curated resources and existing bookmarks
from database.resources import (
    get_bookmarked_resource_ids,
    get_daily_resources,
)
# shared helper checks whether the user's privacy settings allow bookmarks
from flask_app.routes import can_save_bookmarks
# home blueprint owns the public landing page and signed-in Today page
home_blueprint = Blueprint("home", __name__)
# all daily questionnaire decisions use Singapore local time
SINGAPORE_TIMEZONE = ZoneInfo("Asia/Singapore")


def get_resource_duration_label(resource):
    # resources without an estimated duration do not need a time label
    estimated_minutes = resource.get("estimated_minutes")

    if not estimated_minutes:
        return ""

    # normalise the stored value before displaying it
    try:
        estimated_minutes = int(estimated_minutes)
    except (TypeError, ValueError):
        return ""

    # convert the resource type into natural wording for the Today card
    resource_type = str(
        resource.get("resource_type", "") or ""
    ).strip().lower()

    activity = {
        "article": "read",
        "blog": "read",
        "video": "video",
        "podcast": "podcast",
    }.get(resource_type, resource_type)

    # keep a useful duration even when no recognised resource type exists
    return (
        f"{estimated_minutes} min {activity}"
        if activity
        else f"{estimated_minutes} min"
    )


def get_questionnaire_content(is_monday):
    # Monday is the scheduled weekly questionnaire day
    if is_monday:
        return {
            "checkin_label": "Weekly Questionnaire",
            "checkin_title": "Your weekly questionnaire is ready",
            "checkin_message": (
                "A short set of wellbeing questions for your scheduled "
                "weekly check-in."
            ),
            "primary_button_text": "Start Questionnaire",

            # weekly questionnaire cannot be skipped from the Today page
            "allow_daily_skip": False,
        }

    # on other days the same questionnaire remains available but optional
    return {
        "checkin_label": "Optional Questionnaire",
        "checkin_title": "Would you like to answer today's questionnaire?",
        "checkin_message": (
            "Your main weekly questionnaire is on Monday. "
            "Today's questionnaire is optional."
        ),
        "primary_button_text": "Start Questionnaire",
        "allow_daily_skip": True,
    }


def get_questionnaire_skip_state(now, is_monday):
    # date string lets the session remember exactly which day was skipped
    today_key = now.date().isoformat()

    # Monday always clears any earlier optional-day skip
    if is_monday:
        session.pop("questions_skipped_today", None)
        session.pop("questions_skipped_date", None)
        return False

    # if no skip has been recorded there is nothing else to restore
    if not session.get("questions_skipped_today"):
        session.pop("questions_skipped_date", None)
        return False

    skipped_date = session.get("questions_skipped_date")

    # older sessions may have the skip flag without a date, so attach today's date
    if not skipped_date:
        session["questions_skipped_date"] = today_key
        return True

    # keep the skipped state while the user remains on the same calendar day
    if skipped_date == today_key:
        return True

    # a skip from a previous day should not carry into a new optional day
    session.pop("questions_skipped_today", None)
    session.pop("questions_skipped_date", None)
    return False


@home_blueprint.route("/")
def home():
    # account presence decides whether to show the landing page or application home
    user_id = session.get("user_id")

    if not user_id:
        return render_template(
            "home.html",
            app_name=APP_NAME,
            is_logged_in=False,
        )

    # signed-in users must have privacy preferences for the current privacy version
    try:
        if not privacy_is_current(user_id):
            return redirect(url_for("privacy.privacy_settings"))

    except Exception:
        # if privacy state cannot be confirmed, send the user back through privacy setup
        return redirect(url_for("privacy.privacy_settings"))

    # use Singapore time for Monday detection and the date displayed in the UI
    now = datetime.now(SINGAPORE_TIMEZONE)
    is_monday = now.weekday() == 0

    # optional-day skips are valid only for the day they were selected
    questionnaire_skipped_today = get_questionnaire_skip_state(
        now,
        is_monday,
    )

    # prefer the friendly display name shown elsewhere in the application
    display_name = str(
        session.get("display_name", "") or ""
    ).strip()

    username = str(
        session.get("username", "") or ""
    ).strip()

    welcome_name = display_name or username or "there"

    # load bookmark ids separately so Today cards can show saved / unsaved state
    try:
        bookmarked_resource_ids = get_bookmarked_resource_ids(
            user_id
        )

    except Exception as error:
        # resource cards can still be displayed even if bookmark lookup fails
        print(
            "Today bookmark lookup error:",
            error,
        )
        bookmarked_resource_ids = set()

    # retrieve a small daily selection from the curated resource collection
    try:
        daily_resources = get_daily_resources(
            maximum_resources=2,
            current_date=now.date(),
            user_id=user_id,
        )

    except Exception as error:
        # Today page should remain available even if resources cannot be loaded
        print(
            "Today daily-resource lookup error:",
            error,
        )
        daily_resources = []

    # add display-only duration text before passing resources to the template
    for resource in daily_resources:
        resource["duration_label"] = get_resource_duration_label(
            resource
        )

    # render the signed-in Today page with questionnaire and resource state
    return render_template(
        "home.html",
        app_name=APP_NAME,
        is_logged_in=True,
        welcome_name=welcome_name,

        # date information shown in the Today header
        day_name=now.strftime("%A"),
        date_text=now.strftime("%d %B %Y"),
        is_monday=is_monday,

        # controls whether the optional questionnaire card shows as skipped
        questionnaire_skipped_today=questionnaire_skipped_today,

        # curated resources and current bookmark state
        daily_resources=daily_resources,
        bookmarked_resource_ids=bookmarked_resource_ids,

        # privacy preference decides whether new bookmarks may be created
        bookmark_saving_allowed=can_save_bookmarks(user_id),

        # inject Monday-specific or optional-day questionnaire wording
        **get_questionnaire_content(is_monday),
    )