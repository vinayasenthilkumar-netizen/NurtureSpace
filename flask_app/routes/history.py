
# calendar is used to build the month view shown on the History page
import calendar

# stored database timestamps are converted from UTC into Singapore local time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# flask helpers for routes, templates, query parameters, sessions and redirects
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# answer-label dictionaries are reused to convert saved numeric responses
# back into the wording shown to the user
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

# saved structured check-ins and reflection-only entries are stored separately
from database.checkins import delete_checkin, get_checkins_for_user
from database.reflections import (
    delete_saved_reflection,
    get_saved_reflections_for_user,
)

# history requires an authenticated account
from flask_app.routes import account_required

# builds the higher-level Personal Pattern Analysis cards
from patterns.summary import get_pattern_summary_for_user


# all History routes are grouped under /history
history_blueprint = Blueprint(
    "history",
    __name__,
    url_prefix="/history",
)


# shared authentication decorator with a History-specific message
login_required = account_required(
    "Please log in to view your history.",
    "warning",
)


# dates displayed to the user should follow Singapore local time
SINGAPORE_TIMEZONE = ZoneInfo("Asia/Singapore")


# four scored questionnaire fields shown in saved full check-ins
CORE_RESPONSE_FIELDS = (
    ("sleep", "Sleep", SLEEP_LABELS),
    ("mood", "Mood", MOOD_LABELS),
    ("stress", "Stress", STRESS_LABELS),
    ("support", "Support", SUPPORT_LABELS),
)


# six context-only questionnaire fields shown separately in History
CONTEXT_RESPONSE_FIELDS = (
    ("food", "Food", FOOD_LABELS),
    (
        "medication",
        "Medication / supplements",
        MEDICATION_LABELS,
    ),
    (
        "physical_recovery",
        "Physical recovery",
        PHYSICAL_RECOVERY_LABELS,
    ),
    ("baby_care", "Baby care", BABY_CARE_LABELS),
    (
        "other_responsibilities",
        "Other responsibilities",
        OTHER_RESPONSIBILITIES_LABELS,
    ),
    ("personal_care", "Personal care", PERSONAL_CARE_LABELS),
)


# labels are also passed to JavaScript for the trend graph scale
TREND_LABELS = {
    "sleep": SLEEP_LABELS,
    "mood": MOOD_LABELS,
    "stress": STRESS_LABELS,
    "support": SUPPORT_LABELS,
}


def clean_text(value):
    return str(value or "").strip()


def clean_list(values):
    if not values:
        return []

    # older or simplified values may occasionally arrive as one string
    if isinstance(values, str):
        value = values.strip()
        return [value] if value else []

    cleaned = []

    # preserve original order while removing empty and duplicate values
    for value in values:
        item = clean_text(value)

        if item and item not in cleaned:
            cleaned.append(item)

    return cleaned


def history_redirect():
    return redirect(url_for("history.history_page"))


def parse_database_datetime(value):
    text = clean_text(value)

    if not text:
        return None

    parsed = None

    # first try normal ISO format, including timestamps ending in Z
    try:
        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )

    except ValueError:
        # support older database formats that may not include timezone details
        for date_format in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                parsed = datetime.strptime(
                    text,
                    date_format,
                )
                break
            except ValueError:
                continue

    # unrecognised timestamps are skipped rather than breaking History
    if parsed is None:
        return None

    # timestamps without timezone information are treated as stored UTC
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    # calendar dates and displayed times use Singapore local time
    return parsed.astimezone(SINGAPORE_TIMEZONE)


def parse_date_parameter(value):
    try:
        return datetime.strptime(
            clean_text(value),
            "%Y-%m-%d",
        ).date()
    except ValueError:
        return None


def parse_month_parameter(value):
    try:
        parsed = datetime.strptime(
            clean_text(value),
            "%Y-%m",
        )
        return parsed.year, parsed.month
    except ValueError:
        return None


def shift_month(year, month, amount):
    # converting to one month index makes year boundaries easier to handle
    month_index = year * 12 + month - 1 + amount

    return (
        month_index // 12,
        month_index % 12 + 1,
    )


def get_answer_label(value, labels):
    if value is None:
        return "Not recorded"

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        # retain unexpected legacy text rather than hiding it completely
        return clean_text(value) or "Not recorded"

    # label dictionaries may use either integer or string keys
    label = labels.get(numeric_value)

    if label is None:
        label = labels.get(str(numeric_value))

    return clean_text(label) or str(numeric_value)


def build_response_rows(answers, fields):
    return [
        {
            "label": label,
            "value": get_answer_label(
                answers.get(field),
                labels,
            ),
        }
        for field, label, labels in fields
    ]


def prepare_approved_content(record):
    # only user-approved observations and confirmed summary content
    # are prepared for display in saved History
    return {
        "approved_emotions": clean_list(
            record.get("approved_text_emotions", [])
        ),
        "approved_supportive": clean_list(
            record.get("approved_supportive_factors", [])
        ),
        "approved_strain": clean_list(
            record.get("approved_strain_factors", [])
        ),
        "approved_audio": clean_text(
            record.get("approved_vocal_observation")
        ),
        "approved_expression": clean_text(
            record.get("approved_visible_observation")
        ),
        "summary": clean_text(
            record.get("confirmed_summary")
        ),
        "appointment_points": clean_list(
            record.get("confirmed_appointment_points", [])
        ),
    }


def prepare_entry_date(created_at):
    return {
        # machine-friendly key used for grouping and calendar lookup
        "date_key": created_at.strftime("%Y-%m-%d"),

        # human-readable date and time displayed inside the entry
        "date_text": created_at.strftime("%d %B %Y"),
        "time_text": created_at.strftime("%I:%M %p").lstrip("0"),
    }


def prepare_checkin_entry(checkin):
    created_at = parse_database_datetime(
        checkin.get("created_at")
    )

    # invalid dates cannot be placed reliably on the calendar
    if created_at is None:
        return None

    core_answers = checkin.get("core_answers", {}) or {}
    context_answers = checkin.get("context_answers", {}) or {}

    # a full check-in may have both questionnaire and reflection information
    combined_indicator = clean_text(
        checkin.get("combined_wellbeing_indicator")
    )

    reflection_indicator = clean_text(
        checkin.get("reflection_indicator")
    )

    # prefer the combined user-facing indicator whenever it was available
    if combined_indicator:
        indicator_label = "Overall Check-In Indicator"
        indicator = combined_indicator
    else:
        # older or incomplete full entries may contain only a reflection indicator
        indicator_label = "Personal Check-In Indicator"
        indicator = reflection_indicator

    entry = {
        "id": checkin.get("id"),
        "entry_type": "checkin",

        # questionnaire status is kept separately from the overall indicator
        "checkin_status": clean_text(
            checkin.get("checkin_status")
        ),

        "reflection_mode": clean_text(
            checkin.get("reflection_mode")
        ),

        "indicator_label": indicator_label,
        "indicator": indicator,

        # component indicators remain visible for a saved full check-in
        "questionnaire_indicator": clean_text(
            checkin.get("checkin_status")
        ),

        "reflection_indicator": clean_text(
            checkin.get("reflection_indicator")
        ),

        # translate saved numeric questionnaire values back to readable labels
        "core_responses": build_response_rows(
            core_answers,
            CORE_RESPONSE_FIELDS,
        ),

        "context_responses": build_response_rows(
            context_answers,
            CONTEXT_RESPONSE_FIELDS,
        ),
    }

    # add shared date fields and approved reflection/summary content
    entry.update(prepare_entry_date(created_at))
    entry.update(prepare_approved_content(checkin))

    return entry


def prepare_reflection_entry(reflection):
    created_at = parse_database_datetime(
        reflection.get("created_at")
    )

    if created_at is None:
        return None

    # reflection-only entries do not contain questionnaire response rows
    entry = {
        "id": reflection.get("id"),
        "entry_type": "reflection",
        "reflection_mode": clean_text(
            reflection.get("reflection_mode")
        ),
        "indicator_label": "Personal Check-In Indicator",
        "indicator": clean_text(
            reflection.get("reflection_indicator")
        ),
    }

    # use the same date and approved-content structure as full check-ins
    entry.update(prepare_entry_date(created_at))
    entry.update(prepare_approved_content(reflection))

    return entry


def get_combined_history(user_id):
    # these are stored in separate database tables but displayed together
    checkins = get_checkins_for_user(user_id) or []
    reflections = get_saved_reflections_for_user(user_id) or []

    entries = []

    # normalise full check-ins into the common History entry structure
    for checkin in checkins:
        prepared = prepare_checkin_entry(checkin)

        if prepared:
            entries.append(prepared)

    # add reflection-only records using the same common structure
    for reflection in reflections:
        prepared = prepare_reflection_entry(reflection)

        if prepared:
            entries.append(prepared)

    # latest saved entries appear first on the History page
    entries.sort(
        key=lambda item: (
            item["date_key"],
            item["time_text"],
        ),
        reverse=True,
    )

    return entries


def group_entries_by_date(entries):
    grouped = {}

    for entry in entries:
        grouped.setdefault(
            entry["date_key"],
            [],
        ).append(entry)

    return grouped


def build_calendar_month(
    year,
    month,
    entries_by_date,
    selected_date_key,
):
    # firstweekday=0 keeps Monday as the first day of the week
    month_calendar = calendar.Calendar(firstweekday=0)
    weeks = []

    for week in month_calendar.monthdayscalendar(year, month):
        prepared_week = []

        for day_number in week:
            # zero is used by calendar for empty cells outside the selected month
            if day_number == 0:
                prepared_week.append({"day": None})
                continue

            date_key = (
                f"{year:04d}-{month:02d}-{day_number:02d}"
            )

            # one day can contain more than one saved entry
            day_entries = entries_by_date.get(
                date_key,
                [],
            )

            prepared_week.append({
                "day": day_number,
                "date_key": date_key,

                # separate markers let the UI distinguish full and reflection-only entries
                "has_checkin": any(
                    entry.get("entry_type") == "checkin"
                    for entry in day_entries
                ),
                "has_reflection": any(
                    entry.get("entry_type") == "reflection"
                    for entry in day_entries
                ),

                "entry_count": len(day_entries),
                "selected": date_key == selected_date_key,
            })

        weeks.append(prepared_week)

    return weeks


def safe_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None

    # trend graph only accepts values from the questionnaire scale
    return score if score in (1, 2, 3, 4, 5) else None


def build_checkin_trend_data(user_id):
    # reflection-only entries do not contain the four scored questionnaire values
    checkins = get_checkins_for_user(user_id) or []
    latest_by_date = {}

    for checkin in checkins:
        created_at = parse_database_datetime(
            checkin.get("created_at")
        )

        if created_at is None:
            continue

        date_key = created_at.strftime("%Y-%m-%d")
        existing = latest_by_date.get(date_key)

        # use one questionnaire point per day so repeated same-day saves
        # do not distort the trend graph
        if (
            existing is None
            or created_at > existing["created_at"]
        ):
            latest_by_date[date_key] = {
                "created_at": created_at,
                "checkin": checkin,
            }

    # Today label also follows Singapore local date
    today = datetime.now(
        SINGAPORE_TIMEZONE
    ).date()

    graph_points = []

    for date_key, item in latest_by_date.items():
        checkin = item["checkin"]
        created_at = item["created_at"]
        core_answers = checkin.get("core_answers", {}) or {}

        # make the latest-day label easier to recognise in the chart
        display_date = (
            "Today"
            if created_at.date() == today
            else created_at.strftime("%d %b").lstrip("0")
        )

        # only the four core questionnaire responses are plotted
        graph_points.append({
            "date": date_key,
            "display_date": display_date,
            "sleep": safe_score(core_answers.get("sleep")),
            "mood": safe_score(core_answers.get("mood")),
            "stress": safe_score(core_answers.get("stress")),
            "support": safe_score(core_answers.get("support")),
        })

    # chart points need chronological order rather than History's newest-first order
    graph_points.sort(
        key=lambda item: item["date"]
    )

    return graph_points


def flash_delete_result(
    deleted,
    *,
    success_message,
    error_message,
):
    flash(
        success_message if deleted else error_message,
        "success" if deleted else "error",
    )


@history_blueprint.route(
    "/delete-checkin/<int:checkin_id>",
    methods=["POST"],
)
@login_required
def delete_checkin_entry(checkin_id):
    # user_id prevents one account from deleting another user's record
    deleted = delete_checkin(
        checkin_id=checkin_id,
        user_id=session.get("user_id"),
    )

    flash_delete_result(
        deleted,
        success_message="The saved check-in was deleted.",
        error_message="The saved check-in could not be deleted.",
    )

    return history_redirect()


@history_blueprint.route(
    "/delete-reflection/<int:reflection_id>",
    methods=["POST"],
)
@login_required
def delete_reflection_entry(reflection_id):
    deleted = delete_saved_reflection(
        reflection_id=reflection_id,
        user_id=session.get("user_id"),
    )

    flash_delete_result(
        deleted,
        success_message="The saved reflection was deleted.",
        error_message="The saved reflection could not be deleted.",
    )

    return history_redirect()


@history_blueprint.route("/")
@login_required
def history_page():
    user_id = session.get("user_id")

    try:
        # merge saved full check-ins and reflection-only records
        entries = get_combined_history(user_id)

    except Exception as error:
        # keep the page available even if History retrieval fails
        print("History loading error:", error)
        entries = []

        flash(
            "Your saved history could not be loaded.",
            "error",
        )

    # grouping once avoids repeatedly searching the entire History list
    entries_by_date = group_entries_by_date(entries)

    # calendar links may explicitly request a saved date
    requested_date = parse_date_parameter(
        request.args.get("date")
    )

    if requested_date:
        selected_date = requested_date

    elif entries:
        # by default select the date of the newest History entry
        selected_date = datetime.strptime(
            entries[0]["date_key"],
            "%Y-%m-%d",
        ).date()

    else:
        # when there is no History, open on today's Singapore date
        selected_date = datetime.now(
            SINGAPORE_TIMEZONE
        ).date()

    selected_date_key = selected_date.strftime("%Y-%m-%d")

    # all entries saved on the selected date are shown below the calendar
    selected_entries = entries_by_date.get(
        selected_date_key,
        [],
    )

    # month navigation can be independent from the selected entry date
    requested_month = parse_month_parameter(
        request.args.get("month")
    )

    if requested_month:
        year, month = requested_month
    else:
        year, month = selected_date.year, selected_date.month

    # prepare the complete month grid including entry markers
    calendar_weeks = build_calendar_month(
        year=year,
        month=month,
        entries_by_date=entries_by_date,
        selected_date_key=selected_date_key,
    )

    # calculate the keys used by the previous and next month controls
    previous_year, previous_month = shift_month(
        year,
        month,
        -1,
    )

    next_year, next_month = shift_month(
        year,
        month,
        1,
    )

    # counts are kept separate because full and reflection-only entries differ
    checkin_count = sum(
        entry["entry_type"] == "checkin"
        for entry in entries
    )

    reflection_count = sum(
        entry["entry_type"] == "reflection"
        for entry in entries
    )

    try:
        # trend chart uses structured questionnaire history only
        graph_points = build_checkin_trend_data(
            user_id
        )

    except Exception as error:
        # History details remain usable even if trend preparation fails
        print("Trend graph loading error:", error)
        graph_points = []

    try:
        # Personal Pattern Analysis works from the user's saved history
        pattern_summary = get_pattern_summary_for_user(
            user_id
        )

    except Exception as error:
        print("Personal pattern loading error:", error)

        # provide a stable structure so the template can show a safe fallback
        pattern_summary = {
            "available": False,
            "message": (
                "Personal patterns could not be loaded "
                "at the moment."
            ),
            "cards": [],
        }

    # sidebar greeting follows the same display-name preference as other pages
    welcome_name = (
        session.get("display_name")
        or session.get("username")
        or "there"
    )

    # provide all History, calendar, graph and Personal Pattern data to the template
    return render_template(
        "history.html",
        welcome_name=welcome_name,

        # calendar layout and currently visible month
        calendar_weeks=calendar_weeks,
        month_name=calendar.month_name[month],
        calendar_year=year,
        current_month_key=f"{year:04d}-{month:02d}",

        # previous / next controls preserve YYYY-MM format
        previous_month_key=(
            f"{previous_year:04d}-{previous_month:02d}"
        ),
        next_month_key=(
            f"{next_year:04d}-{next_month:02d}"
        ),

        # currently selected calendar day and its saved entries
        selected_date_key=selected_date_key,
        selected_date_text=selected_date.strftime(
            "%d %B %Y"
        ),
        selected_entries=selected_entries,

        # counts distinguish full check-ins from reflection-only saves
        checkin_count=checkin_count,
        reflection_count=reflection_count,

        # trend data and answer-scale labels are used by Chart.js in the template
        graph_points=graph_points,
        trend_labels=TREND_LABELS,

        # Personal Pattern Analysis is kept separate from individual History entries
        pattern_summary=pattern_summary,

        # template uses this to switch between History and empty-state layouts
        has_history=bool(entries),
    )