
# imports saved approved check in used for personal pattern analysis
from database.checkins import get_checkins_for_user

# structured fields used across the longitudinal checks
CORE_FIELDS = [
    "sleep",
    "mood",
    "stress",
    "support",
]

CONTEXT_FIELDS = [
    "food",
    "medication",
    "physical_recovery",
    "baby_care",
    "other_responsibilities",
    "personal_care",
]

STRUCTURED_FIELDS = (CORE_FIELDS+ CONTEXT_FIELDS)

# settings used for personal baseline comparison
RECENT_CHECKIN_COUNT = 3
MINIMUM_BASELINE_CHECKINS = 3
MINIMUM_CHANGE = 1.0
# settings used for recurring themes
RECURRING_LOOKBACK = 5
MINIMUM_RECURRING_OCCURRENCES = 3
# settings used for emerging themes
EMERGING_RECENT_COUNT = 2
EMERGING_EARLIER_COUNT = 3
# settings used for variability
VARIABILITY_LOOKBACK = 5
MINIMUM_VARIABILITY_VALUES = 3
MINIMUM_VARIABILITY_RANGE = 2
# settings used for reflection and check in alignment
ALIGNMENT_LOOKBACK = 5
MINIMUM_ALIGNMENT_OCCURRENCES = 2
LOW_RESPONSE_THRESHOLD = 2
HIGH_RESPONSE_THRESHOLD = 4

# keywords used to connect approved reflection content with structured fields
ALIGNMENT_KEYWORDS = {
    "sleep": [
        "sleep",
        "tired",
        "tiredness",
        "exhaust",
        "fatigue",
        "low energy",
    ],

    "mood": [
        "mood",
        "sad",
        "sadness",
        "low mood",
        "frustrat",
        "emotion",
    ],

    "stress": [
        "stress",
        "overwhelm",
        "pressure",
        "frustrat",
        "tension",
    ],

    "support": [
        "support",
        "help",
        "alone",
        "isolat",
    ],

    "food": [
        "food",
        "meal",
        "eat",
        "eating",
    ],

    "medication": [
        "medication",
        "medicine",
        "supplement",
    ],

    "physical_recovery": [
        "physical recovery",
        "recovery",
        "pain",
        "discomfort",
    ],

    "baby_care": [
        "baby care",
        "childcare",
        "caring for the baby",
        "feeding",
    ],

    "other_responsibilities": [
        "responsibilit",
        "household",
        "work pressure",
        "chores",
    ],

    "personal_care": [
        "personal care",
        "self-care",
        "self care",
        "personal time",
        "time for myself",
    ],
}


# returns the average for a list of stored response values
def calculate_average(values):
    if not values:
        return None

    return sum(values) / len(values)


# gets one structured answer while ignoring context values marked not applicable
def get_structured_answer(checkin, field_name):
    if field_name in CORE_FIELDS:
        answers = checkin.get("core_answers", {})
    else:
        answers = checkin.get("context_answers", {})

    value = answers.get(field_name)

    if value is None:
        return None
    # context value 0 should not be treated as a numerical wellbeing response
    if field_name in CONTEXT_FIELDS and value == 0:
        return None

    return value


# combines approved supportive and strain factors into one theme list
def get_approved_themes(checkin):
    supportive_factors = checkin.get("approved_supportive_factors", [])
    strain_factors = checkin.get("approved_strain_factors", [])
    themes = []

    for theme in supportive_factors:
        clean_theme = str(theme or "").strip()

        if clean_theme:
            themes.append({
                "theme": clean_theme,
                "type": "supportive",
            })
    for theme in strain_factors:
        clean_theme = str(theme or "").strip()

        if clean_theme:
            themes.append({
                "theme": clean_theme,
                "type": "strain",
            })

    return themes


# prepares approved reflection information for alignment checking
def get_reflection_items(checkin):
    items = []

    supportive_factors = checkin.get("approved_supportive_factors", [])
    strain_factors = checkin.get("approved_strain_factors", [])
    text_emotions = checkin.get("approved_text_emotions", [])
    # practical factors already have a supportive or strain meaning
    for factor in supportive_factors:
        clean_factor = str(factor or "").strip()

        if clean_factor:
            items.append({
                "text": clean_factor,
                "type": "supportive",
            })

    for factor in strain_factors:
        clean_factor = str(factor or "").strip()

        if clean_factor:
            items.append({
                "text": clean_factor,
                "type": "strain",
            })

    supportive_emotions = {
        "positive",
        "warmth",
    }

    strain_emotions = {
        "low mood",
        "worry",
        "frustration",
    }

    # only known approved emotion groups are used for alignment
    for emotion in text_emotions:
        clean_emotion = str(emotion or "").strip()
        emotion_key = clean_emotion.lower()

        if emotion_key in supportive_emotions:
            items.append({
                "text": clean_emotion,
                "type": "supportive",
            })

        elif emotion_key in strain_emotions:
            items.append({
                "text": clean_emotion,
                "type": "strain",
            })

    return items


# checks whether approved reflection wording relates to one structured field
def reflection_matches_field(reflection_text, field_name):
    clean_text = str(reflection_text or "").strip().lower()
    keywords = ALIGNMENT_KEYWORDS.get(field_name, [])

    return any(
        keyword in clean_text
        for keyword in keywords
    )


# converts a structured response into supportive, strain or neutral direction
def get_response_direction(field_name, value):
    if value is None:
        return "neutral"
    # stress is reversed because a higher response means more strain
    if field_name == "stress":
        if value >= HIGH_RESPONSE_THRESHOLD:
            return "strain"

        if value <= LOW_RESPONSE_THRESHOLD:
            return "supportive"

        return "neutral"

    if value <= LOW_RESPONSE_THRESHOLD:
        return "strain"

    if value >= HIGH_RESPONSE_THRESHOLD:
        return "supportive"

    return "neutral"


# compares recent structured responses with the user's earlier personal baseline
def detect_baseline_changes(checkins):
    minimum_required = MINIMUM_BASELINE_CHECKINS + RECENT_CHECKIN_COUNT

    if len(checkins) < minimum_required:
        return {
            "available": False,
            "minimum_required": minimum_required,
            "patterns": [],
        }

    # database history is newest first, so reverse it for baseline comparison
    ordered_checkins = list(reversed(checkins))
    baseline_checkins = ordered_checkins[:-RECENT_CHECKIN_COUNT]
    recent_checkins = ordered_checkins[-RECENT_CHECKIN_COUNT:]
    patterns = []

    for field_name in STRUCTURED_FIELDS:
        baseline_values = []
        recent_values = []

        for checkin in baseline_checkins:
            value = get_structured_answer(checkin, field_name)

            if value is not None:
                baseline_values.append(value)

        for checkin in recent_checkins:
            value = get_structured_answer(checkin, field_name)

            if value is not None:
                recent_values.append(value)

        # both periods need enough valid responses before they can be compared
        if len(baseline_values) < MINIMUM_BASELINE_CHECKINS:
            continue

        if len(recent_values) < RECENT_CHECKIN_COUNT:
            continue

        baseline_average = calculate_average(baseline_values)
        recent_average = calculate_average(recent_values)
        difference = recent_average - baseline_average

        if abs(difference) < MINIMUM_CHANGE:
            continue

        if difference > 0:
            direction = "higher"
        else:
            direction = "lower"

        patterns.append({
            "signal": "baseline_change",
            "field": field_name,
            "direction": direction,
            "baseline_average": round(baseline_average, 2),
            "recent_average": round(recent_average, 2),
            "difference": round(difference, 2),
        })

    return {
        "available": True,
        "patterns": patterns,
    }


# finds approved themes that repeat across several recent check-ins
def detect_recurring_themes(checkins):
    if len(checkins) < MINIMUM_RECURRING_OCCURRENCES:
        return {
            "available": False,
            "minimum_required": MINIMUM_RECURRING_OCCURRENCES,
            "patterns": [],
        }

    recent_checkins = checkins[:RECURRING_LOOKBACK]
    theme_counts = {}
    theme_details = {}

    for checkin in recent_checkins:
        themes = get_approved_themes(checkin)
        themes_seen_in_checkin = set()
        # the same theme is counted only once per check-in
        for theme_data in themes:
            theme_name = theme_data["theme"]
            theme_type = theme_data["type"]
            theme_key = (
                theme_name.strip().lower(),
                theme_type,
            )
            if theme_key in themes_seen_in_checkin:
                continue

            themes_seen_in_checkin.add(theme_key)

            theme_counts[theme_key] = (
                theme_counts.get(theme_key, 0) + 1
            )

            if theme_key not in theme_details:
                theme_details[theme_key] = {
                    "theme": theme_name,
                    "type": theme_type,
                }

    patterns = []

    for theme_key, occurrence_count in theme_counts.items():
        if occurrence_count < MINIMUM_RECURRING_OCCURRENCES:
            continue

        details = theme_details[theme_key]

        patterns.append({
            "signal": "recurring_theme",
            "theme": details["theme"],
            "type": details["type"],
            "occurrences": occurrence_count,
            "checkins_reviewed": len(recent_checkins),
        })
    # stronger recurring themes are shown first
    patterns.sort(
        key=lambda pattern: pattern["occurrences"],
        reverse=True,
    )

    return {
        "available": True,
        "patterns": patterns,
    }


# finds themes appearing in the latest check-ins but not the immediately earlier ones
def detect_emerging_themes(checkins):
    minimum_required = EMERGING_RECENT_COUNT + EMERGING_EARLIER_COUNT

    if len(checkins) < minimum_required:
        return {
            "available": False,
            "minimum_required": minimum_required,
            "patterns": [],
        }

    recent_checkins = checkins[:EMERGING_RECENT_COUNT]
    earlier_checkins = checkins[
        EMERGING_RECENT_COUNT:minimum_required
    ]

    recent_counts = {}
    recent_details = {}
    earlier_themes = set()

    # count themes appearing in the recent window
    for checkin in recent_checkins:
        themes = get_approved_themes(checkin)
        themes_seen_in_checkin = set()

        for theme_data in themes:
            theme_name = theme_data["theme"]
            theme_type = theme_data["type"]

            theme_key = (
                theme_name.strip().lower(),
                theme_type,
            )

            if theme_key in themes_seen_in_checkin:
                continue

            themes_seen_in_checkin.add(theme_key)

            recent_counts[theme_key] = (
                recent_counts.get(theme_key, 0) + 1
            )

            if theme_key not in recent_details:
                recent_details[theme_key] = {
                    "theme": theme_name,
                    "type": theme_type,
                }

    # collect themes from the earlier comparison window
    for checkin in earlier_checkins:
        themes = get_approved_themes(checkin)

        for theme_data in themes:
            theme_name = theme_data["theme"]
            theme_type = theme_data["type"]

            theme_key = (
                theme_name.strip().lower(),
                theme_type,
            )

            earlier_themes.add(theme_key)

    patterns = []

    for theme_key, occurrence_count in recent_counts.items():
        if occurrence_count < EMERGING_RECENT_COUNT:
            continue

        if theme_key in earlier_themes:
            continue

        details = recent_details[theme_key]

        patterns.append({
            "signal": "emerging_theme",
            "theme": details["theme"],
            "type": details["type"],
            "recent_occurrences": occurrence_count,
            "recent_checkins_reviewed": len(recent_checkins),
            "earlier_checkins_reviewed": len(earlier_checkins),
        })

    return {
        "available": True,
        "patterns": patterns,
    }


# finds structured fields that have changed noticeably across recent check-ins
def detect_wellbeing_variability(checkins):
    if len(checkins) < MINIMUM_VARIABILITY_VALUES:
        return {
            "available": False,
            "minimum_required": MINIMUM_VARIABILITY_VALUES,
            "patterns": [],
        }

    recent_checkins = checkins[:VARIABILITY_LOOKBACK]
    patterns = []

    for field_name in STRUCTURED_FIELDS:
        values = []

        for checkin in recent_checkins:
            value = get_structured_answer(checkin, field_name)

            if value is not None:
                values.append(value)

        if len(values) < MINIMUM_VARIABILITY_VALUES:
            continue

        lowest_value = min(values)
        highest_value = max(values)
        response_range = highest_value - lowest_value

        # small changes are not treated as variability patterns
        if response_range < MINIMUM_VARIABILITY_RANGE:
            continue

        patterns.append({
            "signal": "wellbeing_variability",
            "field": field_name,
            "lowest_value": lowest_value,
            "highest_value": highest_value,
            "range": response_range,
            "values_reviewed": len(values),
            "checkins_reviewed": len(recent_checkins),
        })

    return {
        "available": True,
        "patterns": patterns,
    }


# finds repeated agreement between structured answers and approved reflection content
def detect_reflection_checkin_alignment(checkins):
    if len(checkins) < MINIMUM_ALIGNMENT_OCCURRENCES:
        return {
            "available": False,
            "minimum_required": MINIMUM_ALIGNMENT_OCCURRENCES,
            "patterns": [],
        }

    recent_checkins = checkins[:ALIGNMENT_LOOKBACK]

    alignment_counts = {}
    alignment_details = {}

    for checkin in recent_checkins:
        reflection_items = get_reflection_items(checkin)
        alignments_seen = set()

        for field_name in STRUCTURED_FIELDS:
            value = get_structured_answer(checkin, field_name)
            response_direction = get_response_direction(
                field_name,
                value,
            )

            if response_direction == "neutral":
                continue

            matched_items = []

            # reflection items must match both the field and the same direction
            for reflection_item in reflection_items:
                if reflection_item["type"] != response_direction:
                    continue

                if reflection_matches_field(
                    reflection_item["text"],
                    field_name,
                ):
                    matched_items.append(
                        reflection_item["text"]
                    )

            if not matched_items:
                continue

            alignment_key = (
                field_name,
                response_direction,
            )

            if alignment_key in alignments_seen:
                continue

            alignments_seen.add(alignment_key)

            alignment_counts[alignment_key] = (
                alignment_counts.get(alignment_key, 0) + 1
            )

            if alignment_key not in alignment_details:
                alignment_details[alignment_key] = {
                    "field": field_name,
                    "direction": response_direction,
                    "examples": [],
                }

            stored_examples = alignment_details[
                alignment_key
            ]["examples"]

            # keep a small set of unique approved examples
            for matched_item in matched_items:
                if matched_item not in stored_examples:
                    stored_examples.append(matched_item)

    patterns = []

    for alignment_key, occurrence_count in alignment_counts.items():
        if occurrence_count < MINIMUM_ALIGNMENT_OCCURRENCES:
            continue

        details = alignment_details[alignment_key]

        patterns.append({
            "signal": "reflection_checkin_alignment",
            "field": details["field"],
            "direction": details["direction"],
            "occurrences": occurrence_count,
            "checkins_reviewed": len(recent_checkins),
            "matched_reflections": details["examples"][:3],
        })

    patterns.sort(
        key=lambda pattern: pattern["occurrences"],
        reverse=True,
    )
    return {
        "available": True,
        "patterns": patterns,
    }


# gets baseline-change patterns for one user's saved check-ins
def get_baseline_changes_for_user(user_id):
    checkins = get_checkins_for_user(user_id)
    return detect_baseline_changes(checkins)

# gets recurring approved themes for one user
def get_recurring_themes_for_user(user_id):
    checkins = get_checkins_for_user(user_id)
    return detect_recurring_themes(checkins)


# gets recently emerging themes for one user
def get_emerging_themes_for_user(user_id):

    checkins = get_checkins_for_user(user_id)
    return detect_emerging_themes(checkins)

# gets structured-response variability patterns for one user
def get_wellbeing_variability_for_user(user_id):

    checkins = get_checkins_for_user(user_id)
    return detect_wellbeing_variability(checkins)

# gets reflection and structured check-in alignment for one user
def get_reflection_alignment_for_user(user_id):
    checkins = get_checkins_for_user(user_id)
    return detect_reflection_checkin_alignment(checkins)

# runs all five personal pattern signals using the same approved history
def get_all_patterns_for_user(user_id):
    checkins = get_checkins_for_user(user_id)
    return {
        "baseline_changes": detect_baseline_changes(checkins),
        "recurring_themes": detect_recurring_themes(checkins),
        "emerging_themes": detect_emerging_themes(checkins),
        "wellbeing_variability": detect_wellbeing_variability(checkins),
        "reflection_checkin_alignment": (
            detect_reflection_checkin_alignment(checkins)
        ),
    }