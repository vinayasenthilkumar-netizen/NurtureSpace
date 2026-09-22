
# imports the user's full pattern history and relevance selection logic
from patterns.history import get_all_patterns_for_user
from patterns.relevance import select_relevant_patterns


# wording used when a structured response changes from the users earlier baseline
FIELD_CHANGE_TEXT = {
    "sleep": {
        "lower": (
            "Your recent sleep quality responses have been "
            "lower than your earlier check-ins."
        ),
        "higher": (
            "Your recent sleep quality responses have been "
            "higher than your earlier check-ins."
        ),
    },

    "mood": {
        "lower": (
            "Your recent mood responses have been lower "
            "than your earlier check-ins."
        ),
        "higher": (
            "Your recent mood responses have been higher "
            "than your earlier check-ins."
        ),
    },

    "stress": {
        "lower": (
            "You have reported less stress recently than "
            "in your earlier check-ins."
        ),
        "higher": (
            "You have reported more stress recently than "
            "in your earlier check-ins."
        ),
    },

    "support": {
        "lower": (
            "You have reported feeling less supported "
            "recently than in your earlier check-ins."
        ),
        "higher": (
            "You have reported feeling more supported "
            "recently than in your earlier check-ins."
        ),
    },

    "food": {
        "lower": (
            "Your recent responses suggest that eating "
            "enough for your own needs has been more "
            "difficult than in your earlier check-ins."
        ),
        "higher": (
            "Your recent responses suggest that eating "
            "enough for your own needs has been easier "
            "than in your earlier check-ins."
        ),
    },

    "medication": {
        "lower": (
            "Your recent responses about taking medicines "
            "or supplements as intended have been lower "
            "than in your earlier check-ins."
        ),
        "higher": (
            "Your recent responses about taking medicines "
            "or supplements as intended have been higher "
            "than in your earlier check-ins."
        ),
    },

    "physical_recovery": {
        "lower": (
            "Your physical recovery or discomfort has felt "
            "less manageable recently than in your earlier "
            "check-ins."
        ),
        "higher": (
            "Your physical recovery or discomfort has felt "
            "more manageable recently than in your earlier "
            "check-ins."
        ),
    },

    "baby_care": {
        "lower": (
            "Caring for your baby has felt less manageable "
            "recently than in your earlier check-ins."
        ),
        "higher": (
            "Caring for your baby has felt more manageable "
            "recently than in your earlier check-ins."
        ),
    },

    "other_responsibilities": {
        "lower": (
            "Your other responsibilities have felt less "
            "manageable recently than in your earlier "
            "check-ins."
        ),
        "higher": (
            "Your other responsibilities have felt more "
            "manageable recently than in your earlier "
            "check-ins."
        ),
    },

    "personal_care": {
        "lower": (
            "You have reported having less time for your "
            "own basic care recently than in your earlier "
            "check-ins."
        ),
        "higher": (
            "You have reported having more time for your "
            "own basic care recently than in your earlier "
            "check-ins."
        ),
    },
}


# wording used when a response has varied across recent checkins
FIELD_VARIABILITY_TEXT = {
    "sleep": (
        "Your sleep quality responses have varied "
        "across recent check-ins."
    ),

    "mood": (
        "Your mood responses have varied across "
        "recent check-ins."
    ),

    "stress": (
        "Your stress responses have varied across "
        "recent check-ins."
    ),

    "support": (
        "Your responses about feeling supported have "
        "varied across recent check-ins."
    ),

    "food": (
        "Your responses about eating enough for your "
        "own needs have varied across recent check-ins."
    ),

    "medication": (
        "Your responses about taking medicines or "
        "supplements as intended have varied across "
        "recent applicable check-ins."
    ),

    "physical_recovery": (
        "Your responses for physical recovery"
        "has varied across recent check-ins."
    ),

    "baby_care": (
        "Your response for caring for your baby "
        "has varied across recent applicable check-ins."
    ),

    "other_responsibilities": (
        "Your response for how manageable your other responsibilities "
        "have felt has varied across recent applicable check-ins."
    ),

    "personal_care": (
        "The amount of time you have reported for your "
        "own basic care has varied across recent check-ins."
    ),
}


# creates the user-facing sentence for a recent baseline change
def create_change_text(pattern):
    field_name = pattern.get("field", "")
    direction = pattern.get("direction", "")
    field_text = FIELD_CHANGE_TEXT.get(field_name,{},)
    # return the wording that matches both the field and direction
    return field_text.get(direction,"",)


# creates wording for a theme that has appeared repeatedly
def create_recurring_text(pattern):
    theme = str(
        pattern.get("theme","",)
    ).strip()
    theme_type = pattern.get("type")

    if not theme:
        return ""
    # supportive themes use slightly different wording
    if theme_type == "supportive":
        return (
            f"{theme} has appeared repeatedly as a "
            "supportive factor across your recent "
            "check-ins."
        )

    return (
        f"{theme} has appeared repeatedly across "
        "your recent check-ins."
    )


# creates wording for a theme that has only started appearing recently
def create_emerging_text(pattern):
    theme = str(
        pattern.get(
            "theme",
            "",
        )
    ).strip()
    theme_type = pattern.get("type")
    if not theme:
        return ""
    # keep supportive factors clearly separated from strain themes
    if theme_type == "supportive":
        return (
            f"{theme} has started appearing as a "
            "supportive factor in your recent "
            "check-ins."
        )

    return (
        f"{theme} has started appearing in your "
        "recent check-ins."
    )


# returns the prepared wording for a variable structured response
def create_variability_text(pattern):
    field_name = pattern.get("field","",)
    return FIELD_VARIABILITY_TEXT.get(field_name,"",)


# converts one selected pattern into a card shown to the user
def create_pattern_card(relevance_item):
    category = relevance_item.get(
        "category",
        "",
    )
    pattern = relevance_item.get(
        "pattern",
        {},
    )

    # choose the card title and wording based on the relevance category
    if category == "recent_change":
        title = "Recent change"
        text = create_change_text(pattern)

    elif category == "repeated_pattern":
        if pattern.get("type") == "supportive":
            title = "Repeated supportive pattern"
        else:
            title = "Repeated pattern"

        text = create_recurring_text(pattern)

    elif category == "new_pattern":
        if pattern.get("type") == "supportive":
            title = "New supportive pattern"
        else:
            title = "New pattern"

        text = create_emerging_text(pattern)

    elif category == "variable_pattern":
        title = "Variable pattern"
        text = create_variability_text(pattern)

    else:
        return None
    # do not create an empty card when no wording is available
    if not text:
        return None

    return {
        "title": title,
        "text": text,
        "category": category,
    }


# builds the final set of pattern cards shown on the dashboard
def create_pattern_summary(
    all_patterns,
    max_surface_patterns=5,
):
    relevance = select_relevant_patterns(
        all_patterns,
        max_surface_patterns=max_surface_patterns,
    )
    cards = []
    # only surfaced patterns are turned into visible cards
    for relevance_item in relevance["surface_patterns"]:
        card = create_pattern_card(
            relevance_item
        )

        if card is not None:
            cards.append(card)

    # show a neutral message when there are not enough clear patterns yet
    if not cards:
        return {
            "available": False,
            "title": "Patterns Across Your Check-ins",
            "message": (
                "No clear personal patterns are being shown yet. As you complete more approved "
                "check-ins, this section may highlight "
                "repeated themes or changes over time."
            ),
            "cards": [],
            "supporting_patterns": relevance[
                "supporting_patterns"
            ],
        }

    # combine card text into one short overall summary
    summary_text = " ".join(
        card["text"]
        for card in cards
    )

    return {
        "available": True,
        "title": "Patterns Across Your Check-ins",
        "message": summary_text,
        "cards": cards,
        "supporting_patterns": relevance[
            "supporting_patterns"
        ],
    }


# gets the usersaved pattern history and builds the final summary
def get_pattern_summary_for_user(user_id):
    all_patterns = get_all_patterns_for_user(
        user_id
    )
    return create_pattern_summary(
        all_patterns
    )