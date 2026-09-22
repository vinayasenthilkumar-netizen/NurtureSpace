# imports the pattern detection functions used to test the five PPA signals
from patterns.history import (
    detect_baseline_changes,
    detect_recurring_themes,
    detect_emerging_themes,
    detect_wellbeing_variability,
    detect_reflection_checkin_alignment,
    get_structured_answer,
)

# imports the PPR function used to prioritise supported personal patterns
from patterns.relevance import select_relevant_patterns
# imports the helper used to convert pattern results into user facing cards
from patterns.summary import create_pattern_summary

# creates a reusable check in dictionary so tests can easily vary selected values
def checkin(
    sleep=3,
    mood=3,
    stress=3,
    support=3,
    food=3,
    medication=3,
    physical_recovery=3,
    baby_care=3,
    other_responsibilities=3,
    personal_care=3,
    supportive=None,
    strain=None,
    emotions=None,
):
    return {
        "core_answers": {
            "sleep": sleep,
            "mood": mood,
            "stress": stress,
            "support": support,
        },
        "context_answers": {
            "food": food,
            "medication": medication,
            "physical_recovery": physical_recovery,
            "baby_care": baby_care,
            "other_responsibilities": other_responsibilities,
            "personal_care": personal_care,
        },
        "approved_supportive_factors": supportive or [],
        "approved_strain_factors": strain or [],
        "approved_text_emotions": emotions or [],
    }


# finds a pattern for a particular questionnaire field
def find(items, field):
    for item in items:
        if item.get("field") == field:
            return item
    return None

# creates a complete PPA result structure for testing the relevance layer
def results(
    baseline=None,
    recurring=None,
    emerging=None,
    variability=None,
    alignment=None,
):
    return {
        "baseline_changes": {
            "available": True,
            "patterns": baseline or [],
        },
        "recurring_themes": {
            "available": True,
            "patterns": recurring or [],
        },
        "emerging_themes": {
            "available": True,
            "patterns": emerging or [],
        },
        "wellbeing_variability": {
            "available": True,
            "patterns": variability or [],
        },
        "reflection_checkin_alignment": {
            "available": True,
            "patterns": alignment or [],
        },
    }


# checks missing-value handling and whether recent values are compared with the earlier baseline
def test_baseline():
    item = checkin(
        medication=0,
        baby_care=0,
        other_responsibilities=0,
    )

    # zero-valued context answers represent not-applicable responses
    assert get_structured_answer(item, "medication") is None
    assert get_structured_answer(item, "baby_care") is None

    old = [
        checkin(sleep=4, stress=2, physical_recovery=4),
        checkin(sleep=4, stress=2, physical_recovery=4),
        checkin(sleep=4, stress=2, physical_recovery=4),
        checkin(sleep=2, stress=4, physical_recovery=2),
        checkin(sleep=2, stress=4, physical_recovery=2),
        checkin(sleep=1, stress=5, physical_recovery=2),
    ]

    # history is supplied newest first to match the application
    result = detect_baseline_changes(list(reversed(old)))
    assert result["available"] is True
    sleep = find(result["patterns"], "sleep")
    stress = find(result["patterns"], "stress")

    recovery = find(result["patterns"], "physical_recovery")
    assert sleep["direction"] == "lower"
    assert stress["direction"] == "higher"
    assert recovery["direction"] == "lower"
    # baseline analysis should remain unavailable until enough history exists
    short = detect_baseline_changes([checkin() for _ in range(5)])
    assert short["available"] is False
    assert short["patterns"] == []

# checks the thresholds used for recurring and newly emerging themes
def test_themes():
    repeated = detect_recurring_themes([
        checkin(strain=["Sleep difficulty"]),
        checkin(strain=["Sleep difficulty"]),
        checkin(strain=["Sleep difficulty"]),
        checkin(strain=["Feeling overwhelmed"]),
        checkin(),
    ])

    assert len(repeated["patterns"]) == 1
    assert repeated["patterns"][0]["theme"] == "Sleep difficulty"
    assert repeated["patterns"][0]["occurrences"] == 3
    # two occurrences should not meet the recurring-theme threshold
    only_two = detect_recurring_themes([
        checkin(strain=["Sleep difficulty"]),
        checkin(strain=["Sleep difficulty"]),
        checkin(),
        checkin(),
        checkin(),
    ])

    assert only_two["patterns"] == []
    # a theme appearing in the latest two check-ins but not the earlier three is emerging
    emerging = detect_emerging_themes([
        checkin(strain=["Limited practical support"]),
        checkin(strain=["Limited practical support"]),
        checkin(),
        checkin(),
        checkin(),
    ])

    assert len(emerging["patterns"]) == 1
    assert emerging["patterns"][0]["theme"] == "Limited practical support"
    assert emerging["patterns"][0]["recent_occurrences"] == 2

# checks variability detection and agreement between reflection and questionnaire evidence
def test_signals():
    history = [
        checkin(baby_care=2),
        checkin(baby_care=5),
        checkin(baby_care=2),
        checkin(baby_care=4),
        checkin(baby_care=1),
    ]

    variable = detect_wellbeing_variability(history)
    baby = find(variable["patterns"], "baby_care")
    assert baby is not None
    assert baby["lowest_value"] == 1
    assert baby["highest_value"] == 5
    assert baby["range"] == 4
    # stable values should not create a variability pattern
    stable = detect_wellbeing_variability([
        checkin(personal_care=3) for _ in range(5)
    ])
    assert find(stable["patterns"], "personal_care") is None
    # low sleep responses and matching reflection themes should produce alignment evidence
    aligned = detect_reflection_checkin_alignment([
        checkin(
            sleep=1,
            strain=["Sleep difficulty"],
        ),
        checkin(
            sleep=2,
            strain=["Sleep difficulty"],
        ),
        checkin(sleep=4),
    ])
    sleep = find(aligned["patterns"], "sleep")
    assert sleep is not None
    assert sleep["direction"] == "strain"
    assert sleep["occurrences"] == 2

# checks how PPR selects main patterns and keeps supporting evidence separate
def test_relevance():
    data = results(
        baseline=[
            {
                "signal": "baseline_change",
                "field": "sleep",
                "direction": "lower",
            },
            {
                "signal": "baseline_change",
                "field": "mood",
                "direction": "lower",
            },
            {
                "signal": "baseline_change",
                "field": "stress",
                "direction": "higher",
            },
            {
                "signal": "baseline_change",
                "field": "support",
                "direction": "lower",
            },
        ],
        recurring=[
            {
                "signal": "recurring_theme",
                "theme": "Sleep difficulty",
                "type": "strain",
                "occurrences": 4,
            },
            {
                "signal": "recurring_theme",
                "theme": "Feeling overwhelmed",
                "type": "strain",
                "occurrences": 3,
            },
        ],
        variability=[
            {
                "signal": "wellbeing_variability",
                "field": "sleep",
                "range": 3,
            }
        ],
        alignment=[
            {
                "signal": "reflection_checkin_alignment",
                "field": "sleep",
                "direction": "strain",
                "occurrences": 2,
            }
        ],
    )

    # only the configured maximum number of main patterns should be surfaced
    selected = select_relevant_patterns(data, max_surface_patterns=5)
    assert selected["surface_count"] == 5
    categories = [
        item["category"] for item in selected["surface_patterns"]
    ]
    assert "recent_change" in categories
    assert "repeated_pattern" in categories
    support = [
        item["category"] for item in selected["supporting_patterns"]
    ]
    # variability for the same field as a baseline change remains supporting evidence
    assert "variable_pattern" in support
    # reflection-questionnaire alignment is also supporting rather than a main card
    assert "reflection_alignment" in support

# checks that detected patterns are converted into suitable user-facing summary cards
def test_pattern_summary():
    data = results(
        baseline=[
            {
                "signal": "baseline_change",
                "field": "sleep",
                "direction": "lower",
            }
        ],
        recurring=[
            {
                "signal": "recurring_theme",
                "theme": "Sleep difficulty",
                "type": "strain",
                "occurrences": 4,
            }
        ],
        emerging=[
            {
                "signal": "emerging_theme",
                "theme": "Limited practical support",
                "type": "strain",
            }
        ],
        variability=[
            {
                "signal": "wellbeing_variability",
                "field": "personal_care",
                "range": 3,
            }
        ],
        alignment=[
            {
                "signal": "reflection_checkin_alignment",
                "field": "sleep",
                "direction": "strain",
                "occurrences": 2,
            }
        ],
    )

    summary = create_pattern_summary(data)
    assert summary["available"] is True
    titles = [card["title"] for card in summary["cards"]]
    assert "Recent change" in titles
    assert "Repeated pattern" in titles
    assert "New pattern" in titles
    assert "Variable pattern" in titles

    text = " ".join(card["text"] for card in summary["cards"])
    assert "sleep" in text.lower()
    # alignment can support another result but should not be presented as its own card
    assert all("alignment" not in title.lower() for title in titles)
    assert len(summary["supporting_patterns"]) >= 1