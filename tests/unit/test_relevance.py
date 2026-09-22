#selects the most relevant personal patterns to surface
from patterns.relevance import select_relevant_patterns

# creates a basic empty pattern-analysis structure that tests can modify
def base(**changes):
    # start with all five pattern categories present and empty
    data = {
        "baseline_changes": {"patterns": []},
        "recurring_themes": {"patterns": []},
        "emerging_themes": {"patterns": []},
        "wellbeing_variability": {"patterns": []},
        "reflection_checkin_alignment": {"patterns": []},
    }

    # replace any default sections supplied by the individual test
    data.update(changes)
    # return the completed test input
    return data


# checks that only the two strongest variability patterns are surfaced
def test_variability_limit():
    # create three variability patterns with different ranges
    data = base(wellbeing_variability={"patterns": [
        {"field": "support", "range": 2},
        {"field": "sleep", "range": 4},
        {"field": "mood", "range": 3},
    ]})

    # run relevance selection with space for up to five surfaced patterns
    result = select_relevant_patterns(data, max_surface_patterns=5)
    # collect the variability fields selected for the main surfaced patterns
    shown = [x["pattern"]["field"] for x in result["surface_patterns"] if x["category"] == "variable_pattern"]
    # collect variability fields kept only as supporting patterns
    hidden = [x["pattern"]["field"] for x in result["supporting_patterns"] if x["category"] == "variable_pattern"]
    # the two largest ranges should be surfaced in descending strength
    assert shown == ["sleep", "mood"]
    # the weaker variability pattern should remain available as supporting evidence
    assert hidden == ["support"]


# checks how emerging themes and reflection-check-in alignment are prioritised
def test_emerging_and_alignment():
    # create one emerging theme and one reflection-check-in alignment pattern
    data = base(
        emerging_themes={"patterns": [{"theme": "Limited practical support", "recent_occurrences": 2}]},
        reflection_checkin_alignment={"patterns": [{"field": "sleep", "direction": "strain", "occurrences": 2}]},
    )

    # run the normal relevance selection rules
    result = select_relevant_patterns(data)
    # the emerging theme should be surfaced as a new pattern
    assert [x["category"] for x in result["surface_patterns"]] == ["new_pattern"]
    # alignment evidence should remain in the supporting pattern section
    assert [x["category"] for x in result["supporting_patterns"]] == ["reflection_alignment"]