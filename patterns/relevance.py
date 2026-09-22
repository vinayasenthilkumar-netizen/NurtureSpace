
# personal pattern relavance
# controls how many patterns are shown directly to the user
MAX_SURFACE_PATTERNS = 5
# limits the number of variability patterns shown at the same time
MAX_VARIABILITY_PATTERNS = 2


# safely gets the pattern list for one personal pattern signal
def get_patterns(all_patterns, signal_name):
    signal_result = all_patterns.get(signal_name, {})
    return signal_result.get("patterns", [])


# adds the category and explanation used by the relevance layer
def make_relevance_item(category, pattern, reason=""):
    return {
        "category": category,
        "pattern": pattern,
        "reason": reason,
    }


# decides which detected patterns are shown and which stay as supporting context
def select_relevant_patterns(
    all_patterns,
    max_surface_patterns=MAX_SURFACE_PATTERNS,
):
# select patterns to show and keep as supporting context.
    surface_candidates = []
    supporting_patterns = []

    # recurring approved themes can be shown directly
    recurring_patterns = get_patterns(all_patterns, "recurring_themes")

    for pattern in recurring_patterns:
        surface_candidates.append(
            make_relevance_item(
                category="repeated_pattern",
                pattern=pattern,
                reason=(
                    "The approved theme has appeared "
                    "repeatedly across recent check-ins."
                ),
            )
        )

    # recently emerging themes can also be surfaced
    emerging_patterns = get_patterns(all_patterns, "emerging_themes")

    for pattern in emerging_patterns:
        surface_candidates.append(
            make_relevance_item(
                category="new_pattern",
                pattern=pattern,
                reason=(
                    "The approved theme has started "
                    "appearing in recent check-ins."
                ),
            )
        )

    # baseline changes are tracked by field so duplicate variability is avoided
    baseline_patterns = get_patterns(all_patterns, "baseline_changes")
    baseline_fields = set()

    for pattern in baseline_patterns:
        field_name = pattern.get("field")

        if field_name:
            baseline_fields.add(field_name)
        surface_candidates.append(
            make_relevance_item(
                category="recent_change",
                pattern=pattern,
                reason=(
                    "Recent responses differ from "
                    "the user's earlier personal pattern."
                ),
            )
        )

    # variability patterns are ordered by the size of their range
    variability_patterns = get_patterns(
        all_patterns,
        "wellbeing_variability",
    )
    variability_patterns = sorted(
        variability_patterns,
        key=lambda pattern: pattern.get("range", 0),
        reverse=True,
    )
    surfaced_variability_count = 0

    for pattern in variability_patterns:
        field_name = pattern.get("field")

        # keep variability as supporting context when the field already changed from baseline
        if field_name in baseline_fields:
            supporting_patterns.append(
                make_relevance_item(
                    category="variable_pattern",
                    pattern=pattern,
                    reason=(
                        "Kept as supporting context because "
                        "this field already has a surfaced "
                        "baseline change."
                    ),
                )
            )
            continue

        # only the strongest variability patterns are shown directly
        if surfaced_variability_count >= MAX_VARIABILITY_PATTERNS:
            supporting_patterns.append(
                make_relevance_item(
                    category="variable_pattern",
                    pattern=pattern,
                    reason=(
                        "Kept as supporting context to avoid "
                        "showing too many similar variability "
                        "patterns."
                    ),
                )
            )
            continue
        surface_candidates.append(
            make_relevance_item(
                category="variable_pattern",
                pattern=pattern,
                reason=(
                    "Responses have varied noticeably "
                    "across recent check-ins."
                ),
            )
        )
        surfaced_variability_count += 1

    # reflection-check-in alignment is kept as supporting evidence
    alignment_patterns = get_patterns(
        all_patterns,
        "reflection_checkin_alignment",
    )
    for pattern in alignment_patterns:
        supporting_patterns.append(
            make_relevance_item(
                category="reflection_alignment",
                pattern=pattern,
                reason=(
                    "Used as supporting evidence when "
                    "explaining relevant personal patterns."
                ),
            )
        )

    # apply the final limit for patterns shown directly to the user
    surface_patterns = surface_candidates[:max_surface_patterns]
    overflow_patterns = surface_candidates[max_surface_patterns:]

    # anything beyond the display limit is still kept as supporting context
    for item in overflow_patterns:
        supporting_patterns.append(
            make_relevance_item(
                category=item["category"],
                pattern=item["pattern"],
                reason=(
                    "Kept as supporting context because "
                    "the user-facing pattern display "
                    "has reached its limit."
                ),
            )
        )
    return {
        "surface_patterns": surface_patterns,
        "supporting_patterns": supporting_patterns,
        "surface_count": len(surface_patterns),
        "supporting_count": len(supporting_patterns),
    }