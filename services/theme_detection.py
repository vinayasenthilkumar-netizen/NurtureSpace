
# json is used to load the practical-theme rules file
import json
# regular expressions are used for safe phrase matching
import re
# caches the rules so the file only needs to be loaded once
from functools import lru_cache
# imports the configured path to the theme rules file
from config.settings import THEME_RULES_PATH


# loads and checks the practical-theme rules used by the detector
@lru_cache(maxsize=1)
def load_theme_rules():
    # Load and cache practical-theme rules

    if not THEME_RULES_PATH.exists():
        raise FileNotFoundError(
            f"Theme rules were not found at: {THEME_RULES_PATH}"
        )

    with THEME_RULES_PATH.open("r", encoding="utf-8") as rules_file:
        rules = json.load(rules_file)

    # both rule groups must be present before detection can run
    if "strain_factors" not in rules:
        raise ValueError(
            "The theme rules file is missing strain_factors."
        )

    if "supportive_factors" not in rules:
        raise ValueError(
            "The theme rules file is missing supportive_factors."
        )

    return rules

# Text preparation
# cleans reflection text before checking it against phrase rules
def normalise_text(text):

    #prerepare text for phrase matching
    clean_text = str(text or "").lower()

    # keep apostrophes and hyphens because some phrases depend on them
    clean_text = re.sub(r"[^a-z0-9'\-\s]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    return clean_text.strip()


# phrase matching
# checks if a complete rule phrase appears in the reflection text
def text_contains_phrase(clean_text, phrase):
    #check whether a complete phrase appears in the text

    clean_text = normalise_text(clean_text)
    clean_phrase = normalise_text(phrase)

    if not clean_text or not clean_phrase:
        return False

    # boundaries stop a phrase from matching inside a larger word
    pattern = (
        rf"(?<![a-z0-9])"
        rf"{re.escape(clean_phrase)}"
        rf"(?![a-z0-9])"
    )

    return re.search(pattern, clean_text) is not None


# checks whether an exclusion phrase prevents a rule from being used
def rule_has_excluded_phrase(clean_text, rule):
    #see whether a rule should be excluded

    excluded_phrases = rule.get("exclude_phrases", [])

    for phrase in excluded_phrases:
        if text_contains_phrase(clean_text, phrase):
            return True

    return False


# returns the first phrase from a rule that appears in the text
def find_matching_phrase(clean_text, rule):
    #return the first phrase that matches a rule
    for phrase in rule.get("phrases", []):
        if text_contains_phrase(clean_text, phrase):
            return phrase
    return None

# rule detection
# applies one group of theme rules and returns the matching factors
def detect_rule_group(clean_text, rules):
    #detect matching themes from one group of rules
    detected_items = []
    for rule in rules:
        # excluded phrases are checked before accepting a match
        if rule_has_excluded_phrase(clean_text, rule):
            continue
        matching_phrase = find_matching_phrase(clean_text, rule)
        if matching_phrase is None:
            continue
        detected_items.append({
            "name": rule["name"],
            "matched_phrase": matching_phrase,
        })

    return detected_items


# practical themes
# detects supportive and strain factors suggested by the reflection text
def detect_practical_themes(text):
    # detect possible supportive factors and areas of strain.only returns suggestions. doesnt approve or save them.
    clean_text = normalise_text(text)
    # empty reflection text has no practical themes to detect
    if not clean_text:
        return {
            "supportive_factors": [],
            "strain_factors": [],
            "theme_found": False,
            "no_significant_theme_detected": True,
        }

    rules = load_theme_rules()
    # supportive and strain rules are checked separately
    supportive_factors = detect_rule_group(
        clean_text,
        rules["supportive_factors"],
    )
    strain_factors = detect_rule_group(
        clean_text,
        rules["strain_factors"],
    )
    theme_found = bool(
        supportive_factors or strain_factors
    )
    return {
        "supportive_factors": supportive_factors,
        "strain_factors": strain_factors,
        "theme_found": theme_found,
        "no_significant_theme_detected": not theme_found,
    }