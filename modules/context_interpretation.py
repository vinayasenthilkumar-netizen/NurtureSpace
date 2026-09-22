
# six context question keys used by the checkin
from core.constants import CONTEXT_QUESTION_KEYS

#  interpretation rules for each possible context response
CONTEXT_RULES = {
    "food": {
        1: {
            "level": "difficulty",
            "summary": "You were not able to eat enough for your own needs today.",
            "appointment": "I would like to discuss difficulties eating enough for my own needs.",
        },
        2: {
            "level": "difficulty",
            "summary": "You were able to eat very little for your own needs today.",
            "appointment": "I would like to discuss difficulties eating enough for my own needs.",
        },
        3: {
            "level": "mixed",
            "summary": "You were partly able to meet your food needs today.",
            "appointment": "",
        },
        4: {
            "level": "positive",
            "summary": "You were mostly able to eat enough for your own needs today.",
            "appointment": "",
        },
        5: {
            "level": "positive",
            "summary": "You were comfortably able to eat enough for your own needs today.",
            "appointment": "",
        },
    },

    "medication": {
        0: {
            "level": "not_applicable",
            "summary": "",
            "appointment": "",
        },
        1: {
            "level": "difficulty",
            "summary": "You were not able to follow your intended medicine or supplement routine today.",
            "appointment": "I found it difficult to follow my intended medicine or supplement routine.",
        },
        2: {
            "level": "difficulty",
            "summary": "You were able to follow only a small part of your intended medicine or supplement routine today.",
            "appointment": "I found it difficult to follow my intended medicine or supplement routine.",
        },
        3: {
            "level": "mixed",
            "summary": "You were able to follow some of your intended medicine or supplement routine today.",
            "appointment": "",
        },
        4: {
            "level": "positive",
            "summary": "You were able to follow most of your intended medicine or supplement routine today.",
            "appointment": "",
        },
        5: {
            "level": "positive",
            "summary": "You were able to follow your intended medicine or supplement routine today.",
            "appointment": "",
        },
    },

    "physical_recovery": {
        1: {
            "level": "difficulty",
            "summary": "Physical recovery or discomfort felt very difficult to manage today.",
            "appointment": "I would like to discuss my physical recovery or discomfort.",
        },
        2: {
            "level": "difficulty",
            "summary": "Physical recovery or discomfort felt difficult to manage today.",
            "appointment": "I would like to discuss my physical recovery or discomfort.",
        },
        3: {
            "level": "mixed",
            "summary": "Physical recovery or discomfort felt mixed today.",
            "appointment": "",
        },
        4: {
            "level": "positive",
            "summary": "Physical recovery or discomfort felt mostly manageable today.",
            "appointment": "",
        },
        5: {
            "level": "positive",
            "summary": "Physical recovery or discomfort felt very manageable today.",
            "appointment": "",
        },
    },

    "baby_care": {
        0: {
            "level": "not_applicable",
            "summary": "",
            "appointment": "",
        },
        1: {
            "level": "difficulty",
            "summary": "Caring for your baby felt very difficult today.",
            "appointment": "I would like to discuss parts of caring for my baby that have felt difficult.",
        },
        2: {
            "level": "difficulty",
            "summary": "Caring for your baby felt difficult today.",
            "appointment": "I would like to discuss parts of caring for my baby that have felt difficult.",
        },
        3: {
            "level": "mixed",
            "summary": "Caring for your baby felt mixed today.",
            "appointment": "",
        },
        4: {
            "level": "positive",
            "summary": "Caring for your baby felt mostly manageable today.",
            "appointment": "",
        },
        5: {
            "level": "positive",
            "summary": "Caring for your baby felt very manageable today.",
            "appointment": "",
        },
    },

    "other_responsibilities": {
        0: {
            "level": "not_applicable",
            "summary": "",
            "appointment": "",
        },
        1: {
            "level": "difficulty",
            "summary": "Your other responsibilities felt very difficult to manage today.",
            "appointment": "Balancing my other responsibilities has felt difficult.",
        },
        2: {
            "level": "difficulty",
            "summary": "Your other responsibilities felt difficult to manage today.",
            "appointment": "Balancing my other responsibilities has felt difficult.",
        },
        3: {
            "level": "mixed",
            "summary": "Managing your other responsibilities felt mixed today.",
            "appointment": "",
        },
        4: {
            "level": "positive",
            "summary": "Your other responsibilities felt mostly manageable today.",
            "appointment": "",
        },
        5: {
            "level": "positive",
            "summary": "Your other responsibilities felt very manageable today.",
            "appointment": "",
        },
    },

    "personal_care": {
        1: {
            "level": "difficulty",
            "summary": "You had no time for your own basic care today.",
            "appointment": "I have had very little or no time for my own basic care.",
        },
        2: {
            "level": "difficulty",
            "summary": "You had very little time for your own basic care today.",
            "appointment": "I have had very little or no time for my own basic care.",
        },
        3: {
            "level": "mixed",
            "summary": "You had some time for your own basic care today.",
            "appointment": "",
        },
        4: {
            "level": "positive",
            "summary": "You had mostly enough time for your own basic care today.",
            "appointment": "",
        },
        5: {
            "level": "positive",
            "summary": "You had enough time for your own basic care today.",
            "appointment": "",
        },
    },
}


# converts the six context responses into descriptive summary information
def interpret_context_answers(context_answers):
    # convert the context answers to summary and appointment points.
    # keep each type of interpretation separate for later use in the dashboard
    result = {
        "summary_points": [],
        "positive_points": [],
        "mixed_points": [],
        "difficulty_points": [],
        "appointment_suggestions": [],
        "not_applicable": [],
    }
    # invalid context data should not produce any interpretation
    if not isinstance(context_answers, dict):
        return result
    # process only the known context questions
    for question_key in CONTEXT_QUESTION_KEYS:
        answer = context_answers.get(question_key)
        rule = CONTEXT_RULES.get(question_key, {}).get(answer)
        # unanswered or unsupported values are ignored
        if rule is None:
            continue
        level = rule["level"]
        # not-applicable answers are recorded but do not create summary text
        if level == "not_applicable":
            result["not_applicable"].append(question_key)
            continue
        summary_text = rule["summary"]
        appointment_text = rule["appointment"]
        # all valid applicable responses can contribute to the summary
        if summary_text:
            result["summary_points"].append(summary_text)
        # also group the summary by its interpretation level
        if level == "positive":
            result["positive_points"].append(summary_text)
        elif level == "mixed":
            result["mixed_points"].append(summary_text)
        elif level == "difficulty":
            result["difficulty_points"].append(summary_text)
        # only difficulty rules normally provide appointment suggestions
        # duplicates are avoided when two answers use the same wording
        if (
            appointment_text
            and appointment_text not in result["appointment_suggestions"]
        ):
            result["appointment_suggestions"].append(appointment_text)

    return result