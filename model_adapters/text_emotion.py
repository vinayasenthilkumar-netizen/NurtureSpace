
# caches the model so it is loaded only once during the application session
from functools import lru_cache

# hugging face pipeline is used to run the selected text-emotion model
from transformers import pipeline

# imports the configured model name
from config.settings import SELECTED_TEXT_EMOTION_MODEL

# imports the six shared emotion categories used across the application
from core.constants import (
    TEXT_EMOTION_CONNECTION,
    TEXT_EMOTION_FRUSTRATION,
    TEXT_EMOTION_LOW_MOOD,
    TEXT_EMOTION_NEUTRAL,
    TEXT_EMOTION_POSITIVE,
    TEXT_EMOTION_WORRY,
)


# maps raw model labels into the smaller set of categories used by the app
TEXT_LABEL_MAP = {
    # positive
    "joy": TEXT_EMOTION_POSITIVE,
    "happiness": TEXT_EMOTION_POSITIVE,
    "happy": TEXT_EMOTION_POSITIVE,
    "optimism": TEXT_EMOTION_POSITIVE,
    "relief": TEXT_EMOTION_POSITIVE,
    "amusement": TEXT_EMOTION_POSITIVE,
    "excitement": TEXT_EMOTION_POSITIVE,
    "calm": TEXT_EMOTION_POSITIVE,

    # warmth / connection
    "love": TEXT_EMOTION_CONNECTION,
    "caring": TEXT_EMOTION_CONNECTION,
    "gratitude": TEXT_EMOTION_CONNECTION,
    "admiration": TEXT_EMOTION_CONNECTION,

    # low mood
    "sadness": TEXT_EMOTION_LOW_MOOD,
    "sad": TEXT_EMOTION_LOW_MOOD,
    "grief": TEXT_EMOTION_LOW_MOOD,
    "disappointment": TEXT_EMOTION_LOW_MOOD,
    "remorse": TEXT_EMOTION_LOW_MOOD,

    # worry
    "fear": TEXT_EMOTION_WORRY,
    "fearful": TEXT_EMOTION_WORRY,
    "nervousness": TEXT_EMOTION_WORRY,
    "worry": TEXT_EMOTION_WORRY,
    "anxiety": TEXT_EMOTION_WORRY,

    # frustration
    "anger": TEXT_EMOTION_FRUSTRATION,
    "angry": TEXT_EMOTION_FRUSTRATION,
    "annoyance": TEXT_EMOTION_FRUSTRATION,
    "disapproval": TEXT_EMOTION_FRUSTRATION,
    "frustration": TEXT_EMOTION_FRUSTRATION,

    # neutral or less directional labels
    "neutral": TEXT_EMOTION_NEUTRAL,
    "surprise": TEXT_EMOTION_NEUTRAL,
    "confusion": TEXT_EMOTION_NEUTRAL,
    "realization": TEXT_EMOTION_NEUTRAL,
    "curiosity": TEXT_EMOTION_NEUTRAL,
}


# fixed category order used when building the full score distribution
TEXT_EMOTION_CATEGORIES = [
    TEXT_EMOTION_POSITIVE,
    TEXT_EMOTION_CONNECTION,
    TEXT_EMOTION_NEUTRAL,
    TEXT_EMOTION_LOW_MOOD,
    TEXT_EMOTION_WORRY,
    TEXT_EMOTION_FRUSTRATION,
]


# minimum category confidence before it can be suggested to the user
MIN_EMOTION_SUGGESTION_SCORE = 0.10


# loads the selected text-emotion model once and reuses it
@lru_cache(maxsize=1)
def load_text_emotion_model():
    return pipeline(
        task="text-classification",
        model=SELECTED_TEXT_EMOTION_MODEL,
        top_k=None,
    )


# converts different hugging face output shapes into one consistent list
def normalise_model_output(raw_output):
    # a single dictionary is treated as one prediction
    if isinstance(raw_output, dict):
        return [raw_output]

    if not isinstance(raw_output, list):
        return []

    # some pipeline versions return predictions inside another list
    if raw_output and isinstance(raw_output[0], list):
        return raw_output[0]

    return raw_output


# converts one raw model label into one of the application's six categories
def map_text_label(raw_label):
    clean_label = str(raw_label).lower().strip()

    # unknown labels fall back to neutral rather than creating a new category
    return TEXT_LABEL_MAP.get(
        clean_label,
        TEXT_EMOTION_NEUTRAL,
    )


# analyses reflection text and returns both full scores and display suggestions
def analyse_text_emotions(
    text,
    maximum_suggestions=3,
):

    # blank reflection text should not be sent to the model
    if not text or not text.strip():
        return {
            "all_scores": [],
            "suggestions": [],
        }

    model = load_text_emotion_model()

    # request scores for all labels from the selected model
    raw_output = model(
        text.strip()
    )

    # convert the model output into a predictable list structure
    model_results = normalise_model_output(
        raw_output
    )

    # start every shared category at zero before adding model evidence
    combined_scores = {
        category: 0.0
        for category in TEXT_EMOTION_CATEGORIES
    }

    # several raw labels may map into the same application category
    for item in model_results:
        if not isinstance(item, dict):
            continue

        raw_label = item.get(
            "label",
            "neutral",
        )

        try:
            score = float(
                item.get("score", 0)
            )
        except (TypeError, ValueError):
            continue

        shared_category = map_text_label(
            raw_label
        )

        # add together confidence from labels that share the same category
        combined_scores[shared_category] += score

    all_scores = []

    # keep every shared category so reflection scoring has the full distribution
    for category in TEXT_EMOTION_CATEGORIES:
        all_scores.append({
            "category": category,
            "score": round(
                combined_scores[category],
                4,
            ),
        })

    # highest-confidence categories are considered first for display
    ranked_scores = sorted(
        all_scores,
        key=lambda item: item["score"],
        reverse=True,
    )

    # neutral is not shown when clearer emotional categories are available
    clear_suggestions = [
        suggestion
        for suggestion in ranked_scores
        if (
            suggestion["category"] != TEXT_EMOTION_NEUTRAL
            and suggestion["score"] >= MIN_EMOTION_SUGGESTION_SCORE
        )
    ]

    if clear_suggestions:
        # show only the strongest few categories to avoid overloading the user
        suggestions = clear_suggestions[
            :maximum_suggestions
        ]

    else:
        suggestions = []

        # if nothing else is clear enough, neutral can be shown on its own
        for suggestion in ranked_scores:
            if (
                suggestion["category"]
                == TEXT_EMOTION_NEUTRAL
            ):
                suggestions = [
                    suggestion
                ]
                break

    return {
        "all_scores": all_scores,
        "suggestions": suggestions,
    }


# simple wrapper used when only user-facing suggestions are needed
def predict_text_emotions(
    text,
    maximum_suggestions=3,
):
    analysis = analyse_text_emotions(
        text=text,
        maximum_suggestions=maximum_suggestions,
    )

    return analysis["suggestions"]