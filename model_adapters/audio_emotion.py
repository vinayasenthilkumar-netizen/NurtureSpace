
# caches the audio model so it only needs to be loaded once
from functools import lru_cache
# used to check the prepared audio file before model inference
from pathlib import Path
# hugging face pipeline is used to run the selected audio model
from transformers import pipeline
# imports the configured audio-emotion model
from config.settings import SELECTED_AUDIO_EMOTION_MODEL
# imports the shared audio observation labels used by the application
from core.constants import (
    AUDIO_OBSERVATION_CALM,
    AUDIO_OBSERVATION_DISGUST,
    AUDIO_OBSERVATION_ENERGETIC,
    AUDIO_OBSERVATION_FRUSTRATED,
    AUDIO_OBSERVATION_LOW_ENERGY,
    AUDIO_OBSERVATION_UNCLEAR,
    AUDIO_OBSERVATION_WORRIED,
)


# converts the model's numbered labels into readable emotion names
RAW_AUDIO_LABEL_MAP = {
    "LABEL_0": "sadness",
    "LABEL_1": "angry",
    "LABEL_2": "disgust",
    "LABEL_3": "fear",
    "LABEL_4": "happy",
    "LABEL_5": "neutral",
}


# maps raw emotion labels into the cautious observation wording used by the app
USER_FACING_AUDIO_MAP = {
    "sadness": AUDIO_OBSERVATION_LOW_ENERGY,
    "sad": AUDIO_OBSERVATION_LOW_ENERGY,

    "angry": AUDIO_OBSERVATION_FRUSTRATED,
    "anger": AUDIO_OBSERVATION_FRUSTRATED,

    "disgust": AUDIO_OBSERVATION_DISGUST,
    "disgusted": AUDIO_OBSERVATION_DISGUST,

    "fear": AUDIO_OBSERVATION_WORRIED,
    "fearful": AUDIO_OBSERVATION_WORRIED,

    "happy": AUDIO_OBSERVATION_ENERGETIC,
    "happiness": AUDIO_OBSERVATION_ENERGETIC,
    "joy": AUDIO_OBSERVATION_ENERGETIC,

    "neutral": AUDIO_OBSERVATION_CALM,
}


# fixed observation order used when building the full score distribution
AUDIO_SCORING_OBSERVATIONS = [
    AUDIO_OBSERVATION_LOW_ENERGY,
    AUDIO_OBSERVATION_FRUSTRATED,
    AUDIO_OBSERVATION_DISGUST,
    AUDIO_OBSERVATION_WORRIED,
    AUDIO_OBSERVATION_ENERGETIC,
    AUDIO_OBSERVATION_CALM,
]


# only the dominant observation needs to pass this threshold for display
MIN_AUDIO_OBSERVATION_SCORE = 0.40


# loads the selected audio-classification model once and reuses it
@lru_cache(maxsize=1)
def load_audio_emotion_model():
    return pipeline(
        task="audio-classification",
        model=SELECTED_AUDIO_EMOTION_MODEL,
    )


# creates a consistent fallback when no clear vocal pattern is available
def create_unclear_result(
    error_message="",
    all_scores=None,
):
    return {
        "observation": AUDIO_OBSERVATION_UNCLEAR,
        "raw_label": "",
        "score": 0.0,
        "clear_result": False,
        "all_scores": list(all_scores or []),
        "error": str(error_message or ""),
    }


# converts different pipeline output shapes into one sorted list
def normalise_audio_output(raw_output):
    # a single prediction is converted into a one-item list
    if isinstance(raw_output, dict):
        results = [raw_output]

    # some pipeline versions return either a list or a nested list
    elif isinstance(raw_output, list):
        if raw_output and isinstance(raw_output[0], list):
            results = raw_output[0]
        else:
            results = raw_output

    else:
        return []

    cleaned_results = []

    # keep only dictionary predictions with a label and numerical score
    for item in results:
        if not isinstance(item, dict):
            continue

        cleaned_results.append({
            "label": str(
                item.get("label", "")
            ).strip(),
            "score": float(
                item.get("score", 0)
            ),
        })

    # highest-confidence prediction is placed first
    cleaned_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return cleaned_results


# converts labels such as LABEL_0 into readable lower-case emotion names
def normalise_raw_audio_label(raw_label):
    clean_label = str(raw_label or "").strip()

    return RAW_AUDIO_LABEL_MAP.get(
        clean_label.upper(),
        clean_label.lower(),
    )


# converts one raw audio label into the application's observation wording
def map_audio_observation(raw_label):
    normalised_label = normalise_raw_audio_label(raw_label)

    # unknown labels are treated as unclear rather than forced into a category
    return USER_FACING_AUDIO_MAP.get(
        normalised_label,
        AUDIO_OBSERVATION_UNCLEAR,
    )


# combines all model predictions into the observation scores used internally
def build_all_audio_scores(model_results):
    # start every known observation at zero
    combined_scores = {
        observation: 0.0
        for observation in AUDIO_SCORING_OBSERVATIONS
    }
    # several raw labels may contribute to the same application observation
    for item in model_results:
        raw_label = normalise_raw_audio_label(
            item.get("label", "")
        )
        observation = map_audio_observation(
            raw_label
        )
        if observation not in combined_scores:
            continue

        try:
            score = float(
                item.get("score", 0)
            )
        except (TypeError, ValueError):
            continue
        # add the confidence to its mapped observation category
        combined_scores[observation] += score
    all_scores = []

    # keep every observation so reflection scoring receives the full distribution
    for observation in AUDIO_SCORING_OBSERVATIONS:
        all_scores.append({
            "observation": observation,
            "score": round(
                combined_scores[observation],
                4,
            ),
        })

    return all_scores


# analyses one prepared wav file and returns both display and scoring information
def predict_audio_observation(audio_path):
    path = Path(str(audio_path or ""))

    # model inference cannot continue if the prepared audio is missing
    if not audio_path or not path.exists():
        return create_unclear_result(
            "The prepared audio file was not found."
        )

    try:
        # use the cached audio-classification model
        model = load_audio_emotion_model()

        # request scores for all available model labels
        raw_output = model(
            str(path),
            top_k=None,
        )

        model_results = normalise_audio_output(
            raw_output
        )

        if not model_results:
            return create_unclear_result(
                "The audio model returned no predictions."
            )

        # retain all mapped scores for internal reflection scoring
        all_scores = build_all_audio_scores(
            model_results
        )

        # the highest-ranked raw result is used for the display observation
        primary_result = model_results[0]

        raw_label = normalise_raw_audio_label(
            primary_result["label"]
        )

        score = round(
            primary_result["score"],
            4,
        )

        observation = map_audio_observation(
            raw_label
        )

        # a display observation needs enough confidence and a known mapping
        clear_result = (
            score >= MIN_AUDIO_OBSERVATION_SCORE
            and observation != AUDIO_OBSERVATION_UNCLEAR
        )

        # weak or unmapped predictions are shown as unclear
        if not clear_result:
            observation = AUDIO_OBSERVATION_UNCLEAR

        return {
            "observation": observation,
            "raw_label": raw_label,
            "score": score,
            "clear_result": clear_result,
            "all_scores": all_scores,
            "error": "",
        }

    # any unexpected model error is returned through the normal unclear result
    except Exception as error:
        return create_unclear_result(
            f"Audio analysis failed: {error}"
        )