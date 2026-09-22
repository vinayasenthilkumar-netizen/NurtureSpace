
# common six-class space used to compare models that originally use
# different emotion vocabularies
BENCHMARK_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]


# simpler categories used by the application after model outputs are mapped
APPLICATION_LABELS = [
    "Low Mood",
    "Positive",
    "Warmth",
    "Frustration",
    "Worry",
    "Neutral",
]


# final conversion from benchmark labels into application-level categories
# unsupported labels are treated as Neutral at the application stage
BENCHMARK_TO_APPLICATION = {
    "sadness": "Low Mood",
    "joy": "Positive",
    "love": "Warmth",
    "anger": "Frustration",
    "fear": "Worry",
    "surprise": "Neutral",
    "other": "Neutral",
}

# raw model mapping
# candidate models use different names for similar emotions
# these are grouped into one shared benchmark space before comparison
RAW_LABEL_TO_BENCHMARK = {

    # low mood-related labels
    "sadness": "sadness",
    "sad": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "remorse": "sadness",
    "embarrassment": "sadness",
    "pessimism": "sadness",

    # positive or encouraging emotions
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "approval": "joy",
    "admiration": "joy",
    "positive": "joy",

    # warmth, support and connection-related labels
    "love": "love",
    "caring": "love",
    "care": "love",
    "desire": "love",
    "trust": "love",
    "connection": "love",
    "warmth": "love",

    # anger and frustration-related labels
    "anger": "anger",
    "angry": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    "frustration": "anger",
    "frustrated": "anger",

    # fear, anxiety and worry-related labels
    "fear": "fear",
    "fearful": "fear",
    "nervousness": "fear",
    "worry": "fear",
    "worried": "fear",
    "anxiety": "fear",
    "anxious": "fear",

    # neutral, uncertain or unclear labels are grouped here
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    "realization": "surprise",
    "anticipation": "surprise",
    "unclear": "surprise",
}


# clean one raw label
# standardise model label text before looking it up in the mapping table
def clean_raw_label(raw_label):
    # different models may use capitals, hyphens or underscores
    # so these are normalised before matching
    return (
        str(raw_label or "")
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def map_raw_label_to_benchmark(raw_label):
    clean_label = clean_raw_label(
        raw_label
    )

    # unknown labels stay visible as other instead of being silently discarded
    return RAW_LABEL_TO_BENCHMARK.get(
        clean_label,
        "other",
    )


# normalise the hugging face output
# Hugging Face pipelines can return slightly different list/dictionary shapes
# depending on model and pipeline settings
def normalise_pipeline_output(raw_output):
    # a single prediction dictionary is wrapped in a list
    if isinstance(raw_output, dict):
        return [raw_output]

    if not isinstance(raw_output, list):
        return []

    # some pipelines return [[{label, score}, ...]]
    if raw_output and isinstance(
        raw_output[0],
        list,
    ):
        return raw_output[0]

    # others already return [{label, score}, ...]
    if raw_output and isinstance(
        raw_output[0],
        dict,
    ):
        return raw_output

    return []


# combine scores from several original model labels when they map
# into the same benchmark emotion
def aggregate_benchmark_scores(raw_output):
    # start every benchmark category at zero
    combined_scores = {
        label: 0.0
        for label in BENCHMARK_LABELS
    }

    # other collects labels that are not supported by the mapping
    combined_scores["other"] = 0.0

    model_results = normalise_pipeline_output(
        raw_output
    )

    for result in model_results:
        raw_label = result.get(
            "label",
            "",
        )

        # score values should be numeric, but invalid values safely become zero
        try:
            score = float(
                result.get(
                    "score",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            score = 0.0

        # map the model's original label into the shared benchmark label
        mapped_label = map_raw_label_to_benchmark(
            raw_label
        )

        # several raw labels can therefore add to the same benchmark score
        combined_scores[
            mapped_label
        ] += score

    return combined_scores

# sort the combined benchmark scores from strongest to weakest
def rank_benchmark_predictions(raw_output):
    combined_scores = aggregate_benchmark_scores(
        raw_output
    )

    return sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


# return only the highest-ranked benchmark category while keeping
# all scores available for later inspection
def get_top_benchmark_prediction(raw_output):
    ranked_predictions = rank_benchmark_predictions(
        raw_output
    )

    # safe fallback in case the model returns no usable output
    if not ranked_predictions:
        return {
            "label": "other",
            "score": 0.0,
            "all_scores": {},
        }

    predicted_label, score = ranked_predictions[0]

    return {
        "label": predicted_label,
        "score": score,
        "all_scores": dict(
            ranked_predictions
        ),
    }


# move from the shared benchmark space into the application's
# six user-facing interpretation categories
def aggregate_application_scores(raw_output):
    # first use the same benchmark aggregation used during model evaluation
    benchmark_scores = aggregate_benchmark_scores(
        raw_output
    )

    application_scores = {
        label: 0.0
        for label in APPLICATION_LABELS
    }

    # each benchmark score is transferred into its application category
    for benchmark_label, score in benchmark_scores.items():
        application_label = benchmark_to_application_label(
            benchmark_label
        )

        application_scores[
            application_label
        ] += score

    return application_scores


# rank the application categories after all mapped probabilities are combined
def rank_application_predictions(raw_output):
    application_scores = aggregate_application_scores(
        raw_output
    )

    return sorted(
        application_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


# return only the strongest application category and keep all scores as context
def get_top_application_prediction(raw_output):

    ranked_predictions = rank_application_predictions(
        raw_output
    )

    # Neutral is the safe fallback if no usable predictions are available
    if not ranked_predictions:
        return {
            "label": "Neutral",
            "score": 0.0,
            "all_scores": {},
        }

    predicted_label, score = ranked_predictions[0]

    return {
        "label": predicted_label,
        "score": score,
        "all_scores": dict(
            ranked_predictions
        ),
    }


# small helper used when converting benchmark scores into application categories
def benchmark_to_application_label(benchmark_label):
    # empty or missing benchmark values are treated as other
    clean_label = str(
        benchmark_label or "other"
    ).strip().lower()

    # unknown benchmark values fall back to Neutral
    return BENCHMARK_TO_APPLICATION.get(
        clean_label,
        "Neutral",
    )