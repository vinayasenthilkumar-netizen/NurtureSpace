# used for temporarily changing an environment setting while loading the model
import os
# counts how often each expression appears across sampled face crops
from collections import Counter
# caches the expression model so it is loaded only once
from functools import lru_cache
# used for checking and opening temporary face-crop paths
from pathlib import Path
# opencv loads face crops and torch checks whether cuda is available
import cv2
import torch
# imports the selected model and minimum amount of face evidence required
from config.settings import (
    MIN_VIDEO_EXPRESSION_FACES,
    SELECTED_VIDEO_EXPRESSION_MODEL,
)
# imports raw expression labels and safer user-facing observation labels
from core.constants import (
    VIDEO_EXPRESSION_ANGER,
    VIDEO_EXPRESSION_CATEGORIES,
    VIDEO_EXPRESSION_CONTEMPT,
    VIDEO_EXPRESSION_DISGUST,
    VIDEO_EXPRESSION_FEAR,
    VIDEO_EXPRESSION_HAPPINESS,
    VIDEO_EXPRESSION_NEUTRAL,
    VIDEO_EXPRESSION_SADNESS,
    VIDEO_EXPRESSION_SURPRISE,
    VIDEO_OBSERVATION_DISAPPROVING,
    VIDEO_OBSERVATION_DOWNTURNED,
    VIDEO_OBSERVATION_NEUTRAL,
    VIDEO_OBSERVATION_POSITIVE,
    VIDEO_OBSERVATION_SURPRISED,
    VIDEO_OBSERVATION_TENSE,
    VIDEO_OBSERVATION_UNCLEAR,
    VIDEO_OBSERVATION_UNCOMFORTABLE,
    VIDEO_OBSERVATION_UNCERTAIN,
)

# the hsemotion package may not be available in every environment
try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True

except ImportError:
    HSEmotionRecognizer = None
    HSEMOTION_AVAILABLE = False

# standard fallback shown when the video evidence is not clear enough
UNCLEAR_VISIBLE_EXPRESSION = VIDEO_OBSERVATION_UNCLEAR
# maps raw model labels into more cautious wording for the user
VISIBLE_EXPRESSION_MAP = {
    VIDEO_EXPRESSION_ANGER: VIDEO_OBSERVATION_TENSE,
    VIDEO_EXPRESSION_CONTEMPT: VIDEO_OBSERVATION_DISAPPROVING,
    VIDEO_EXPRESSION_DISGUST: VIDEO_OBSERVATION_UNCOMFORTABLE,
    VIDEO_EXPRESSION_FEAR: VIDEO_OBSERVATION_UNCERTAIN,
    VIDEO_EXPRESSION_HAPPINESS: VIDEO_OBSERVATION_POSITIVE,
    VIDEO_EXPRESSION_NEUTRAL: VIDEO_OBSERVATION_NEUTRAL,
    VIDEO_EXPRESSION_SADNESS: VIDEO_OBSERVATION_DOWNTURNED,
    VIDEO_EXPRESSION_SURPRISE: VIDEO_OBSERVATION_SURPRISED,
}


# loads the selected expression model once and reuses it
@lru_cache(maxsize=1)
def load_video_expression_model():
    # return no model if the optional dependency is unavailable
    if not HSEMOTION_AVAILABLE:
        return None
    # use cuda when a compatible gpu is available, otherwise use cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # keep the existing torch environment value so it can be restored
    previous_setting = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    try:
        try:
            # first try the best device available on this computer
            return HSEmotionRecognizer(
                model_name=SELECTED_VIDEO_EXPRESSION_MODEL,
                device=device,
            )

        except Exception:
            # if cuda was detected but model loading fails, retry on cpu
            if device == "cuda":
                return HSEmotionRecognizer(
                    model_name=SELECTED_VIDEO_EXPRESSION_MODEL,
                    device="cpu",
                )

            raise

    finally:
        # restore the environment to its earlier state after model loading
        if previous_setting is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_setting


# creates the standard fallback result when usable expression evidence is missing
def create_empty_expression_result(error_message=""):
    return {
        "observation": UNCLEAR_VISIBLE_EXPRESSION,
        "dominant_raw_label": "",
        "raw_predictions": [],
        "all_scores": [],
        "frame_agreement": 0.0,

        "analysed_face_count": 0,
        "dominant_prediction_count": 0,
        "required_majority_count": 0,
        "minimum_required_face_count": MIN_VIDEO_EXPRESSION_FACES,
        "enough_face_evidence": False,
        "has_strict_majority": False,
        "evidence_reason": (
            "No sufficient visible-expression evidence was available."
        ),

        "clear_result": False,
        "error": str(error_message or ""),
    }


# cleans one raw expression label before it is compared or mapped
def normalise_raw_expression(raw_label):
    return str(raw_label or "").strip().lower()


# converts the model's raw expression into cautious display wording
def map_visible_expression(raw_label):
    normalised_label = normalise_raw_expression(raw_label)

    return VISIBLE_EXPRESSION_MAP.get(
        normalised_label,
        UNCLEAR_VISIBLE_EXPRESSION,
    )


# loads one temporary face crop and converts it into rgb for the model
def load_face_crop(face_path):
    path = Path(str(face_path or ""))

    if not face_path or not path.exists():
        raise FileNotFoundError(
            f"Face crop was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            "The supplied face-crop path is not a file."
        )

    # opencv reads images in bgr format
    bgr_image = cv2.imread(str(path))

    if bgr_image is None:
        raise ValueError(
            "The face crop could not be opened."
        )

    # hsemotion expects normal rgb channel order
    return cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2RGB,
    )


# finds the most common frame prediction and how strongly the frames agree
def calculate_frame_agreement(raw_predictions):
    # remove blank predictions before counting them
    cleaned_predictions = [
        str(prediction).strip()
        for prediction in raw_predictions or []
        if str(prediction).strip()
    ]

    if not cleaned_predictions:
        return "", 0.0

    # lower-case values are used only for matching and counting
    normalised_predictions = [
        prediction.lower()
        for prediction in cleaned_predictions
    ]

    prediction_counts = Counter(normalised_predictions)

    # get the expression with the most frame votes
    dominant_label, dominant_count = prediction_counts.most_common(1)[0]

    # agreement is the proportion of frames sharing the dominant label
    agreement = dominant_count / len(normalised_predictions)

    # return the original label rather than the lower-case version
    original_index = normalised_predictions.index(dominant_label)

    return (
        cleaned_predictions[original_index],
        round(agreement, 4),
    )


# converts all frame predictions into proportions for internal scoring
def build_expression_scores(raw_predictions):
    normalised_predictions = [
        normalise_raw_expression(prediction)
        for prediction in raw_predictions or []
        if normalise_raw_expression(prediction)
    ]

    if not normalised_predictions:
        return []

    prediction_counts = Counter(normalised_predictions)
    total_predictions = len(normalised_predictions)
    all_scores = []

    # keep every expression category, including categories with zero evidence
    for label in VIDEO_EXPRESSION_CATEGORIES:
        score = prediction_counts.get(label, 0) / total_predictions

        all_scores.append({
            "raw_label": label,
            "score": round(score, 4),
        })

    return all_scores


# calculates how many matching predictions are needed for more than half
def calculate_required_majority(prediction_count):
    try:
        clean_count = int(prediction_count)

    except (TypeError, ValueError):
        return 0

    if clean_count <= 0:
        return 0

    # for example, 5 predictions require at least 3 matching votes
    return (clean_count // 2) + 1


# analyses temporary face crops and decides whether there is enough evidence to display
def predict_visible_expression(face_paths):
    # remove empty paths before trying to open any face crops
    valid_paths = [
        face_path
        for face_path in face_paths or []
        if face_path
    ]

    if not valid_paths:
        return create_empty_expression_result(
            "No face crops were supplied for analysis."
        )

    # load the cached model
    try:
        expression_model = load_video_expression_model()

    except Exception as error:
        return create_empty_expression_result(
            "The facial-expression model could not be loaded: "
            f"{error}"
        )

    if expression_model is None:
        return create_empty_expression_result(
            "The facial-expression model is not available."
        )

    face_images = []

    # one bad crop should not fail the whole video
    for face_path in valid_paths:
        try:
            face_images.append(
                load_face_crop(face_path)
            )

        except Exception:
            continue

    if not face_images:
        return create_empty_expression_result(
            "No usable face crops could be opened."
        )

    try:
        # process the representative face crops together
        raw_predictions, _ = expression_model.predict_multi_emotions(
            face_images,
            logits=False,
        )

        # remove any blank predictions returned by the model
        raw_predictions = [
            str(prediction).strip()
            for prediction in raw_predictions or []
            if str(prediction).strip()
        ]

        # retain all frame evidence for internal scoring
        all_scores = build_expression_scores(raw_predictions)
        analysed_face_count = len(raw_predictions)

        # find the most common visible-expression prediction
        dominant_raw_label, frame_agreement = calculate_frame_agreement(
            raw_predictions
        )

        normalised_dominant_label = normalise_raw_expression(
            dominant_raw_label
        )

        # count how many crops produced the dominant prediction
        dominant_prediction_count = sum(
            1
            for prediction in raw_predictions
            if normalise_raw_expression(prediction) == normalised_dominant_label
        )

        # work out how many votes are needed for more than half
        required_majority_count = calculate_required_majority(
            analysed_face_count
        )

        # enough usable face predictions must be available
        enough_face_evidence = (
            analysed_face_count >= MIN_VIDEO_EXPRESSION_FACES
        )

        # exactly half is not accepted as a majority
        has_strict_majority = (
            dominant_prediction_count >= required_majority_count
            and required_majority_count > 0
        )

        # map the raw model label into cautious wording
        observation = map_visible_expression(
            dominant_raw_label
        )

        # unmapped labels should not become clear user-facing observations
        mapped_observation_is_allowed = (
            observation != UNCLEAR_VISIBLE_EXPRESSION
        )

        # all evidence conditions must pass
        clear_result = (
            enough_face_evidence
            and has_strict_majority
            and mapped_observation_is_allowed
        )

        # explain why the result was accepted or rejected
        if not enough_face_evidence:
            evidence_reason = (
                f"Only {analysed_face_count} usable face prediction(s) "
                f"were available. At least "
                f"{MIN_VIDEO_EXPRESSION_FACES} are required."
            )

        elif not has_strict_majority:
            evidence_reason = (
                "No expression label received a strict majority. "
                f"The dominant label appeared "
                f"{dominant_prediction_count} time(s), while "
                f"{required_majority_count} were required."
            )

        elif not mapped_observation_is_allowed:
            evidence_reason = (
                "The dominant raw model label was not mapped to a "
                "user-facing visible-expression observation."
            )

        else:
            evidence_reason = (
                "The minimum face count was met and the dominant "
                "label received a strict majority."
            )

        # unclear evidence should never produce a definite display observation
        if not clear_result:
            observation = UNCLEAR_VISIBLE_EXPRESSION

        return {
            "observation": observation,
            "dominant_raw_label": dominant_raw_label,
            "raw_predictions": raw_predictions,
            "all_scores": all_scores,
            "frame_agreement": frame_agreement,

            "analysed_face_count": analysed_face_count,
            "dominant_prediction_count": dominant_prediction_count,
            "required_majority_count": required_majority_count,
            "minimum_required_face_count": MIN_VIDEO_EXPRESSION_FACES,
            "enough_face_evidence": enough_face_evidence,
            "has_strict_majority": has_strict_majority,
            "evidence_reason": evidence_reason,

            "clear_result": clear_result,
            "error": "",
        }

    # unexpected inference errors use the same safe unclear result
    except Exception as error:
        return create_empty_expression_result(
            f"Visible-expression analysis failed: {error}"
        )