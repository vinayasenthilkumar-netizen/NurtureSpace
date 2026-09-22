
# imports the approved text, audio and video labels used when assigning directions
from core.constants import (
    AUDIO_OBSERVATION_CALM,AUDIO_OBSERVATION_ENERGETIC,
    AUDIO_OBSERVATION_FRUSTRATED,AUDIO_OBSERVATION_LOW_ENERGY,
    AUDIO_OBSERVATION_DISGUST,AUDIO_OBSERVATION_WORRIED,
    TEXT_EMOTION_CONNECTION,TEXT_EMOTION_FRUSTRATION,
    TEXT_EMOTION_LOW_MOOD,TEXT_EMOTION_POSITIVE,
    TEXT_EMOTION_NEUTRAL,TEXT_EMOTION_WORRY,
    VIDEO_EXPRESSION_ANGER,VIDEO_EXPRESSION_CONTEMPT,
    VIDEO_EXPRESSION_DISGUST,VIDEO_EXPRESSION_FEAR,
    VIDEO_EXPRESSION_HAPPINESS,VIDEO_EXPRESSION_NEUTRAL,
    VIDEO_EXPRESSION_SADNESS,VIDEO_EXPRESSION_SURPRISE,
)


# all question and reflection scores use the same 1-5 range
MIN_RESPONSE = 1
MAX_RESPONSE = 5
# thresholds used to convert scores into the three user-facing bands
SOME_STRAIN_THRESHOLD = 2.34
STEADY_THRESHOLD = 3.67
# weights used only when both reflection and question scores are available
REFLECTION_WEIGHT = 0.70
QUESTION_WEIGHT = 0.30
# labels shown for the four-question check-in result
STEADY_STATUS = "Feeling relatively steady today"
SOME_STRAIN_STATUS = "Experiencing some strain today"
SIGNIFICANT_STRAIN_STATUS = "Experiencing significant strain today"

# cautious wording used for reflection-only scoring
REFLECTION_STEADY_INDICATOR = (
    "Reflection suggests a relatively steady experience today"
)

REFLECTION_SOME_STRAIN_INDICATOR = (
    "Reflection suggests some strain today"
)

REFLECTION_SIGNIFICANT_STRAIN_INDICATOR = (
    "Reflection suggests more strain today"
)

# shorter labels used for the overall combined indicator
COMBINED_STEADY_INDICATOR = "Relatively steady"
COMBINED_SOME_STRAIN_INDICATOR = "Some strain"
COMBINED_SIGNIFICANT_STRAIN_INDICATOR = "Significant strain"


# validation 

# checks one structured question response before it is used in scoring
def validate_checkin_response(response, question_name):
    # check if response is from 1 to 5
    if response is None:
        raise ValueError(
            f"Please complete the {question_name} question."
        )
    # bool is not accepted because the response must be a normal integer
    if type(response) is not int:
        raise ValueError(
            f"The {question_name} response must be a whole number "
            f"between {MIN_RESPONSE} and {MAX_RESPONSE}."
        )
    if response < MIN_RESPONSE or response > MAX_RESPONSE:
        raise ValueError(
            f"The {question_name} response must be between "
            f"{MIN_RESPONSE} and {MAX_RESPONSE}."
        )


# validates all four core responses used in the Question Score
def validate_all_checkin_responses(
    sleep,
    mood,
    stress,
    support,
):
    # validate the 4 core responses for the question score 
    validate_checkin_response(sleep, "sleep")
    validate_checkin_response(mood, "mood")
    validate_checkin_response(stress, "stress")
    validate_checkin_response(support, "perceived support")
    return True


# safely converts any score into the shared 1 to 5 range
def normalise_score(score):
    if score is None:
        return None
    try:
        clean_score = float(score)
    except (TypeError, ValueError):
        return None
    # invalid values are treated as unavailable rather than clipped
    if not 1.0 <= clean_score <= 5.0:
        return None
    return clean_score


#question scoring 
# reverses the stress answer so higher values point in the same direction
def reverse_stress_response(stress):
    validate_checkin_response(stress, "stress")
    # for a 1 to 5 scale this converts 1 to number 5, 2 to a 4, 4 to a 2 and 5 to a 1
    return 6 - stress


# calculates the equally weighted score from the four core questions
def calculate_checkin_average(
    sleep,
    mood,
    stress,
    support,
):
  # equal weightage 

    validate_all_checkin_responses(
        sleep,
        mood,
        stress,
        support,
    )
    # stress is reversed before averaging with the other three questions
    reversed_stress = reverse_stress_response(stress)
    # each of the four core questions contributes 25%
    average = (sleep+ mood+ reversed_stress+ support) / 4
    return round(average, 2)


# converts the question score into questionnaire Status
def get_status_from_average(checkin_average):
    clean_score = normalise_score(checkin_average)
    if clean_score is None:
        raise ValueError(
            "The check-in average must be between 1.00 and 5.00."
        )

    # higher scores represent a steadier overall response pattern
    if clean_score >= STEADY_THRESHOLD:
        return STEADY_STATUS
    if clean_score >= SOME_STRAIN_THRESHOLD:
        return SOME_STRAIN_STATUS
    return SIGNIFICANT_STRAIN_STATUS


# calculates the four-question score and returns its status label
def calculate_checkin_status(
    sleep,
    mood,
    stress,
    support,
):

    average = calculate_checkin_average(sleep=sleep,mood=mood,stress=stress,support=support,)
    return get_status_from_average(average)


# returns the values used to explain how the Question Score was produced
def generate_status_breakdown(sleep,mood,stress,support,):
    validate_all_checkin_responses(
        sleep,
        mood,
        stress,
        support,
    )
    reversed_stress = reverse_stress_response(stress)
    average = calculate_checkin_average(
        sleep=sleep,
        mood=mood,
        stress=stress,
        support=support,
    )
    status = get_status_from_average(average)

    # this breakdown makes it clear that ai reflection models do not affect this status
    return {
        "sleep_response": sleep,
        "mood_response": mood,
        "stress_response": stress,
        "reversed_stress_response": reversed_stress,
        "support_response": support,
        "weight_per_question": "25%",
        "internal_average": average,
        "checkin_status": status,
        "text_model_used_in_status": False,
        "audio_model_used_in_status": False,
        "visual_model_used_in_status": False,
    }


# reflection direction (important as well)
# maps each approved text category onto supportive,neutral or strain direction
TEXT_EMOTION_DIRECTIONS = {
    TEXT_EMOTION_POSITIVE: 1,
    TEXT_EMOTION_CONNECTION: 1,
    TEXT_EMOTION_NEUTRAL: 0,
    TEXT_EMOTION_LOW_MOOD: -1,
    TEXT_EMOTION_WORRY: -1,
    TEXT_EMOTION_FRUSTRATION: -1,
}
# maps each audio observation onto the same common direction scale
AUDIO_OBSERVATION_DIRECTIONS = {
    AUDIO_OBSERVATION_ENERGETIC: 1,
    AUDIO_OBSERVATION_CALM: 0,
    AUDIO_OBSERVATION_LOW_ENERGY: -1,
    AUDIO_OBSERVATION_DISGUST: -1,
    AUDIO_OBSERVATION_WORRIED: -1,
    AUDIO_OBSERVATION_FRUSTRATED: -1,
}
# visible-expression categories also use the common -1, 0, +1 direction
VIDEO_EXPRESSION_DIRECTIONS = {
    VIDEO_EXPRESSION_HAPPINESS: 1,
    VIDEO_EXPRESSION_NEUTRAL: 0,
    VIDEO_EXPRESSION_SURPRISE: 0,
    VIDEO_EXPRESSION_ANGER: -1,
    VIDEO_EXPRESSION_CONTEMPT: -1,
    VIDEO_EXPRESSION_DISGUST: -1,
    VIDEO_EXPRESSION_FEAR: -1,
    VIDEO_EXPRESSION_SADNESS: -1,
}


# text reflection score
# combines the full text-emotion distribution into one 1-5 component score
def calculate_text_reflection_score(
    emotion_suggestions
):
    # convert all Text emotion scores to the common 1 to 5 scale
    if not emotion_suggestions:
        return None

    directional_evidence = 0.0
    total_evidence = 0.0

    # every valid text category contributes according to its model confidence
    for suggestion in emotion_suggestions:
        if not isinstance(suggestion, dict):
            continue
        category = suggestion.get("category")

        # ignore any category that does not have a scoring direction
        if category not in TEXT_EMOTION_DIRECTIONS:
            continue
        try:
            model_score = float(suggestion.get("score", 0))
        except (TypeError, ValueError):
            continue
        if model_score <= 0:
            continue
        direction = TEXT_EMOTION_DIRECTIONS[category]
        # confidence is multiplied by -1, 0 or +1 depending on the category
        directional_evidence += direction * model_score
        total_evidence += model_score
    if total_evidence <= 0:
        return None

    # balance shows whether the available evidence leans positive or negative
    direction_balance = directional_evidence / total_evidence
    # cap total confidence at 1 so the final value stays controlled
    evidence_strength = min(total_evidence, 1.0)
    normalised_evidence = direction_balance * evidence_strength
    # map evidence from the -1 to +1 direction range onto the 1-5 score range
    text_score = 3 + (2 * normalised_evidence)
    # final safeguard keeps the score inside the common range
    text_score = max(
        1.0,
        min(5.0, text_score),
    )
    return round(text_score, 2)


# audio reflection score
# converts the full audio observation distribution into one 1-5 component score
def calculate_audio_reflection_score(audio_result):
    # audio emotion scores to the common 1-5. unclear audio result is treated as unavailable.
    if not isinstance(audio_result, dict):
        return None
    # unclear audio is not allowed to influence the reflection score
    if not audio_result.get("clear_result", False):
        return None
    all_scores = audio_result.get("all_scores", [])
    if not all_scores:
        return None
    directional_evidence = 0.0
    total_evidence = 0.0
    # use all valid audio categories rather than only the dominant result
    for result in all_scores:
        if not isinstance(result, dict):
            continue
        observation = result.get("observation")
        if observation not in AUDIO_OBSERVATION_DIRECTIONS:
            continue
        try:
            model_score = float(result.get("score", 0))
        except (TypeError, ValueError):
            continue

        if model_score <= 0:
            continue
        direction = AUDIO_OBSERVATION_DIRECTIONS[observation]
        directional_evidence += direction * model_score
        total_evidence += model_score

    if total_evidence <= 0:
        return None

    # calculate the overall direction of the audio evidence
    direction_balance = directional_evidence / total_evidence
    evidence_strength = min(total_evidence, 1.0)
    normalised_evidence = direction_balance * evidence_strength

    # midpoint 3 represents balanced or neutral evidence
    audio_score = 3 + (2 * normalised_evidence)
    audio_score = max(
        1.0,
        min(5.0, audio_score),
    )

    return round(audio_score, 2)


# video reflection 
# converts visible-expression proportions into one 1-5 component score
def calculate_video_reflection_score(video_result):
    if not isinstance(video_result, dict):
        return None
    # video should only contribute when enough face evidence was available
    if not video_result.get("enough_face_evidence", False):
        return None
    all_scores = video_result.get("all_scores", [])
    if not all_scores:
        return None
    directional_evidence = 0.0
    total_evidence = 0.0
    # use the full visible-expression distribution from the sampled frames
    for result in all_scores:
        if not isinstance(result, dict):
            continue
        raw_label = str(
            result.get("raw_label", "") or ""
        ).strip().lower()
        if raw_label not in VIDEO_EXPRESSION_DIRECTIONS:
            continue

        try:
            model_score = float(result.get("score", 0))
        except (TypeError, ValueError):
            continue

        if model_score <= 0:
            continue

        direction = VIDEO_EXPRESSION_DIRECTIONS[raw_label]
        directional_evidence += direction * model_score
        total_evidence += model_score

    if total_evidence <= 0:
        return None

    # neutral and surprise contribute zero direction but still count as evidence
    direction_balance = directional_evidence / total_evidence
    evidence_strength = min(total_evidence, 1.0)
    normalised_evidence = direction_balance * evidence_strength
    video_score = 3 + (2 * normalised_evidence)
    video_score = max(1.0,min(5.0, video_score))
    return round(video_score, 2)


# combined
# combines whichever reflection components are available for the selected mode
def calculate_reflection_score(
    text_score=None,
    audio_score=None,
    video_score=None,
):
    # combine Reflection components take note

    #Text:
        #100% Text
    #Voice:
        #50% Text
        #50% Audio
    #Video:
        #40% Text
        #40% Audio
        #20% Video
    #Video remains cut off at 20% when another component is unavailable.
  

    # make sure each component is a valid score before combining it
    text_score = normalise_score(text_score)
    audio_score = normalise_score(audio_score)
    video_score = normalise_score(video_score)
    # full video reflection uses transcript, audio and visible-expression evidence
    if (
        text_score is not None
        and audio_score is not None
        and video_score is not None
    ):
        reflection_score = ((text_score * 0.40)+ (audio_score * 0.40)+ (video_score * 0.20))

    # if audio is missing, text takes the unused non-video share
    elif (
        text_score is not None
        and audio_score is None
        and video_score is not None
    ):
        reflection_score = ((text_score * 0.80)+ (video_score * 0.20))

    # if text is missing, audio takes the unused non-video share
    elif (
        text_score is None
        and audio_score is not None
        and video_score is not None
    ):
        reflection_score = ((audio_score * 0.80)+ (video_score * 0.20))

    # voice reflections use equal transcript and audio weighting
    elif (
        text_score is not None
        and audio_score is not None
    ):
        reflection_score = ((text_score * 0.50)+ (audio_score * 0.50))

    # text-only reflection uses the text score directly
    elif text_score is not None:
        reflection_score = text_score
    # audio can still be used if it is the only available valid component
    elif audio_score is not None:
        reflection_score = audio_score
    else:
        return None

    return round(reflection_score, 2)


# converts the numerical Reflection Score into cautious user-facing wording
def get_reflection_indicator(reflection_score):
    # convert score into gentle indicator
    clean_score = normalise_score(reflection_score)
    if clean_score is None:
        return None
    if clean_score >= STEADY_THRESHOLD:
        return REFLECTION_STEADY_INDICATOR
    if clean_score >= SOME_STRAIN_THRESHOLD:
        return REFLECTION_SOME_STRAIN_INDICATOR
    return REFLECTION_SIGNIFICANT_STRAIN_INDICATOR

# combines reflection and question scores only when both are available
def calculate_combined_wellbeing_score(
    reflection_score,
    question_score,
):
    
    #Overall Check-In Indicator when both components are present
    #Reflection contributes 70%.
    #Questions contribute 30%.
    reflection_score = normalise_score(reflection_score)
    question_score = normalise_score(question_score)
    # the combined value is not calculated for reflection-only check-ins
    if reflection_score is None or question_score is None:
        return None
    # reflection has the larger contribution to the internal combined value
    combined_value = (reflection_score * REFLECTION_WEIGHT) + (question_score * QUESTION_WEIGHT)
    return round(combined_value, 2)


# converts the internal combined value into the final overall indicator
def get_combined_wellbeing_indicator(
    combined_value,
):
    clean_value = normalise_score(combined_value)
    if clean_value is None:
        return None
    # use the same three bands as the other 1-5 scores
    if clean_value >= STEADY_THRESHOLD:
        return COMBINED_STEADY_INDICATOR
    if clean_value >= SOME_STRAIN_THRESHOLD:
        return COMBINED_SOME_STRAIN_INDICATOR
    return COMBINED_SIGNIFICANT_STRAIN_INDICATOR