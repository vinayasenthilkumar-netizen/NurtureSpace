
# 4 main questions
SLEEP_QUESTION = "How would you describe your sleep quality today?"
MOOD_QUESTION = "How would you describe your mood today?"
STRESS_QUESTION = "How stressed do you feel today?"
SUPPORT_QUESTION = "How supported do you feel today?"

# 6 context questions
FOOD_QUESTION = "Were you able to eat enough for your own needs today?"
MEDICATION_QUESTION = (
    "If you had medicines or supplements to take today, "
    "were you able to take them as intended?"
)
PHYSICAL_RECOVERY_QUESTION = "How manageable was your physical recovery or discomfort today?"
BABY_CARE_QUESTION = "How manageable did caring for your baby feel today?"

OTHER_RESPONSIBILITIES_QUESTION = "How manageable did your other responsibilities feel today?"
PERSONAL_CARE_QUESTION = "How much time did you have for your own basic care today?"

# 5 response labels
# The dictionary keys are the numerical values used by the status calculation(values are the labels shown to the user)
SLEEP_LABELS = {
    1: "Very poorly",
    2: "Poorly",
    3: "Neither poorly nor well",
    4: "Well",
    5: "Very well"
}
MOOD_LABELS = {
    1: "Very low",
    2: "Low",
    3: "Neutral",
    4: "Good",
    5: "Very good"
}
STRESS_LABELS = {
    1: "Not stressed",
    2: "Slightly stressed",
    3: "Moderately stressed",
    4: "Very stressed",
    5: "Extremely stressed"
}
SUPPORT_LABELS = {
    1: "Not supported",
    2: "Slightly supported",
    3: "Moderately supported",
    4: "Well supported",
    5: "Very well supported"
}

# version of questions used for this questionnaire. (this is also saved with each database record)
QUESTION_SET_VERSION = "2.0"
# core
CORE_STATUS_QUESTION_KEYS = [
    "sleep",
    "mood",
    "stress",
    "support"
]
# context
CONTEXT_QUESTION_KEYS = [
    "food",
    "medication",
    "physical_recovery",
    "baby_care",
    "other_responsibilities",
    "personal_care"
]

# postpartum context response labels
FOOD_LABELS = {
    1: "Not at all",
    2: "Very little",
    3: "Partly",
    4: "Mostly",
    5: "Yes, comfortably"
}
MEDICATION_LABELS = {
    1: "None of them",
    2: "A few of them",
    3: "Some of them",
    4: "Most of them",
    5: "All of them",
    0: "Not applicable today"
}
PHYSICAL_RECOVERY_LABELS = {
    1: "Very difficult",
    2: "Difficult",
    3: "Mixed",
    4: "Mostly manageable",
    5: "Very manageable"
}
BABY_CARE_LABELS = {
    1: "Very difficult",
    2: "Difficult",
    3: "Mixed",
    4: "Mostly manageable",
    5: "Very manageable",
    0: "Not applicable today"
}
OTHER_RESPONSIBILITIES_LABELS = {
    1: "Very difficult",
    2: "Difficult",
    3: "Mixed",
    4: "Mostly manageable",
    5: "Very manageable",
    0: "Not applicable today"
}
PERSONAL_CARE_LABELS = {
    1: "No time",
    2: "Very little time",
    3: "Some time",
    4: "Mostly enough time",
    5: "Enough time"
}


# question status labels ( indicators)
STEADY_STATUS = "Feeling relatively steady today"
SOME_STRAIN_STATUS = "Experiencing some strain today"
SIGNIFICANT_STRAIN_STATUS = (
    "Experiencing significant strain today"
)
INCOMPLETE_STATUS = (
    "Please complete all four questions before calculating "
    "Today’s Check-In Status."
)

# reflection choices
TEXT_REFLECTION = "Text reflection"
VOICE_REFLECTION = "Voice reflection"
VIDEO_REFLECTION = "Video reflection"

REFLECTION_OPTIONS = [
    TEXT_REFLECTION,
    VOICE_REFLECTION,
    VIDEO_REFLECTION
]

# the labels  help the users understand where each of the result came from.
SOURCE_USER_RESPONSE = "Your response"
SOURCE_AI_SUGGESTION = "AI suggestion"
SOURCE_USER_NOTE = "Your added note"

# text categories
# simple emotion names used throughout the application.
TEXT_EMOTION_POSITIVE = "Positive"
TEXT_EMOTION_CONNECTION = "Warmth"
TEXT_EMOTION_LOW_MOOD = "Low mood"
TEXT_EMOTION_WORRY = "Worry"
TEXT_EMOTION_FRUSTRATION = "Frustration"
TEXT_EMOTION_NEUTRAL = "Neutral"

TEXT_EMOTION_CATEGORIES = [
    TEXT_EMOTION_POSITIVE,
    TEXT_EMOTION_CONNECTION,
    TEXT_EMOTION_LOW_MOOD,
    TEXT_EMOTION_WORRY,
    TEXT_EMOTION_FRUSTRATION,
    TEXT_EMOTION_NEUTRAL
]

# audio observation categories
# labels describe the broad vocal patterns only.they are suggestions and are not diagnoses.
AUDIO_OBSERVATION_CALM = "Calm"
AUDIO_OBSERVATION_ENERGETIC = "Energetic"
AUDIO_OBSERVATION_LOW_ENERGY = "Low energy"
AUDIO_OBSERVATION_WORRIED = "Worried"
AUDIO_OBSERVATION_FRUSTRATED = "Frustrated"
AUDIO_OBSERVATION_DISGUST = "Discomfort"
AUDIO_OBSERVATION_UNCLEAR = "Unclear"

AUDIO_OBSERVATION_CATEGORIES = [
    AUDIO_OBSERVATION_CALM,
    AUDIO_OBSERVATION_ENERGETIC,
    AUDIO_OBSERVATION_LOW_ENERGY,
    AUDIO_OBSERVATION_WORRIED,
    AUDIO_OBSERVATION_FRUSTRATED,
    AUDIO_OBSERVATION_DISGUST,
    AUDIO_OBSERVATION_UNCLEAR
]

# vieo expression categories
# raw expression labels produced by the selected video model.
VIDEO_EXPRESSION_ANGER = "anger"
VIDEO_EXPRESSION_CONTEMPT = "contempt"
VIDEO_EXPRESSION_DISGUST = "disgust"
VIDEO_EXPRESSION_FEAR = "fear"
VIDEO_EXPRESSION_HAPPINESS = "happiness"
VIDEO_EXPRESSION_NEUTRAL = "neutral"
VIDEO_EXPRESSION_SADNESS = "sadness"
VIDEO_EXPRESSION_SURPRISE = "surprise"

VIDEO_EXPRESSION_CATEGORIES = [
    VIDEO_EXPRESSION_ANGER,
    VIDEO_EXPRESSION_CONTEMPT,
    VIDEO_EXPRESSION_DISGUST,
    VIDEO_EXPRESSION_FEAR,
    VIDEO_EXPRESSION_HAPPINESS,
    VIDEO_EXPRESSION_NEUTRAL,
    VIDEO_EXPRESSION_SADNESS,
    VIDEO_EXPRESSION_SURPRISE,
]

# video observation labels
# cautious user facing descriptions of visible expression patterns. they do not describe a persons
#underlying emotional state or provide a diagnosis.
VIDEO_OBSERVATION_TENSE = "Tense-looking expression"
VIDEO_OBSERVATION_DISAPPROVING = "Disapproving-looking expression"
VIDEO_OBSERVATION_UNCOMFORTABLE ="Uncomfortable-looking expression"
VIDEO_OBSERVATION_UNCERTAIN = "Uncertain or wide-eyed expression"
VIDEO_OBSERVATION_POSITIVE = "Smiling or positive-looking expression"
VIDEO_OBSERVATION_NEUTRAL = "Neutral-looking expression"
VIDEO_OBSERVATION_DOWNTURNED = "Downturned-looking expression"
VIDEO_OBSERVATION_SURPRISED = "Surprised-looking expression"
VIDEO_OBSERVATION_UNCLEAR = "No clear visible-expression pattern"