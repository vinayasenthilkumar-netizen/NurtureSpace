# imports the scoring module being tested
from modules import scoring
# imports the standard emotion and expression labels used in scoring tests
from core.constants import (
    AUDIO_OBSERVATION_CALM, AUDIO_OBSERVATION_ENERGETIC,
    AUDIO_OBSERVATION_LOW_ENERGY, TEXT_EMOTION_LOW_MOOD,
    TEXT_EMOTION_NEUTRAL, TEXT_EMOTION_POSITIVE,
    VIDEO_EXPRESSION_HAPPINESS, VIDEO_EXPRESSION_NEUTRAL,
    VIDEO_EXPRESSION_SADNESS, VIDEO_EXPRESSION_SURPRISE,
)


# checks that score values are converted and validated correctly
def test_normalise():
    # the minimum valid score should be returned as a float
    assert scoring.normalise_score(1) == 1.0
    # a numeric string within the valid range should be converted to a float
    assert scoring.normalise_score("3.5") == 3.5
    # the maximum valid score should be returned as a float
    assert scoring.normalise_score(5) == 5.0
    # invalid types and values outside the 1 to 5 range should return none
    for value in [None, "bad", 0.99, 5.01, [], {}]:
        assert scoring.normalise_score(value) is None


# checks how text emotion probabilities are converted into a reflection score
def test_text_score():
    # create a fixed set of positive, low mood and neutral emotion scores
    scores = [
        {"category": TEXT_EMOTION_POSITIVE, "score": 0.60},
        {"category": TEXT_EMOTION_LOW_MOOD, "score": 0.20},
        {"category": TEXT_EMOTION_NEUTRAL, "score": 0.20},
    ]

    # the weighted emotion evidence should produce the expected text score
    assert scoring.calculate_text_reflection_score(scores) == 3.8
    # lower total evidence should have lower strength
    weak = [{"category": TEXT_EMOTION_POSITIVE, "score": 0.40}]
    # normalising the available evidence should still produce the same direction score
    assert scoring.calculate_text_reflection_score(weak) == 3.8
    # fully neutral text evidence should remain at the midpoint of 3
    neutral = [{"category": TEXT_EMOTION_NEUTRAL, "score": 1.0}]
    assert scoring.calculate_text_reflection_score(neutral) == 3.0
    # create invalid or unusable emotion-score entries
    bad = [
        {"category": "unknown", "score": 0.8},
        {"category": TEXT_EMOTION_POSITIVE, "score": "bad"},
        {"category": TEXT_EMOTION_LOW_MOOD, "score": 0},
    ]

    # unusable emotion evidence should not produce a reflection score
    assert scoring.calculate_text_reflection_score(bad) is None
    # an empty emotion list should also return none
    assert scoring.calculate_text_reflection_score([]) is None


# checks how audio observation probabilities are converted into a reflection score
def test_audio_score():
    # create a clear audio result containing the full observation distribution
    result = {
        "clear_result": True,
        "all_scores": [
            {"observation": AUDIO_OBSERVATION_ENERGETIC, "score": 0.60},
            {"observation": AUDIO_OBSERVATION_CALM, "score": 0.10},
            {"observation": AUDIO_OBSERVATION_LOW_ENERGY, "score": 0.30},
        ],
    }

    # the fixed audio probabilities should produce the expected score
    assert scoring.calculate_audio_reflection_score(result) == 3.6
    # mark the same result as unclear
    result["clear_result"] = False
    # unclear audio evidence should not be used for scoring
    assert scoring.calculate_audio_reflection_score(result) is None
    # missing audio information should also return none
    assert scoring.calculate_audio_reflection_score({}) is None


# checks how visible-expression probabilities are converted into a reflection score
def test_video_score():
    # create a valid visible-expression result with enough face evidence
    result = {
        "enough_face_evidence": True,
        "all_scores": [
            {"raw_label": VIDEO_EXPRESSION_HAPPINESS, "score": 0.40},
            {"raw_label": VIDEO_EXPRESSION_NEUTRAL, "score": 0.20},
            {"raw_label": VIDEO_EXPRESSION_SURPRISE, "score": 0.10},
            {"raw_label": VIDEO_EXPRESSION_SADNESS, "score": 0.30},
        ],
    }

    # the full visible-expression probability distribution should produce the expected score
    assert scoring.calculate_video_reflection_score(result) == 3.2
    # create a result containing only neutral and surprise evidence
    neutral = {
        "enough_face_evidence": True,
        "all_scores": [
            {"raw_label": VIDEO_EXPRESSION_NEUTRAL, "score": 0.55},
            {"raw_label": VIDEO_EXPRESSION_SURPRISE, "score": 0.45},
        ],
    }

    # neutral and surprise are treated as midpoint evidence in this scoring test
    assert scoring.calculate_video_reflection_score(neutral) == 3.0
    # mark the original video result as having insufficient face evidence
    result["enough_face_evidence"] = False
    # video evidence should not be scored when there are not enough usable face predictions
    assert scoring.calculate_video_reflection_score(result) is None


# checks the weighting rules used to combine reflection components
def test_reflection_weights():
    # a text-only reflection should use its text score directly
    assert scoring.calculate_reflection_score(text_score=4.2) == 4.2
    # an audio-only score should be returned directly when it is the only available component
    assert scoring.calculate_reflection_score(audio_score=2.5) == 2.5
    # text and audio together should be weighted equally
    assert scoring.calculate_reflection_score(text_score=5, audio_score=1) == 3.0
    # video = 40% text + 40% audio + 20% video
    assert scoring.calculate_reflection_score(5, 1, 3) == 3.0
    # when audio is missing, the remaining text and video weights are adjusted
    assert scoring.calculate_reflection_score(4, None, 2) == 3.6
    # when text is missing, the remaining audio and video weights are adjusted
    assert scoring.calculate_reflection_score(None, 4, 2) == 3.6
    # video evidence alone is not enough to create a reflection score
    assert scoring.calculate_reflection_score(None, None, 4) is None
    # invalid scores are ignored
    assert scoring.calculate_reflection_score(6, 4) == 4.0
    # unusable text with only video evidence should not produce a reflection score
    assert scoring.calculate_reflection_score("bad", None, 3) is None


# checks the boundary values used for reflection indicator labels
def test_reflection_indicators():
    # define scores directly around each indicator threshold
    cases = [
        (3.67, scoring.REFLECTION_STEADY_INDICATOR),
        (3.66, scoring.REFLECTION_SOME_STRAIN_INDICATOR),
        (2.34, scoring.REFLECTION_SOME_STRAIN_INDICATOR),
        (2.33, scoring.REFLECTION_SIGNIFICANT_STRAIN_INDICATOR),
    ]
    # check that every boundary score receives the expected indicator
    for score, expected in cases:
        assert scoring.get_reflection_indicator(score) == expected
    # no score should produce no reflection indicator
    assert scoring.get_reflection_indicator(None) is None


# checks the 70 percent reflection and 30 percent question combined score
def test_combined_score():
    # reflection 4 and question 2 should produce a combined score of 3.4
    assert scoring.calculate_combined_wellbeing_score(4, 2) == 3.4
    # reflection 2 and question 4 should produce a combined score of 2.6
    assert scoring.calculate_combined_wellbeing_score(2, 4) == 2.6
    # a combined score requires a valid reflection score
    assert scoring.calculate_combined_wellbeing_score(None, 4) is None
    # a combined score also requires a valid question score
    assert scoring.calculate_combined_wellbeing_score(4, None) is None
    # define scores around the combined wellbeing indicator boundaries
    cases = [
        (3.67, scoring.COMBINED_STEADY_INDICATOR),
        (3.66, scoring.COMBINED_SOME_STRAIN_INDICATOR),
        (2.34, scoring.COMBINED_SOME_STRAIN_INDICATOR),
        (2.33, scoring.COMBINED_SIGNIFICANT_STRAIN_INDICATOR),
    ]

    # check that each combined score receives the correct indicator
    for score, expected in cases:
        assert scoring.get_combined_wellbeing_indicator(score) == expected
    # a missing combined score should not produce an indicator
    assert scoring.get_combined_wellbeing_indicator(None) is None