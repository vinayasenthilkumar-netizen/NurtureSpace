
# text-emotion model adapter used to analyse reflection text
from model_adapters.text_emotion import analyse_text_emotions

# cleans the reflection text before it is sent for analysis
def combine_reflection_text(typed_reflection):
    #clean the reflection text entered by the user
    return str(typed_reflection or "").strip()
# analyses one written reflection and separates scoring data from display suggestions
def analyse_written_reflection(
    typed_reflection,
    maximum_suggestions=3,
):
# Analyse a written reflection and return emotion scores and suggestions.
    combined_text = combine_reflection_text(
        typed_reflection
    )
    # empty reflection text should not be sent to the model
    if not combined_text:
        return {
            "combined_text": "",
            "emotion_suggestions": [],
            "all_emotion_scores": [],
        }
    # run the cleaned text through the emotion model
    emotion_result = analyse_text_emotions(
        text=combined_text,
        maximum_suggestions=maximum_suggestions,
    )
    # keep both the display suggestions and full scores
    return {
        "combined_text": combined_text,
        "emotion_suggestions": emotion_result.get(
            "suggestions",
            [],
        ),
        "all_emotion_scores": emotion_result.get(
            "all_scores",
            [],
        ),
    }