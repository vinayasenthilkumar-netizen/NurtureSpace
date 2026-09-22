
# caches the whisper model so it only needs to be loaded once
from functools import lru_cache

# used to check that the prepared audio file exists before transcription
from pathlib import Path

# imports the selected whisper model and default language
from config.settings import (
    DEFAULT_LANGUAGE_CODE,
    SELECTED_SPEECH_TO_TEXT_MODEL,
)
# whisper may not be installed in every environment
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    WHISPER_AVAILABLE = False


# loads the selected whisper model once and reuses it
@lru_cache(maxsize=1)
def load_speech_to_text_model():
    # return no model if whisper is unavailable
    if not WHISPER_AVAILABLE:
        return None

    return whisper.load_model(
        SELECTED_SPEECH_TO_TEXT_MODEL
    )


# creates a consistent result when transcription cannot be completed
def create_transcription_error(message):
    return {
        "transcript": "",
        "language": DEFAULT_LANGUAGE_CODE,
        "success": False,
        "error": str(message or ""),
    }


# transcribes one prepared temporary audio file using whisper
def transcribe_audio(audio_path):
    path = Path(str(audio_path or ""))

    # transcription cannot continue if the prepared audio is missing
    if not audio_path or not path.exists():
        return create_transcription_error(
            "The prepared audio file was not found."
        )
    # make sure the supplied path points to an actual file
    if not path.is_file():
        return create_transcription_error(
            "The supplied audio path is not a file."
        )
    # use the cached whisper model
    model = load_speech_to_text_model()

    if model is None:
        return create_transcription_error(
            "Whisper is not available."
        )
    try:
        # audio is already prepared as 16 khz mono before reaching this function
        result = model.transcribe(
            str(path),
            language=DEFAULT_LANGUAGE_CODE,
            fp16=False,
        )
        # remove extra whitespace from the returned transcript
        transcript = str(
            result.get("text", "")
        ).strip()
        # an empty transcript is treated as an unsuccessful transcription
        if not transcript:
            return create_transcription_error(
                "No clear speech could be transcribed."
            )
        # keep the language returned by whisper, with the configured language as fallback
        detected_language = str(
            result.get(
                "language",
                DEFAULT_LANGUAGE_CODE,
            )
        ).strip()

        return {
            "transcript": transcript,
            "language": (
                detected_language
                or DEFAULT_LANGUAGE_CODE
            ),
            "success": True,
            "error": "",
        }

    # return a standard error result instead of allowing whisper errors to stop the workflow
    except Exception as error:
        return create_transcription_error(
            f"Transcription failed: {error}"
        )