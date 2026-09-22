
# used for type hints in the returned dictionaries
from typing import Any
# imports helpers for saving, preparing and deleting temporary audio files
from modules.audio_utils import (
    delete_temp_files,
    prepare_audio_file,
    save_audio_to_temp_file,
)
# imports the audio model and speech transcription functions
from model_adapters.audio_emotion import predict_audio_observation
from model_adapters.speech_to_text import transcribe_audio
# creates the standard empty result used when voice processing cannot continue
def create_empty_voice_result(error_message: str = "") -> dict[str, Any]:
    return {
        "processing_complete": False,
        "transcript": "",
        "transcription_success": False,
        "text_emotion_suggestions": [],
        "supportive_factor_suggestions": [],
        "strain_factor_suggestions": [],
        "theme_found": False,
        "audio_observation": {
            "observation": "Unclear",
            "raw_label": "",
            "score": 0.0,
            "clear_result": False,
            "error": "",
        },
        "duration_seconds": 0.0,
        "prepared_audio_path": "",
        "temporary_files": [],
        "errors": [str(error_message)] if error_message else [],
    }

# prepares one voice reflection, transcribes it and analyses the vocal pattern
def process_voice_reflection(audio_file) -> dict[str, Any]:
    # prepare and analyse one voice reflection. wait for the user to confirm the transcript before analysing its emotions and themes.
    temporary_files = []
    errors = []
    # save the uploaded audio and convert it to the required format
    try:
        original_audio_path = save_audio_to_temp_file(audio_file)
        temporary_files.append(original_audio_path)

        prepared_result = prepare_audio_file(original_audio_path)
        prepared_audio_path = prepared_result["prepared_path"]
        temporary_files.append(prepared_audio_path)

    except Exception as error:
        # remove temporary files if preparation fails
        delete_temp_files(temporary_files)
        return create_empty_voice_result(str(error))

    # transcribe the prepared audio
    try:
        transcription_result = transcribe_audio(prepared_audio_path)

    except Exception as error:
        transcription_result = {
            "transcript": "",
            "language": "en",
            "success": False,
            "error": f"Transcription failed: {error}",
        }

    transcript = str(
        transcription_result.get("transcript", "") or ""
    ).strip()
    transcription_success = bool(
        transcription_result.get("success", False)
    )
    transcription_error = str(
        transcription_result.get("error", "") or ""
    ).strip()
    if transcription_error:
        errors.append(transcription_error)

    # run the vocal-pattern model on the prepared audio
    try:
        audio_observation = predict_audio_observation(
            prepared_audio_path
        )

    except Exception as error:
        audio_observation = {
            "observation": "Unclear",
            "raw_label": "",
            "score": 0.0,
            "clear_result": False,
            "error": f"Audio analysis failed: {error}",
        }

    audio_error = str(
        audio_observation.get("error", "") or ""
    ).strip()
    if audio_error:
        errors.append(audio_error)

    return {
        "processing_complete": True,
        "transcript": transcript,
        "transcription_success": transcription_success,
        # these are filled later after the user reviews the transcript
        "text_emotion_suggestions": [],
        "supportive_factor_suggestions": [],
        "strain_factor_suggestions": [],
        "theme_found": False,
        "audio_observation": audio_observation,
        "duration_seconds": prepared_result["duration_seconds"],
        "prepared_audio_path": prepared_audio_path,
        "temporary_files": temporary_files,
        "errors": errors,
    }


# removes temporary audio files after the voice reflection is no longer needed
def cleanup_voice_result(voice_result: dict[str, Any]) -> int:
    #delete temporary files belonging to one voice result
    if not isinstance(voice_result, dict):
        return 0
    temporary_files = voice_result.get("temporary_files", [])
    deleted_count = delete_temp_files(temporary_files)
    # clear stored temporary paths after cleanup
    voice_result["temporary_files"] = []
    voice_result["prepared_audio_path"] = ""
    return deleted_count