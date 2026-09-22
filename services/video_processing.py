
# used for checking and changing temporary video file extensions
from pathlib import Path
# used for type hints in returned video-processing dictionaries
from typing import Any
# imports the configured number of frames sampled from each video
from config.settings import VIDEO_EXPRESSION_SAMPLE_FRAMES
# imports helpers used to save, prepare, extract and remove temporary video files
from modules.video_utils import (
    cleanup_video_files,
    convert_live_recording_to_mp4,
    extract_face_crops,
    extract_representative_frames,
    extract_video_audio,
    save_video_to_temp_file,
    validate_video_file,
)


# adds a processing error only when it is not empty or already recorded
def add_processing_error(
    errors: list[str],
    message: str,
) -> None:
    #a dd a non-empty processing error without duplicates

    clean_message = str(
        message or ""
    ).strip()

    if (
        clean_message
        and clean_message not in errors
    ):
        errors.append(
            clean_message
        )

# prepares the temporary video files needed before ai processing begins
def prepare_video_media(
    video_file,
    number_of_frames=VIDEO_EXPRESSION_SAMPLE_FRAMES,
):
    # prepare temporary media files. AI models run later in the Reflection langgraph
    errors = []
    temporary_video_paths = []
    temporary_audio_paths = []
    temporary_frame_paths = []
    temporary_face_paths = []
    # save the uploaded video and convert webm recordings when needed
    try:
        video_path = save_video_to_temp_file(
            video_file
        )
        temporary_video_paths.append(
            video_path
        )

        if Path(video_path).suffix.lower() == ".webm":
            mp4_path = str(
                Path(video_path).with_suffix(
                    ".mp4"
                )
            )
            converted_path = (
                convert_live_recording_to_mp4(
                    recording_path=video_path,
                    preview_path=mp4_path,
                )
            )

            temporary_video_paths.append(
                converted_path
            )

            video_path = converted_path

        video_metadata = validate_video_file(
            video_path
        )

    except Exception as error:
        # remove any temporary video files if preparation fails
        cleanup_video_files(
            temporary_video_paths
        )

        return {
            "preparation_complete": False,
            "video_path": "",
            "extracted_audio_path": "",
            "duration_seconds": 0.0,
            "temporary_video_paths": [],
            "temporary_audio_paths": [],
            "temporary_frame_paths": [],
            "temporary_face_paths": [],
            "temporary_files": [],
            "errors": [
                str(error)
            ],
        }

    # sample a small set of frames from across the video
    try:
        temporary_frame_paths = (
            extract_representative_frames(
                video_path=video_path,
                number_of_frames=number_of_frames,
            )
        )

    except Exception as error:
        add_processing_error(
            errors,
            f"Video frame extraction failed: {error}",
        )

    # extract face crops only when usable frames were created
    if temporary_frame_paths:
        try:
            temporary_face_paths = (
                extract_face_crops(
                    temporary_frame_paths
                )
            )

        except Exception as error:
            add_processing_error(
                errors,
                f"Face extraction failed: {error}",
            )

    extracted_audio_path = ""
    # extract audio for transcription and vocal-pattern analysis
    try:
        extracted_audio_path = (
            extract_video_audio(
                video_path
            )
        )

        temporary_audio_paths.append(
            extracted_audio_path
        )

    except Exception as error:
        add_processing_error(
            errors,
            str(error),
        )
    # keep all temporary paths together so they can be cleaned up later
    temporary_files = (
        temporary_video_paths
        + temporary_audio_paths
        + temporary_frame_paths
        + temporary_face_paths
    )
    return {
        "preparation_complete": True,
        "video_path": video_path,
        "extracted_audio_path": extracted_audio_path,
        "duration_seconds": video_metadata[
            "duration_seconds"
        ],
        "temporary_video_paths": temporary_video_paths,
        "temporary_audio_paths": temporary_audio_paths,
        "temporary_frame_paths": temporary_frame_paths,
        "temporary_face_paths": temporary_face_paths,
        "temporary_files": temporary_files,
        "errors": errors,
    }


# removes all temporary files created for one video reflection
def cleanup_video_result(
    video_result: dict[str, Any],
) -> int:
    #Delete all temporary files tracked by a video result

    if not isinstance(
        video_result,
        dict,
    ):
        return 0
    temporary_files = video_result.get(
        "temporary_files",
        [],
    )
    deleted_count = cleanup_video_files(
        temporary_files
    )

    # clear saved paths after the files have been removed
    video_result["temporary_files"] = []
    video_result["temporary_video_paths"] = []
    video_result["temporary_audio_paths"] = []
    video_result["temporary_frame_paths"] = []
    video_result["temporary_face_paths"] = []

    return deleted_count