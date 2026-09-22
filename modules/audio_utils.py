
# pathlib is used for temporary audio file paths and checks
from pathlib import Path
# creates unique names for temporary audio files
from uuid import uuid4
# pydub is used to open, validate and convert audio files
from pydub import AudioSegment
# imports the temporary folder and audio preparation settings
from config.settings import (
    TEMP_DATA_DIR,
    MAX_RECORDING_SECONDS,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
)

# audio formats accepted for uploads and browser recordings
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".m4a",
    ".ogg",
    ".webm",
    ".flac",
    ".aac",
}


# creates the temporary media folder if it doesnt exist exist
def ensure_temp_directory():
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DATA_DIR

# checks that the supplied audio uses a supported file extension 
def validate_audio_extension(file_name):
    suffix = Path(str(file_name or "")).suffix.lower()
    # browser recordings tend to come without an extension
    if not suffix:
        return ".wav"
    # stop unsupported formats before attempting audio processing
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format. Please use one of: {supported}."
        )
    return suffix


# reads audio content from uploaded files or raw byte values
def read_audio_bytes(audio_file):
    if audio_file is None:
        raise ValueError("No audio recording or upload was provided.")
    # raw bytes can be used directly
    if isinstance(audio_file, (bytes, bytearray)):
        audio_bytes = bytes(audio_file)
    # some uploaded file objects provide getvalue()
    elif hasattr(audio_file, "getvalue"):
        audio_bytes = audio_file.getvalue()
    # otherwise use the normal file read method
    elif hasattr(audio_file, "read"):
        # move to the beginning before reading when possible
        if hasattr(audio_file, "seek"):
            audio_file.seek(0)
        audio_bytes = audio_file.read()
        # reset the file position so the object can still be reused
        if hasattr(audio_file, "seek"):
            audio_file.seek(0)
    else:
        raise ValueError("The supplied audio input could not be read.")

    # empty recordings should not be saved or processed
    if not audio_bytes:
        raise ValueError("The supplied audio file is empty.")
    return audio_bytes


# saves the original uploaded or recorded audio into the temp folder
def save_audio_to_temp_file(audio_file):
    # keep the original extension when a filename is available
    uploaded_name = (
        getattr(audio_file, "filename", None)
        or getattr(audio_file, "name", None)
        or "recording.wav"
    )
    suffix = validate_audio_extension(uploaded_name)
    audio_bytes = read_audio_bytes(audio_file)
    # use a unique filename so multiple recordings do not overwrite each other
    temp_directory = ensure_temp_directory()
    temp_path = temp_directory / f"voice_original_{uuid4().hex}{suffix}"
    temp_path.write_bytes(audio_bytes)
    return str(temp_path)

#audio prep
# opens and validates an audio file before conversion
def load_audio_segment(audio_path):
   #open audio file using pydub
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    if not path.is_file():
        raise ValueError("The supplied audio path is not a file.")
    # make sure the stored file still has a supported extension
    validate_audio_extension(path.name)

    try:
        # pydub use ffmpeg for formats for decoding
        return AudioSegment.from_file(str(path))

    except Exception as error:
        raise ValueError(
            "The audio file could not be opened. "
            "It may be damaged or use an unsupported encoding."
        ) from error


# validates duration and converts audio into the format used by the models
def prepare_audio_file(audio_path):
    #validate the audio and convert it to 16000 Hz mono
    audio_segment = load_audio_segment(audio_path)
    # pydub stores duration in milliseconds
    duration_seconds = round(len(audio_segment) / 1000, 2)

    if duration_seconds <= 0:
        raise ValueError(
            "The audio recording does not contain usable audio."
        )

    # recordings longer than the application limit are rejected
    if duration_seconds > MAX_RECORDING_SECONDS:
        raise ValueError(
            "The audio recording is too long. "
            f"The maximum duration is {MAX_RECORDING_SECONDS} seconds."
        )

    # standardise the audio before transcription and vocal-pattern analysis
    prepared_audio = (
        audio_segment
        .set_channels(AUDIO_CHANNELS)
        .set_frame_rate(AUDIO_SAMPLE_RATE)
        .set_sample_width(2)
    )

    # save the prepared version separately from the original upload
    temp_directory = ensure_temp_directory()
    prepared_path = temp_directory / f"voice_prepared_{uuid4().hex}.wav"

    prepared_audio.export(str(prepared_path), format="wav")

    return {
        "prepared_path": str(prepared_path),
        "duration_seconds": duration_seconds,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
    }

# checks that a file belongs to the application's own temporary folder
def is_temporary_audio_path(file_path):
    if not file_path:
        return False
    temp_root = ensure_temp_directory().resolve()
    candidate_path = Path(file_path).resolve()
    # prevents cleanup code from deleting files outside the temp folder
    return candidate_path.is_relative_to(temp_root)


# safely removes one tracked temporary audio file
def delete_temp_file(file_path):
    if not file_path:
        return False
    path = Path(file_path)
    # never delete a file unless it belongs to the application temp directory
    if not is_temporary_audio_path(path):
        return False

    if not path.exists() or not path.is_file():
        return False
    path.unlink()
    return True


# removes several temporary files and reports how many were deleted
def delete_temp_files(file_paths):
    deleted_count = 0
    # cleanup continues even if some paths are empty or already missing
    for file_path in file_paths or []:
        if delete_temp_file(file_path):
            deleted_count += 1
    return deleted_count