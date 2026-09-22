
# pathlib is used for working with temporary video and image file paths
from pathlib import Path
# creates unique names for temporary media files
from uuid import uuid4
# check whether ffmpeg is available
import shutil
# run ffmpeg commands for conversion and audio extraction
import subprocess
# opencv is used for video reading, frame extraction and face detection
import cv2
# imports recording limits and audio settings used during processing
from config.settings import (
    MAX_RECORDING_SECONDS,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
)
# reuses the common temporary-file helpers from the audio utilities
from modules.audio_utils import ensure_temp_directory, delete_temp_files


# video formats accepted by the application
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}

# checks that the uploaded or recorded video uses a supported file type
def validate_video_extension(file_name):
    # get the file extension in a consistent lowercase form
    suffix = Path(str(file_name or "")).suffix.lower()
    # browser recordings may arrive without a normal filename
    if not suffix:
        return ".mp4"
    # stop unsupported formats before any processing starts
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video format. Please use one of: {supported}."
        )
    return suffix


# reads video content from uploaded files or raw bytes
def read_video_bytes(video_file):

    if video_file is None:
        raise ValueError("No video recording or upload was provided.")

    # raw bytes can be used directly
    if isinstance(video_file, (bytes, bytearray)):
        video_bytes = bytes(video_file)

    # some uploaded file objects provide getvalue()
    elif hasattr(video_file, "getvalue"):
        video_bytes = video_file.getvalue()

    # fall back to the normal file read method
    elif hasattr(video_file, "read"):
        # move to the start before reading when the object supports seek
        if hasattr(video_file, "seek"):
            video_file.seek(0)

        video_bytes = video_file.read()
        # reset the position so the same file object can still be reused
        if hasattr(video_file, "seek"):
            video_file.seek(0)
    else:
        raise ValueError("The supplied video input could not be read.")
    # an empty upload should not be written to the temporary folder
    if not video_bytes:
        raise ValueError("The supplied video file is empty.")
    return video_bytes


# saves an uploaded or recorded video into the temporary media folder
def save_video_to_temp_file(video_file):
    # use the original filename where possible
    uploaded_name = (
        getattr(video_file, "filename", None)
        or getattr(video_file, "name", None)
        or "recording.mp4"
    )

    suffix = validate_video_extension(uploaded_name)
    video_bytes = read_video_bytes(video_file)

    # create a unique temporary path to avoid filename clashes
    temp_directory = ensure_temp_directory()
    video_path = temp_directory / f"video_original_{uuid4().hex}{suffix}"
    video_path.write_bytes(video_bytes)
    return str(video_path)


#video validation 
# reads basic video information using opencv
def get_video_metadata(video_path):

    path = Path(str(video_path or ""))
    if not path.exists():
        raise FileNotFoundError(f"Video file was not found: {path}")
    if not path.is_file():
        raise ValueError("The supplied video path is not a file.")
    # check the extension before attempting to open the video
    validate_video_extension(path.name)
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        raise ValueError(
            "The video file could not be opened. "
            "It may be damaged or use an unsupported encoding."
        )

    try:
        # read the metadata needed for validation and frame sampling
        frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        # always release the video handle
        capture.release()
    if frames_per_second <= 0 or total_frames <= 0:
        raise ValueError("The video duration could not be determined.")
    # duration is calculated from the total frame count and frame rate
    duration_seconds = round(total_frames / frames_per_second, 2)
    return {
        "duration_seconds": duration_seconds,
        "frames_per_second": frames_per_second,
        "total_frames": total_frames,
        "width": width,
        "height": height,
    }


# checks if the video can be processed and is within the time limit
def validate_video_file(video_path):

    metadata = get_video_metadata(video_path)
    duration_seconds = metadata["duration_seconds"]
    # videos without usable duration information should not continue
    if duration_seconds <= 0:
        raise ValueError("The video does not contain usable frames.")
    # recordings are limited to the configured maximum duration
    if duration_seconds > MAX_RECORDING_SECONDS:
        raise ValueError(
            "The video is too long. "
            f"The maximum duration is {MAX_RECORDING_SECONDS} seconds."
        )
    return metadata


# browser recording conversion
# converts browser-recorded video into a more reliable mp4 format
def convert_live_recording_to_mp4(recording_path, preview_path):
    source_path = Path(recording_path)
    output_path = Path(preview_path)
    # check that the original browser recording actually exists
    if not source_path.exists() or not source_path.is_file():
        raise ValueError("The video recording could not be found.")
    if source_path.stat().st_size == 0:
        raise ValueError("The video recording is empty.")
    # ffmpeg is needed because browser recordings may use webm or other codecs
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required to prepare the video.")

    # convert video to h264 and audio to aac for more reliable playback/processing
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]
    # run the conversion without printing ffmpeg output to the console
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
    )
    # remove an incomplete output file when conversion fails
    if result.returncode != 0 or not output_path.exists():
        if output_path.exists():
            output_path.unlink()

        error = (result.stderr or "").strip()
        if error:
            raise ValueError(
                f"The video recording could not be prepared: {error}"
            )
        raise ValueError("The video recording could not be prepared.")
    # also reject a conversion that produced an empty file
    if output_path.stat().st_size == 0:
        output_path.unlink()
        raise ValueError("The converted video file is empty.")
    return str(output_path)


# extracts the videos audio into the format expected by speech and audio models
def extract_video_audio(video_path):
    path = Path(str(video_path or ""))
    if not path.exists():
        raise FileNotFoundError(f"Video file was not found: {path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for video audio extraction.")
    # extracted audio is kept only as a temporary wav file
    temp_directory = ensure_temp_directory()
    audio_path = temp_directory / f"video_audio_{uuid4().hex}.wav"
    # remove the video stream and convert audio to the configured format
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(path),
        "-vn",
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    # extraction is treated as failed if no usable wav file was created
    failed = (
        result.returncode != 0
        or not audio_path.exists()
        or audio_path.stat().st_size == 0
    )
    # failed audio files are removed rather than kept as empty temporary files
    if failed:
        if audio_path.exists():
            audio_path.unlink()

        raise ValueError(
            "No usable audio could be extracted from the video."
        )
    return str(audio_path)



# representative frames
# calculates evenly spaced positions from the middle section of the video
def get_frame_positions(duration_seconds, number_of_frames):

    if number_of_frames <= 0:
        raise ValueError("The number of frames must be greater than zero.")
    # when only one frame is needed, use the middle of the recording
    if number_of_frames == 1:
        return [duration_seconds * 0.5]
    # avoid relying only on the very beginning or end of the recording
    start_fraction = 0.2
    end_fraction = 0.8
    step = (end_fraction - start_fraction) / (number_of_frames - 1)
    # convert each relative position into seconds
    return [
        duration_seconds * (start_fraction + step * index)
        for index in range(number_of_frames)
    ]


# extracts representative frames that can later be used for face detection
def extract_representative_frames(video_path, number_of_frames=3):
    # validation also gives the duration needed for frame positions
    metadata = validate_video_file(video_path)

    positions = get_frame_positions(
        metadata["duration_seconds"],
        number_of_frames,
    )
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise ValueError(
            "The video could not be opened for frame extraction."
        )

    temp_directory = ensure_temp_directory()
    frame_paths = []

    try:
        for position_seconds in positions:
            # move to the chosen position in the video
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                position_seconds * 1000,
            )

            success, frame = capture.read()
            # skip a sample position if opencv cannot read a usable frame
            if not success or frame is None:
                continue
            frame_path = (
                temp_directory
                / f"video_frame_{uuid4().hex}.jpg"
            )
            # only record the path when the image was written successfully
            if cv2.imwrite(str(frame_path), frame):
                frame_paths.append(str(frame_path))

    finally:
        #make sure the video file is released even when extraction fails
        capture.release()
    if not frame_paths:
        raise ValueError(
            "No representative frames could be extracted."
        )
    return frame_paths

#loads opencv's built-in frontal-face detector
def load_face_detector():
    # use the haar cascade included with the opencv installation
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(str(cascade_path))

    if face_detector.empty():
        raise RuntimeError(
            "The OpenCV face detector could not be loaded."
        )
    return face_detector

#adds some extra space around a detected face before saving the crop
def add_crop_margin(
    x,
    y,
    width,
    height,
    frame_width,
    frame_height,
    margin_ratio=0.15,
):
    #calculate the extra space from the size of the detected face
    horizontal_margin = int(width * margin_ratio)
    vertical_margin = int(height * margin_ratio)
    # keep crop coordinates inside the original frame
    start_x = max(0, x - horizontal_margin)
    start_y = max(0, y - vertical_margin)
    end_x = min(frame_width, x + width + horizontal_margin)
    end_y = min(frame_height, y + height + vertical_margin)
    return start_x, start_y, end_x, end_y


#detects faces in extracted frames and saves temporary face crops
def extract_face_crops(frame_paths, maximum_faces_per_frame=1):
    if maximum_faces_per_frame <= 0:
        raise ValueError(
            "The maximum number of faces must be greater than zero."
        )
    face_detector = load_face_detector()
    temp_directory = ensure_temp_directory()
    face_paths = []

    # process each representative frame separately
    for frame_path in frame_paths or []:
        frame = cv2.imread(str(frame_path))
        # skip frames that cannot be read
        if frame is None:
            continue
        # face detection works on a grayscale version of the frame
        grayscale_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        # detect possible frontal faces in the current frame
        detected_faces = face_detector.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        # use the largest detected face first
        sorted_faces = sorted(
            detected_faces,
            key=lambda face: face[2] * face[3],
            reverse=True,
        )
        frame_height, frame_width = frame.shape[:2]
        # normally only the largest face is kept from each sampled frame
        for x, y, width, height in sorted_faces[:maximum_faces_per_frame]:
            start_x, start_y, end_x, end_y = add_crop_margin(
                x=x,
                y=y,
                width=width,
                height=height,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            face_crop = frame[start_y:end_y, start_x:end_x]
            # ignore invalid crop coordinates that produce an empty image
            if face_crop.size == 0:
                continue
            face_path = (
                temp_directory
                / f"video_face_{uuid4().hex}.jpg"
            )
            # keep only face crops that were successfully written to disk
            if cv2.imwrite(str(face_path), face_crop):
                face_paths.append(str(face_path))
    return face_paths


# removes the temporary video, frame, face and audio files after processing
def cleanup_video_files(file_paths):
    # use the shared cleanup helper so temporary media is removed consistently
    return delete_temp_files(file_paths)