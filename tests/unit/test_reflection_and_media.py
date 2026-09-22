# used to create in-memory audio data for file tests
from io import BytesIO
# used for temporary file paths
from pathlib import Path
# used to create a simple image for expression tests
import numpy as np
# used for exception checks and test fixtures
import pytest
# used to create and inspect wav audio files
from pydub import AudioSegment
# imports the modules being tested
from modules import audio_utils
from modules import video_utils
from services import voice_processing
from services import video_processing
from services import theme_detection
from model_adapters import video_expression


# small set of theme rules used only for testing
RULES = {
    "supportive_factors": [
        {
            "name": "Restful sleep",
            "phrases": ["slept well"],
            "exclude_phrases": []
        },
        {
            "name": "Feeling supported",
            "phrases": ["my sister helped"],
            "exclude_phrases": []
        }
    ],
    "strain_factors": [
        {
            "name": "Sleep difficulty",
            "phrases": ["barely slept"],
            "exclude_phrases": ["slept well"]
        },
        {
            "name": "Feeling overwhelmed",
            "phrases": ["feel overwhelmed"],
            "exclude_phrases": []
        }
    ]
}


# creates a temporary silent wav file with chosen audio settings
def make_wav(path, duration=1000, rate=8000, channels=2):
    audio = AudioSegment.silent(duration=duration, frame_rate=rate)
    audio = audio.set_channels(channels)
    audio.export(str(path), format="wav")


# checks audio extensions, byte reading and safe temporary-file deletion
def test_audio_files(monkeypatch, tmp_path):
    assert audio_utils.validate_audio_extension("voice.wav") == ".wav"
    assert audio_utils.validate_audio_extension("voice.webm") == ".webm"

    # unsupported audio extensions should be rejected
    with pytest.raises(ValueError):
        audio_utils.validate_audio_extension("voice.txt")

    data = BytesIO(b"audio-data")
    assert audio_utils.read_audio_bytes(data) == b"audio-data"

    # empty audio data should not be accepted
    with pytest.raises(ValueError):
        audio_utils.read_audio_bytes(b"")

    temp = tmp_path / "temp"
    temp.mkdir()

    # use the pytest temporary folder instead of the real application folder
    monkeypatch.setattr(audio_utils, "TEMP_DATA_DIR", temp)

    inside = temp / "voice.wav"
    inside.write_bytes(b"temporary")

    # files inside the temporary folder should be deleted
    assert audio_utils.delete_temp_file(inside) is True
    assert inside.exists() is False

    outside = tmp_path / "important.txt"
    outside.write_text("keep", encoding="utf-8")

    # files outside the temporary folder should be protected
    assert audio_utils.delete_temp_file(outside) is False
    assert outside.exists() is True


# checks audio preparation and conversion to the required format
def test_audio_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_utils, "TEMP_DATA_DIR", tmp_path)

    original = tmp_path / "original.wav"
    make_wav(original, duration=1500, rate=8000, channels=2)

    result = audio_utils.prepare_audio_file(original)
    prepared = Path(result["prepared_path"])

    assert prepared.exists()
    assert result["duration_seconds"] == 1.5
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1

    # confirm the actual prepared file is 16 khz mono
    audio = AudioSegment.from_file(str(prepared))
    assert audio.frame_rate == 16000
    assert audio.channels == 1

    # audio longer than the configured limit should be rejected
    monkeypatch.setattr(audio_utils, "MAX_RECORDING_SECONDS", 1)

    with pytest.raises(ValueError, match="maximum duration"):
        audio_utils.prepare_audio_file(original)


# checks the normal voice reflection processing flow
def test_voice(monkeypatch):
    # replace real file and model operations with fixed test results
    monkeypatch.setattr(
        voice_processing,
        "save_audio_to_temp_file",
        lambda audio: "original.wav"
    )

    monkeypatch.setattr(
        voice_processing,
        "prepare_audio_file",
        lambda path: {
            "prepared_path": "prepared.wav",
            "duration_seconds": 12.5,
            "sample_rate": 16000,
            "channels": 1
        }
    )

    monkeypatch.setattr(
        voice_processing,
        "transcribe_audio",
        lambda path: {
            "transcript": "I feel tired but my family supports me.",
            "language": "en",
            "success": True,
            "error": ""
        }
    )

    monkeypatch.setattr(
        voice_processing,
        "predict_audio_observation",
        lambda path: {
            "observation": "Low energy",
            "raw_label": "sadness",
            "score": 0.82,
            "clear_result": True,
            "error": ""
        }
    )

    result = voice_processing.process_voice_reflection(audio_file=object())

    assert result["processing_complete"] is True
    assert result["transcription_success"] is True
    assert result["transcript"] == "I feel tired but my family supports me."
    assert result["audio_observation"]["observation"] == "Low energy"
    assert result["duration_seconds"] == 12.5
    assert result["temporary_files"] == ["original.wav", "prepared.wav"]

    # text and theme analysis should wait until transcript review
    assert result["text_emotion_suggestions"] == []
    assert result["supportive_factor_suggestions"] == []
    assert result["strain_factor_suggestions"] == []


# checks voice preparation and model error handling
def test_voice_errors(monkeypatch):
    deleted = []

    monkeypatch.setattr(
        voice_processing,
        "save_audio_to_temp_file",
        lambda audio: "original.wav"
    )

    # simulate audio preparation failure
    def bad_prepare(path):
        raise ValueError("Audio is too long.")

    monkeypatch.setattr(voice_processing, "prepare_audio_file", bad_prepare)

    # record which temporary files are cleaned up
    monkeypatch.setattr(
        voice_processing,
        "delete_temp_files",
        lambda paths: (deleted.extend(paths) or len(paths))
    )

    failed = voice_processing.process_voice_reflection(audio_file=object())

    assert failed["processing_complete"] is False
    assert deleted == ["original.wav"]

    # allow preparation but simulate transcription and audio-model failure
    monkeypatch.setattr(
        voice_processing,
        "prepare_audio_file",
        lambda path: {
            "prepared_path": "prepared.wav",
            "duration_seconds": 8.0,
            "sample_rate": 16000,
            "channels": 1
        }
    )

    monkeypatch.setattr(
        voice_processing,
        "transcribe_audio",
        lambda path: {
            "transcript": "",
            "language": "en",
            "success": False,
            "error": "No clear speech."
        }
    )

    monkeypatch.setattr(
        voice_processing,
        "predict_audio_observation",
        lambda path: {
            "observation": "Unclear",
            "raw_label": "",
            "score": 0.0,
            "clear_result": False,
            "error": "Audio model unavailable"
        }
    )

    result = voice_processing.process_voice_reflection(audio_file=object())

    assert result["processing_complete"] is True
    assert result["transcription_success"] is False
    assert result["audio_observation"]["clear_result"] is False

    # both model errors should be kept in the returned error list
    assert any("No clear speech" in error for error in result["errors"])
    assert any("Audio model unavailable" in error for error in result["errors"])


# checks supported video formats and maximum duration
def test_video_files(monkeypatch):
    assert video_utils.validate_video_extension("clip.mp4") == ".mp4"
    assert video_utils.validate_video_extension("clip.webm") == ".webm"

    # unsupported video extensions should be rejected
    with pytest.raises(ValueError):
        video_utils.validate_video_extension("clip.txt")

    monkeypatch.setattr(video_utils, "MAX_RECORDING_SECONDS", 60)

    # simulate valid video metadata
    monkeypatch.setattr(
        video_utils,
        "get_video_metadata",
        lambda path: {
            "duration_seconds": 60.0,
            "frames_per_second": 25.0,
            "total_frames": 1500,
            "width": 640,
            "height": 480
        }
    )

    valid = video_utils.validate_video_file("clip.mp4")
    assert valid["duration_seconds"] == 60.0

    # simulate a video that is one second too long
    monkeypatch.setattr(
        video_utils,
        "get_video_metadata",
        lambda path: {
            "duration_seconds": 61.0,
            "frames_per_second": 25.0,
            "total_frames": 1525,
            "width": 640,
            "height": 480
        }
    )

    with pytest.raises(ValueError, match="maximum duration"):
        video_utils.validate_video_file("clip.mp4")


# checks the normal video preparation flow
def test_video_prepare(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        video_processing,
        "save_video_to_temp_file",
        lambda video: "clip.webm"
    )

    # record the source and converted video paths
    def convert(recording_path, preview_path):
        seen["source"] = recording_path
        seen["mp4"] = preview_path
        return preview_path

    monkeypatch.setattr(
        video_processing,
        "convert_live_recording_to_mp4",
        convert
    )

    # return fixed metadata after validation
    def validate(path):
        seen["validated"] = path
        return {
            "duration_seconds": 20.0,
            "frames_per_second": 25.0,
            "total_frames": 500,
            "width": 1280,
            "height": 720
        }

    monkeypatch.setattr(video_processing, "validate_video_file", validate)

    monkeypatch.setattr(
        video_processing,
        "extract_representative_frames",
        lambda video_path, number_of_frames: [
            "frame1.jpg", "frame2.jpg", "frame3.jpg"
        ]
    )

    monkeypatch.setattr(
        video_processing,
        "extract_face_crops",
        lambda paths: ["face1.jpg", "face2.jpg", "face3.jpg"]
    )

    monkeypatch.setattr(
        video_processing,
        "extract_video_audio",
        lambda path: "audio.wav"
    )

    result = video_processing.prepare_video_media(video_file=object())

    assert seen["source"] == "clip.webm"
    assert seen["mp4"] == "clip.mp4"
    assert seen["validated"] == "clip.mp4"

    assert result["preparation_complete"] is True
    assert result["video_path"] == "clip.mp4"
    assert result["temporary_video_paths"] == ["clip.webm", "clip.mp4"]
    assert result["temporary_face_paths"] == [
        "face1.jpg", "face2.jpg", "face3.jpg"
    ]
    assert result["extracted_audio_path"] == "audio.wav"


# checks partial video-processing failures
def test_video_error(monkeypatch):
    monkeypatch.setattr(
        video_processing,
        "save_video_to_temp_file",
        lambda video: "clip.mp4"
    )

    monkeypatch.setattr(
        video_processing,
        "validate_video_file",
        lambda path: {"duration_seconds": 15.0}
    )

    # simulate failure while extracting representative frames
    def bad_frames(video_path, number_of_frames):
        raise ValueError("Frames unavailable")

    monkeypatch.setattr(
        video_processing,
        "extract_representative_frames",
        bad_frames
    )

    monkeypatch.setattr(
        video_processing,
        "extract_video_audio",
        lambda path: "audio.wav"
    )

    result = video_processing.prepare_video_media(video_file=object())

    # audio processing can continue even if frame extraction fails
    assert result["preparation_complete"] is True
    assert result["temporary_frame_paths"] == []
    assert result["temporary_face_paths"] == []
    assert result["extracted_audio_path"] == "audio.wav"
    assert any("frame" in error.lower() for error in result["errors"])

    # now allow a frame but simulate failure while extracting audio
    monkeypatch.setattr(
        video_processing,
        "extract_representative_frames",
        lambda video_path, number_of_frames: ["frame.jpg"]
    )

    monkeypatch.setattr(
        video_processing,
        "extract_face_crops",
        lambda paths: []
    )

    def bad_audio(path):
        raise ValueError("No usable audio")

    monkeypatch.setattr(
        video_processing,
        "extract_video_audio",
        bad_audio
    )

    result = video_processing.prepare_video_media(video_file=object())

    assert result["preparation_complete"] is True
    assert result["extracted_audio_path"] == ""
    assert result["temporary_audio_paths"] == []
    assert any("No usable audio" in error for error in result["errors"])


# checks that all temporary video files are cleaned up correctly
def test_video_cleanup(monkeypatch):
    result = {
        "temporary_video_paths": ["clip.webm", "clip.mp4"],
        "temporary_audio_paths": ["audio.wav"],
        "temporary_frame_paths": ["frame.jpg"],
        "temporary_face_paths": ["face.jpg"],
        "temporary_files": [
            "clip.webm", "clip.mp4", "audio.wav", "frame.jpg", "face.jpg"
        ]
    }

    cleaned = []

    # record the files passed to the cleanup function
    monkeypatch.setattr(
        video_processing,
        "cleanup_video_files",
        lambda paths: (cleaned.extend(paths) or len(paths))
    )

    count = video_processing.cleanup_video_result(result)

    assert count == 5
    assert cleaned == [
        "clip.webm", "clip.mp4", "audio.wav", "frame.jpg", "face.jpg"
    ]

    # stored temporary paths should be cleared after cleanup
    assert result["temporary_files"] == []
    assert result["temporary_video_paths"] == []
    assert result["temporary_audio_paths"] == []
    assert result["temporary_frame_paths"] == []
    assert result["temporary_face_paths"] == []

    # no result means there is nothing to clean
    assert video_processing.cleanup_video_result(None) == 0


# checks visible-expression evidence and minimum face requirements
def test_expression(monkeypatch):
    # create a simple image to act as a loaded face crop
    image = np.zeros((48, 48, 3), dtype=np.uint8)

    monkeypatch.setattr(
        video_expression,
        "load_face_crop",
        lambda path: image.copy()
    )

    # small fake model used to return controlled expression labels
    class Model:
        values = []

        def predict_multi_emotions(self, images, logits):
            return self.values, None

    model = Model()

    monkeypatch.setattr(
        video_expression,
        "load_video_expression_model",
        lambda: model
    )

    # five predictions with a clear neutral majority
    model.values = [
        "Neutral", "Neutral", "Surprise", "Neutral", "Sadness"
    ]

    clear = video_expression.predict_visible_expression(
        ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg", "f5.jpg"]
    )
    assert clear["analysed_face_count"] == 5
    assert clear["dominant_prediction_count"] == 3
    assert clear["required_majority_count"] == 3
    assert clear["clear_result"] is True

    # two usable faces are below the minimum evidence requirement
    model.values = ["Surprise", "Surprise"]

    weak = video_expression.predict_visible_expression(
        ["f1.jpg", "f2.jpg"]
    )

    assert weak["analysed_face_count"] == 2
    assert weak["clear_result"] is False

    # no loaded model should also return an unclear result
    monkeypatch.setattr(
        video_expression,
        "load_video_expression_model",
        lambda: None
    )
    missing = video_expression.predict_visible_expression(["f1.jpg"])
    assert missing["clear_result"] is False


# checks rule-based supportive and strain factor detection
def test_themes(monkeypatch):
    # use the small fixed rule set instead of the project json file
    monkeypatch.setattr(
        theme_detection,
        "load_theme_rules",
        lambda: RULES
    )

    result = theme_detection.detect_practical_themes(
        "I barely slept and feel overwhelmed, "
        "but my sister helped with the baby."
    )

    # collect only the standard factor names returned by the detector
    supportive = [
        item["name"]
        for item in result["supportive_factors"]
    ]

    strain = [
        item["name"]
        for item in result["strain_factors"]
    ]

    assert supportive == ["Feeling supported"]
    assert strain == ["Sleep difficulty", "Feeling overwhelmed"]
    assert result["theme_found"] is True

    # an exclusion phrase should prevent sleep difficulty from being matched
    positive = theme_detection.detect_practical_themes(
        "I slept well last night."
    )
    assert positive["strain_factors"] == []

    # empty reflection text should return no detected theme
    empty = theme_detection.detect_practical_themes("")
    assert empty["theme_found"] is False
    assert empty["no_significant_theme_detected"] is True