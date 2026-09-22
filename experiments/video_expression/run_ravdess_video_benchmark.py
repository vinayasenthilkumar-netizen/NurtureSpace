
# gc is used between models so memory from the previous recogniser can be released
import gc
import random
import shutil
import time
from collections import Counter
from pathlib import Path

# OpenCV handles video frames and MediaPipe is used for face detection/cropping
import cv2
import mediapipe as mp

# plotting and data libraries used for evaluation and report outputs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# HSEmotionONNX provides the candidate facial-expression models being compared
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# standard classification metrics used to compare the five models
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# keep dataset, cached frames and results relative to this experiment script
SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_FOLDER / "data"
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "validation"
FACE_CACHE_FOLDER = SCRIPT_FOLDER / "frame_cache" / "validation"

# create the output folders on the first run
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
FACE_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


# Actors 01-03 are used for model selection only
# Actor 04 is deliberately excluded and kept for the final test
VALIDATION_ACTORS = {"01", "02", "03"}

# None means all available validation videos are used
MAX_VIDEOS = None

# fixed random state keeps any optional sampling reproducible
RANDOM_STATE = 42

# eight representative frames are sampled from each video
FRAMES_PER_VIDEO = 8

# avoid the start and end of the clip and focus on the middle section
START_FRACTION = 0.20
END_FRACTION = 0.80

# turn this on only when cached face crops need to be recreated
REBUILD_FACE_CACHE = False


# five candidate HSEmotionONNX models compared during validation
MODEL_NAMES = [
    "enet_b0_8_best_vgaf",
    "enet_b0_8_best_afew",
    "enet_b0_8_va_mtl",
    "enet_b2_8",
    "enet_b2_7",
]


# convert RAVDESS emotion codes into the labels used in this evaluation
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


# these are the seven target expression classes scored during validation
TARGET_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


# extra categories keep preprocessing failures and unsupported outputs visible
CONFUSION_LABELS = TARGET_LABELS + [
    "other",
    "no_face",
    "error",
]


# create file-safe model names for CSV and image outputs
def safe_filename(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


# read useful information from the coded RAVDESS filename
def parse_ravdess_filename(path):
    parts = path.stem.split("-")

    # normal RAVDESS filenames contain seven coded sections
    if len(parts) != 7:
        return None

    # 02 selects video-only files and 01 selects speech
    if parts[0] != "02" or parts[1] != "01":
        return None

    emotion = RAVDESS_EMOTION_MAP.get(parts[2])
    actor = parts[6]

    # Actor 04 must not enter this validation/model-selection stage
    if not emotion or actor not in VALIDATION_ACTORS:
        return None

    return {
        "video_path": path,
        "filename": path.name,
        "true_label": emotion,
        "emotion_code": parts[2],
        "intensity": parts[3],
        "actor": actor,
    }


# collect all valid validation videos from Actors 01-03
def find_ravdess_videos():
    records = []

    # search recursively because files may be stored inside actor folders
    for path in DATA_FOLDER.rglob("*.mp4"):
        record = parse_ravdess_filename(path)

        if record:
            records.append(record)

    # fixed ordering helps make repeated experiment runs consistent
    records.sort(key=lambda x: str(x["video_path"]))

    if not records:
        raise FileNotFoundError(
            f"No validation videos found in {DATA_FOLDER}"
        )

    # optional limit is useful for quicker development runs
    # the fixed seed keeps the sample reproducible
    if MAX_VIDEOS is not None:
        rng = random.Random(RANDOM_STATE)
        rng.shuffle(records)
        records = records[:MAX_VIDEOS]

    print(f"Validation actors: {sorted(VALIDATION_ACTORS)}")
    print(f"Videos: {len(records)}")

    # show the class balance before model evaluation begins
    counts = Counter(x["true_label"] for x in records)

    for label in TARGET_LABELS:
        print(f"{label}: {counts.get(label, 0)}")

    return records


# crop the largest detected face from one video frame
def crop_main_face(frame_bgr, face_mesh):
    # MediaPipe works with RGB images while OpenCV reads BGR
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        return None

    height, width = frame_rgb.shape[:2]
    best_crop = None
    best_area = 0

    # if more than one face is detected, keep the largest one
    for face in results.multi_face_landmarks:
        xs = [point.x for point in face.landmark]
        ys = [point.y for point in face.landmark]

        # convert normalised landmark positions into pixel coordinates
        x1, x2 = int(min(xs) * width), int(max(xs) * width)
        y1, y2 = int(min(ys) * height), int(max(ys) * height)

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            continue

        # give the crop some space around the detected landmarks
        margin = int(max(face_width, face_height) * 0.20)

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)

        crop = frame_rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        area = crop.shape[0] * crop.shape[1]

        if area > best_area:
            best_area = area
            best_crop = crop.copy()

    return best_crop


# sample representative frames and save their face crops to the cache
def extract_face_frames(video_path, cache_folder, face_mesh):
    # rebuilding removes old cached frames before extracting again
    if REBUILD_FACE_CACHE and cache_folder.exists():
        shutil.rmtree(cache_folder)

    cache_folder.mkdir(parents=True, exist_ok=True)

    # reuse previous crops when available so all five models see the same frames
    existing = sorted(cache_folder.glob("frame_*.jpg"))

    if existing:
        return existing

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        return []

    # sample only from the middle section of the clip
    start = int(total_frames * START_FRACTION)
    end = min(
        int(total_frames * END_FRACTION),
        total_frames - 1,
    )

    # evenly spread the eight requested positions across the selected range
    positions = sorted(set(
        np.linspace(
            start,
            end,
            FRAMES_PER_VIDEO,
            dtype=int,
        ).tolist()
    ))

    saved = []

    for number, position in enumerate(positions, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = capture.read()

        if not success:
            continue

        # frames without a detected face are simply not included
        face = crop_main_face(frame, face_mesh)

        if face is None:
            continue

        output = cache_folder / f"frame_{number:02d}.jpg"

        # convert RGB face crop back to BGR before saving with OpenCV
        cv2.imwrite(
            str(output),
            cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
        )

        saved.append(output)

    capture.release()
    return saved


# prepare one common face-frame cache before comparing any models
def prepare_face_cache(records):
    print("\nPreparing face frames...")

    prepared = []
    rows = []

    # one FaceMesh object is reused instead of recreating it for every frame
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        for index, record in enumerate(records, start=1):
            cache_folder = (
                FACE_CACHE_FOLDER
                / record["video_path"].stem
            )

            face_paths = extract_face_frames(
                record["video_path"],
                cache_folder,
                face_mesh,
            )

            # add the cached face paths to the original video information
            item = dict(record)
            item["face_paths"] = face_paths
            prepared.append(item)

            # keep a separate record of how many usable faces each video produced
            rows.append({
                "Filename": record["filename"],
                "Actor": record["actor"],
                "True Label": record["true_label"],
                "Usable Face Frames": len(face_paths),
            })

            if index % 10 == 0 or index == len(records):
                print(f"Prepared {index}/{len(records)}")

    pd.DataFrame(rows).to_csv(
        RESULTS_FOLDER / "video_validation_face_cache.csv",
        index=False,
    )

    return prepared


# standardise slightly different label names returned by the candidate models
def normalise_model_label(label):
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "happiness": "happy",
        "happy": "happy",
        "sadness": "sad",
        "sad": "sad",
        "fear": "fearful",
        "fearful": "fearful",
        "disgust": "disgust",
        "surprise": "surprised",
        "surprised": "surprised",
        "neutral": "neutral",

        # contempt is not one of the seven target RAVDESS labels
        "contempt": "other",
    }

    return mapping.get(
        str(label).strip().lower(),
        "other",
    )


# load the cached JPG face crops back into RGB arrays for inference
def load_face_images(paths):
    images = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is not None:
            images.append(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )
            )
    return images


# save a confusion matrix for each candidate model
def save_confusion(model_name, true_labels, predicted_labels):
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=CONFUSION_LABELS,
    )

    name = safe_filename(model_name)

    # numerical matrix is kept so exact counts can be inspected
    pd.DataFrame(
        matrix,
        index=[f"True {x}" for x in CONFUSION_LABELS],
        columns=[f"Pred {x}" for x in CONFUSION_LABELS],
    ).to_csv(
        RESULTS_FOLDER / f"confusion_matrix_{name}.csv"
    )

    # image version is easier to use in the report
    fig, ax = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CONFUSION_LABELS,
    )

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=45,
    )

    ax.set_title(
        f"Video Validation Confusion Matrix - {model_name}"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / f"confusion_matrix_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run one candidate model over the complete validation dataset
def evaluate_model(model_name, records):
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    # measure model-loading time separately from inference time
    load_start = time.perf_counter()
    recogniser = HSEmotionRecognizer(model_name=model_name)
    load_time = time.perf_counter() - load_start

    rows = []
    total_inference = 0
    successful = 0

    for index, record in enumerate(records, start=1):

        # no_face is the default if there were no usable crops
        predicted = "no_face"
        original_label = ""
        top_probability = None
        frame_agreement = None
        inference_time = 0
        error_message = ""

        images = load_face_images(record["face_paths"])

        if images:
            try:
                start = time.perf_counter()

                # process all usable face frames from this video together
                frame_labels, scores = recogniser.predict_multi_emotions(
                    images,
                    logits=False,
                )

                inference_time = time.perf_counter() - start
                total_inference += inference_time
                successful += 1

                # make sure score output is always handled as a 2D array
                scores = np.asarray(
                    scores,
                    dtype=np.float32,
                )

                if scores.ndim == 1:
                    scores = scores.reshape(1, -1)

                # keep only emotion classes known by this recogniser
                class_count = len(recogniser.idx_to_class)
                emotion_scores = scores[:, :class_count]

                # normalise each frame's score vector before averaging
                totals = emotion_scores.sum(
                    axis=1,
                    keepdims=True,
                )

                totals = np.where(
                    totals == 0,
                    1,
                    totals,
                )

                emotion_scores = emotion_scores / totals

                # average evidence across the sampled frames
                average_scores = emotion_scores.mean(axis=0)
                prediction_index = int(
                    np.argmax(average_scores)
                )

                original_label = str(
                    recogniser.idx_to_class[prediction_index]
                )

                predicted = normalise_model_label(
                    original_label
                )

                # keep the winning averaged score for later inspection
                top_probability = float(
                    average_scores[prediction_index]
                )

                # normalise frame-level labels before calculating agreement
                frame_labels = [
                    normalise_model_label(x)
                    for x in frame_labels
                ]

                # agreement gives the proportion of frames matching the final prediction
                frame_agreement = (
                    sum(
                        x == predicted
                        for x in frame_labels
                    )
                    / len(frame_labels)
                )

            except Exception as error:
                # failed inference remains visible rather than dropping the video
                predicted = "error"
                error_message = str(error)

        # keep detailed results for every model/video combination
        rows.append({
            "Model": model_name,
            "Filename": record["filename"],
            "Actor": record["actor"],
            "True Label": record["true_label"],
            "Predicted Label": predicted,
            "Original Model Label": original_label,
            "Correct": record["true_label"] == predicted,
            "Usable Face Frames": len(record["face_paths"]),

            "Frame Agreement": (
                round(frame_agreement, 4)
                if frame_agreement is not None
                else None
            ),

            "Top Probability": (
                round(top_probability, 4)
                if top_probability is not None
                else None
            ),

            "Inference Time (s)": round(
                inference_time,
                4,
            ),

            "Processing Error": error_message,
        })

        if index % 10 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    predictions = pd.DataFrame(rows)

    true_labels = predictions["True Label"].tolist()
    predicted_labels = predictions["Predicted Label"].tolist()

    # overall accuracy counts no_face, error and other as wrong predictions
    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    # macro F1 gives every target expression equal importance
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=TARGET_LABELS,
        average="macro",
        zero_division=0,
    )

    # weighted F1 also reflects the number of examples in each class
    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=TARGET_LABELS,
        average="weighted",
        zero_division=0,
    )

    # full per-class report is saved for deeper comparison
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    name = safe_filename(model_name)

    predictions.to_csv(
        RESULTS_FOLDER / f"predictions_{name}.csv",
        index=False,
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_FOLDER / f"classification_report_{name}.csv"
    )

    save_confusion(
        model_name,
        true_labels,
        predicted_labels,
    )

    # average only the videos where frame-agreement could be calculated
    agreement = predictions["Frame Agreement"].dropna()

    summary = {
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),

        # average inference time excludes videos where inference never ran
        "Average Inference Time (s)": round(
            total_inference / successful
            if successful
            else 0,
            4,
        ),

        "Total Inference Time (s)": round(
            total_inference,
            2,
        ),

        "Model Load Time (s)": round(
            load_time,
            2,
        ),

        "Average Frame Agreement": round(
            agreement.mean()
            if not agreement.empty
            else 0,
            4,
        ),

        # these make preprocessing and unsupported predictions visible
        "No Face Videos": int(
            (
                predictions["Predicted Label"]
                == "no_face"
            ).sum()
        ),
        "Failed Videos": int(
            (
                predictions["Predicted Label"]
                == "error"
            ).sum()
        ),
        "Other Predictions": int(
            (
                predictions["Predicted Label"]
                == "other"
            ).sum()
        ),

        "Status": "Completed",
    }

    # print the main metrics immediately after each model finishes
    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Avg time: {summary['Average Inference Time (s)']:.4f}s"
    )

    # remove the model before loading the next candidate
    del recogniser
    gc.collect()

    return summary, rows


# compare accuracy and both F1 measures across completed models
def save_metric_graph(df):
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - width,
        df["Accuracy"],
        width,
        label="Accuracy",
    )

    ax.bar(
        x,
        df["Macro F1"],
        width,
        label="Macro F1",
    )

    ax.bar(
        x + width,
        df["Weighted F1"],
        width,
        label="Weighted F1",
    )

    ax.set_title(
        "Video Expression Model Performance - Validation"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Model"],
        rotation=25,
        ha="right",
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "video_validation_metric_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare the runtime cost of the five candidate models
def save_time_graph(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        df["Model"],
        df["Average Inference Time (s)"],
    )

    ax.set_title(
        "Video Expression Model Inference Time"
    )
    ax.set_ylabel("Seconds per video")
    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "video_validation_inference_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save a compact report-friendly table of the main validation metrics
def save_results_table(df):
    # only the most useful comparison fields are included in the image
    table_df = df[
        [
            "Model",
            "Accuracy",
            "Macro F1",
            "Weighted F1",
            "Average Inference Time (s)",
        ]
    ].copy()

    # shorten long headings so the figure stays readable
    table_df.columns = [
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Avg Time (s)",
    ]

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title(
        "Video Expression Model Validation Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "video_validation_scores_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save the validation-set class balance as supporting evidence
def save_class_distribution(records):
    counts = Counter(
        x["true_label"]
        for x in records
    )

    values = [
        counts.get(label, 0)
        for label in TARGET_LABELS
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        TARGET_LABELS,
        values,
    )

    ax.set_title(
        "RAVDESS Video Validation Class Distribution"
    )
    ax.set_ylabel("Videos")
    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "video_validation_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run face preparation once, then evaluate every candidate model on the same data
def main():
    # collect only Actors 01-03 for model selection
    records = find_ravdess_videos()

    # save class balance before comparing any models
    save_class_distribution(records)

    # every candidate model uses exactly the same cached face frames
    prepared = prepare_face_cache(records)

    summaries = []
    all_predictions = []

    # evaluate the five candidate models one at a time
    for model_name in MODEL_NAMES:
        try:
            summary, predictions = evaluate_model(
                model_name,
                prepared,
            )

        except Exception as error:
            # keep failed models in the comparison instead of silently removing them
            print(f"{model_name} failed: {error}")

            summary = {
                "Model": model_name,
                "Accuracy": None,
                "Macro F1": None,
                "Weighted F1": None,
                "Average Inference Time (s)": None,
                "Total Inference Time (s)": None,
                "Model Load Time (s)": None,
                "Average Frame Agreement": None,
                "No Face Videos": None,
                "Failed Videos": None,
                "Other Predictions": None,
                "Status": f"Failed: {error}",
            }

            predictions = []

        summaries.append(summary)
        all_predictions.extend(predictions)

        # clear unused Python objects before loading the next model
        gc.collect()

    comparison = pd.DataFrame(summaries)

    # prioritise Macro F1, then Accuracy, then lower inference time
    # failed models naturally move to the bottom
    comparison = comparison.sort_values(
        ["Macro F1", "Accuracy", "Average Inference Time (s)"],
        ascending=[False, False, True],
        na_position="last",
    )

    # one compact file compares all candidate models
    comparison.to_csv(
        RESULTS_FOLDER / "video_validation_model_comparison.csv",
        index=False,
    )

    # one larger file keeps every prediction from every successful model
    pd.DataFrame(all_predictions).to_csv(
        RESULTS_FOLDER / "video_validation_all_predictions.csv",
        index=False,
    )

    # report figures should use only models that completed successfully
    completed = comparison[
        comparison["Status"] == "Completed"
    ].copy()

    if not completed.empty:
        save_metric_graph(completed)
        save_time_graph(completed)
        save_results_table(completed)

    print("\n" + "=" * 60)
    print("VIDEO VALIDATION RESULTS")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print(f"\nResults saved in:\n{RESULTS_FOLDER}")


# run the experiment only when this script is executed directly
if __name__ == "__main__":
    main()