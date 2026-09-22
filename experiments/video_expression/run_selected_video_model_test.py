
# used for rebuilding the saved face-frame cache when required
import shutil
import time
from collections import Counter
from pathlib import Path

# computer vision and face-landmark libraries used for frame processing
import cv2
import mediapipe as mp

# plotting and data libraries used for evaluation outputs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# selected facial-expression model used for the final held-out test
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# classification metrics used to report the final model performance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# keep the dataset, cache and result files relative to this script
SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_FOLDER / "data"
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "test"
CACHE_FOLDER = SCRIPT_FOLDER / "frame_cache" / "test"

# create output folders if this is the first run
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


# final model and held-out actor chosen before this test
SELECTED_MODEL = "enet_b0_8_va_mtl"
TEST_ACTOR = "04"

# sample eight frames from the middle 60% of each video
FRAMES_PER_VIDEO = 8
START_FRACTION = 0.20
END_FRACTION = 0.80

# set this to True only when the saved face crops need to be rebuilt
REBUILD_FACE_CACHE = False


# map RAVDESS emotion codes to the labels used in this evaluation
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


# target classes included in the accuracy and F1 evaluation
TARGET_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


# extra labels make failures and unsupported outputs visible in the confusion matrix
CONFUSION_LABELS = TARGET_LABELS + [
    "other",
    "no_face",
    "error",
]


# find only the RAVDESS videos belonging to the held-out test actor
def find_test_videos():
    records = []

    # search through the local RAVDESS data folder
    for path in DATA_FOLDER.rglob("*.mp4"):
        parts = path.stem.split("-")

        # standard RAVDESS filenames contain seven coded parts
        if len(parts) != 7:
            continue

        # 02 means video-only and 01 means speech
        if parts[0] != "02" or parts[1] != "01":
            continue

        # this final test must use Actor 04 only
        if parts[6] != TEST_ACTOR:
            continue

        emotion = RAVDESS_EMOTION_MAP.get(parts[2])

        # skip files with an emotion code outside the expected mapping
        if not emotion:
            continue

        records.append({
            "video_path": path,
            "filename": path.name,
            "actor": parts[6],
            "true_label": emotion,
        })

    # fixed ordering makes repeated runs easier to compare
    records.sort(key=lambda x: str(x["video_path"]))

    if not records:
        raise FileNotFoundError(
            f"No Actor {TEST_ACTOR} videos found in {DATA_FOLDER}"
        )

    # show basic test-set information before processing starts
    print(f"Test actor: {TEST_ACTOR}")
    print(f"Videos: {len(records)}")

    counts = Counter(x["true_label"] for x in records)

    # print the number of examples available for each target class
    for label in TARGET_LABELS:
        print(f"{label}: {counts.get(label, 0)}")

    return records


# use MediaPipe landmarks to crop the main visible face from one frame
def crop_main_face(frame, face_mesh):

    # MediaPipe expects RGB while OpenCV reads frames as BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    height, width = rgb.shape[:2]
    best_crop = None
    best_area = 0

    # normally only one face is expected, but choose the largest if more are found
    for face in results.multi_face_landmarks:
        xs = [point.x for point in face.landmark]
        ys = [point.y for point in face.landmark]

        # convert normalised landmark positions back into pixel coordinates
        x1, x2 = int(min(xs) * width), int(max(xs) * width)
        y1, y2 = int(min(ys) * height), int(max(ys) * height)

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            continue

        # add a small margin so the crop is not too tight around the landmarks
        margin = int(max(face_width, face_height) * 0.20)

        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(width, x2 + margin), min(height, y2 + margin)

        crop = rgb[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        area = crop.shape[0] * crop.shape[1]

        # keep the largest valid face crop found in the frame
        if area > best_area:
            best_crop = crop.copy()
            best_area = area

    return best_crop


# sample frames from a video, crop the face and save those crops to the local cache
def extract_face_frames(video_path, cache_folder, face_mesh):
    # optional rebuild is useful if preprocessing settings have changed
    if REBUILD_FACE_CACHE and cache_folder.exists():
        shutil.rmtree(cache_folder)

    cache_folder.mkdir(parents=True, exist_ok=True)

    # reuse previously generated crops to avoid repeating face detection
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

    # avoid the very beginning and end of the clip where expression evidence may be weaker
    start = int(total_frames * START_FRACTION)
    end = min(
        int(total_frames * END_FRACTION),
        total_frames - 1,
    )

    # spread the requested frames evenly through the selected part of the video
    positions = sorted(set(
        np.linspace(
            start,
            end,
            FRAMES_PER_VIDEO,
            dtype=int,
        ).tolist()
    ))

    paths = []

    for number, position in enumerate(positions, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        success, frame = capture.read()

        if not success:
            continue

        # only frames with a usable face are kept
        face = crop_main_face(frame, face_mesh)

        if face is None:
            continue

        output = cache_folder / f"frame_{number:02d}.jpg"

        # convert back to BGR before saving with OpenCV
        cv2.imwrite(
            str(output),
            cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
        )

        paths.append(output)

    capture.release()
    return paths


# prepare the cached face images for all held-out test videos
def prepare_faces(records):

    print("\nPreparing face frames...")

    prepared = []
    rows = []

    # one MediaPipe FaceMesh instance is reused across the full dataset
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:

        for index, record in enumerate(records, start=1):
            folder = CACHE_FOLDER / record["video_path"].stem

            face_paths = extract_face_frames(
                record["video_path"],
                folder,
                face_mesh,
            )

            # keep the original video information together with its usable face frames
            item = dict(record)
            item["face_paths"] = face_paths
            prepared.append(item)

            # save this separately so missing-face cases can be checked later
            rows.append({
                "Filename": record["filename"],
                "Actor": record["actor"],
                "True Label": record["true_label"],
                "Usable Face Frames": len(face_paths),
            })

            if index % 10 == 0 or index == len(records):
                print(f"Prepared {index}/{len(records)}")

    pd.DataFrame(rows).to_csv(
        RESULTS_FOLDER / "video_test_face_cache.csv",
        index=False,
    )

    return prepared


# load the cached face images into RGB arrays before model inference
def load_images(paths):
    images = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is not None:
            images.append(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

    return images


# make the model's class names match the labels used in the RAVDESS test
def normalise_label(label):

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

        # contempt is outside the target classes used here
        "contempt": "other",
    }

    return mapping.get(
        str(label).strip().lower(),
        "other",
    )


# run the selected expression model on all prepared Actor 04 videos
def evaluate_model(records):
    print(f"\nLoading {SELECTED_MODEL}...")

    # record model loading separately from per-video inference time
    load_start = time.perf_counter()
    recogniser = HSEmotionRecognizer(
        model_name=SELECTED_MODEL
    )
    load_time = time.perf_counter() - load_start

    rows = []
    total_inference = 0
    successful = 0

    for index, record in enumerate(records, start=1):

        # default to no_face until usable face images are available
        predicted = "no_face"
        original_label = ""
        probability = None
        agreement = None
        inference_time = 0
        error_message = ""

        images = load_images(record["face_paths"])

        if images:
            try:
                start = time.perf_counter()

                # predict all sampled face frames in one call
                frame_labels, scores = recogniser.predict_multi_emotions(
                    images,
                    logits=False,
                )

                inference_time = time.perf_counter() - start
                total_inference += inference_time
                successful += 1

                # convert output to a consistent 2D array
                scores = np.asarray(
                    scores,
                    dtype=np.float32,
                )

                if scores.ndim == 1:
                    scores = scores.reshape(1, -1)

                # keep only the actual emotion classes returned by the recogniser
                class_count = len(recogniser.idx_to_class)
                emotion_scores = scores[:, :class_count]

                # normalise each frame's values before averaging them
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

                predicted = normalise_label(
                    original_label
                )

                # store the average confidence of the final selected class
                probability = float(
                    average_scores[prediction_index]
                )

                # normalise each frame label so agreement uses the same label names
                frame_labels = [
                    normalise_label(label)
                    for label in frame_labels
                ]

                # agreement shows how many frames support the final video prediction
                agreement = (
                    sum(
                        label == predicted
                        for label in frame_labels
                    )
                    / len(frame_labels)
                )

            except Exception as error:
                # failed inference stays visible as its own prediction category
                predicted = "error"
                error_message = str(error)

        # keep one row per test video for later error analysis
        rows.append({
            "Filename": record["filename"],
            "Actor": record["actor"],
            "True Label": record["true_label"],
            "Predicted Label": predicted,
            "Original Model Label": original_label,
            "Correct": record["true_label"] == predicted,
            "Usable Face Frames": len(record["face_paths"]),

            "Frame Agreement": (
                round(agreement, 4)
                if agreement is not None
                else None
            ),

            "Top Probability": (
                round(probability, 4)
                if probability is not None
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

    true_labels = predictions[
        "True Label"
    ].tolist()

    predicted_labels = predictions[
        "Predicted Label"
    ].tolist()

    # overall accuracy includes no_face, error and other as incorrect predictions
    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    # macro F1 gives equal importance to each target expression
    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=TARGET_LABELS,
        average="macro",
        zero_division=0,
    )

    # weighted F1 also considers the number of examples in each class
    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=TARGET_LABELS,
        average="weighted",
        zero_division=0,
    )

    # average only across videos where model inference actually ran
    average_time = (
        total_inference / successful
        if successful
        else 0
    )

    agreement_values = predictions[
        "Frame Agreement"
    ].dropna()

    # collect the main final-test results into one summary row
    summary = pd.DataFrame([{
        "Model": SELECTED_MODEL,
        "Test Actor": TEST_ACTOR,
        "Test Videos": len(records),
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Average Inference Time (s)": round(
            average_time,
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

        # this gives an idea of consistency across the sampled frames
        "Average Frame Agreement": round(
            agreement_values.mean()
            if not agreement_values.empty
            else 0,
            4,
        ),

        # keep preprocessing and inference failures visible in the final results
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
    }])

    return summary, predictions


# save the confusion matrix in both numerical and image form
def save_confusion(predictions):

    matrix = confusion_matrix(
        predictions["True Label"],
        predictions["Predicted Label"],
        labels=CONFUSION_LABELS,
    )

    # CSV keeps the exact counts available for the report
    pd.DataFrame(
        matrix,
        index=[
            f"True {x}"
            for x in CONFUSION_LABELS
        ],
        columns=[
            f"Pred {x}"
            for x in CONFUSION_LABELS
        ],
    ).to_csv(
        RESULTS_FOLDER
        / "selected_video_confusion_matrix.csv"
    )

    # create a report-friendly confusion matrix image
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
        "Selected Video Model - Test Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "selected_video_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save the precision, recall and F1 results for each expression class
def save_class_metrics(predictions):

    report = classification_report(
        predictions["True Label"],
        predictions["Predicted Label"],
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    # full classification report is kept as CSV
    report_df.to_csv(
        RESULTS_FOLDER
        / "selected_video_classification_report.csv"
    )

    # only the main class metrics are needed for the bar chart
    class_df = report_df.loc[
        TARGET_LABELS,
        ["precision", "recall", "f1-score"],
    ]

    ax = class_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Selected Video Model - Per-Class Performance"
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Expression")
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis="x",
        rotation=25,
    )
    ax.legend([
        "Precision",
        "Recall",
        "F1",
    ])

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "selected_video_class_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# make a compact image table containing the headline final-test results
def save_summary_table(summary):
    # keep only the metrics that are most useful for the report figure
    table_df = summary[
        [
            "Model",
            "Test Videos",
            "Accuracy",
            "Macro F1",
            "Weighted F1",
            "Average Inference Time (s)",
        ]
    ].copy()

    # shorter column headings make the figure easier to read
    table_df.columns = [
        "Model",
        "Videos",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Avg Time (s)",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 2.5)
    )

    # the figure contains only a table, so normal axes are hidden
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
        "Selected Video Model - Test Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "selected_video_scores_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# show how many Actor 04 videos belong to each RAVDESS expression class
def save_class_distribution(records):
    counts = Counter(
        x["true_label"]
        for x in records
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        TARGET_LABELS,
        [
            counts.get(label, 0)
            for label in TARGET_LABELS
        ],
    )

    ax.set_title(
        "RAVDESS Video Test Class Distribution"
    )
    ax.set_ylabel("Videos")
    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "video_test_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the complete held-out evaluation and save all evidence files
def main():

    print("=" * 60)
    print("SELECTED VIDEO MODEL - TEST")
    print("=" * 60)

    # first identify the held-out Actor 04 videos
    records = find_test_videos()

    # save the test-set class balance before running the model
    save_class_distribution(records)

    # extract or reuse cached face crops for each video
    prepared = prepare_faces(records)

    # run the selected model and calculate the final metrics
    summary, predictions = evaluate_model(prepared)

    # headline final-test results
    summary.to_csv(
        RESULTS_FOLDER
        / "selected_video_model_results.csv",
        index=False,
    )

    # one row per video for detailed checking
    predictions.to_csv(
        RESULTS_FOLDER
        / "selected_video_predictions.csv",
        index=False,
    )

    # keep incorrect cases separately for error analysis
    predictions[
        predictions["Correct"] == False
    ].to_csv(
        RESULTS_FOLDER
        / "selected_video_errors.csv",
        index=False,
    )

    # save the main report figures and class-level evidence
    save_confusion(predictions)
    save_class_metrics(predictions)
    save_summary_table(summary)

    print("\n" + "=" * 60)
    print("FINAL VIDEO TEST RESULTS")
    print("=" * 60)

    print(
        summary.to_string(index=False)
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the test only when this file is executed directly
if __name__ == "__main__":
    main()