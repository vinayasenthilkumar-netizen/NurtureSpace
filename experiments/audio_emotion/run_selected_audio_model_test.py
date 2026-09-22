
#Khoa/w2v-speech-emotion-recognition

# time is used to measure inference speed
from pathlib import Path
import time

# plotting and result tables
import matplotlib.pyplot as plt
import pandas as pd

# torch checks whether a cuda gpu is available
import torch

# standard classification metrics for the final model test
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Hugging Face pipeline runs the selected speech  emotion model
from transformers import pipeline


# experiment and dataset folders
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent.parent
RAVDESS_FOLDER = PROJECT_FOLDER / "ravdess_audio"
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "test"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


# Actors 05 and 06 were kept separate for the final held out test
TEST_ACTORS = {"05", "06"}

# final model selected before this test
SELECTED_MODEL = "Khoa/w2v-speech-emotion-recognition"
MODEL_NAME = "Wav2Vec2 Khoa SER"


# six classes used by the application evaluation
TARGET_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
]

# other keeps unsupported or failed predictions visible in the matrix
CONFUSION_LABELS = TARGET_LABELS + ["other"]


# RAVDESS stores emotion labels as numeric codes in the filename
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


# read the actor and emotion from one RAVDESS filename
def get_ravdess_info(audio_path):
    parts = audio_path.stem.split("-")
    # valid RAVDESS filenames contain seven sections
    if len(parts) != 7:
        return None
    emotion = RAVDESS_EMOTION_MAP.get(parts[2])
    actor = parts[6]
    # calm and surprised are not part of this six class evaluation
    if emotion not in TARGET_LABELS:
        return None

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path),
        "actor": actor,
        "true_label": emotion,
    }


# collect only the recordings belonging to held out Actors 05 and 06
def collect_test_files():
    if not RAVDESS_FOLDER.exists():
        raise FileNotFoundError(
            f"RAVDESS folder not found: {RAVDESS_FOLDER}"
        )

    rows = []

    for audio_path in RAVDESS_FOLDER.rglob("*.wav"):
        info = get_ravdess_info(audio_path)

        if info and info["actor"] in TEST_ACTORS:
            rows.append(info)

    if not rows:
        raise FileNotFoundError(
            "No usable test recordings were found."
        )

    # sorting makes the saved test file easier to inspect
    return (
        pd.DataFrame(rows)
        .sort_values(["actor", "file_name"])
        .reset_index(drop=True)
    )


# convert the models generic LABEL values into readable emotion names
def normalise_audio_label(raw_label):
    label = (
        str(raw_label or "")
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )

    label_map = {
        "label 0": "sad",
        "label 1": "angry",
        "label 2": "disgust",
        "label 3": "fearful",
        "label 4": "happy",
        "label 5": "neutral",
    }

    # unknown labels remain visible as other
    return label_map.get(label, "other")


# take the highest scoring prediction from the Hugging Face output
def get_prediction(output):
    if isinstance(output, dict):
        predictions = [output]

    elif isinstance(output, list):
        if output and isinstance(output[0], list):
            predictions = output[0]
        elif output and isinstance(output[0], dict):
            predictions = output
        else:
            return "other", "", 0.0

    else:
        return "other", "", 0.0

    if not predictions:
        return "other", "", 0.0

    # choose the label with the largest confidence score
    top = max(
        predictions,
        key=lambda item: item.get("score", 0),
    )

    raw_label = top.get("label", "")
    confidence = float(top.get("score", 0.0))

    return (
        normalise_audio_label(raw_label),
        raw_label,
        confidence,
    )


# run the selected model over every held out recording
def evaluate_model(classifier, data):
    y_true = []
    y_pred = []
    rows = []
    errors = []

    start = time.perf_counter()
    for index, row in data.iterrows():
        try:
            # request all class scores so the strongest prediction can be selected
            output = classifier(
                row["file_path"],
                top_k=None,
            )
            predicted, raw_label, confidence = get_prediction(
                output
            )
            processing_error = ""
        except Exception as error:
            # failed recordings stay in the evaluation as other
            predicted = "other"
            raw_label = ""
            confidence = 0.0
            processing_error = str(error)

        true_label = row["true_label"]

        y_true.append(true_label)
        y_pred.append(predicted)

        # keep one detailed row for every audio file
        result = {
            "File": row["file_name"],
            "Actor": row["actor"],
            "True Label": true_label,
            "Predicted Label": predicted,
            "Raw Model Label": raw_label,
            "Confidence": round(confidence, 4),
            "Correct": predicted == true_label,
            "Processing Error": processing_error,
        }
        rows.append(result)

        # incorrect cases are saved separately for error analysis
        if predicted != true_label:
            errors.append(
                result.copy()
            )
        # print occasional progress during the test
        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(data)}"
            )

    total_time = time.perf_counter() - start

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        # Macro F1 gives each emotion class equal weight
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=TARGET_LABELS,
            average="macro",
            zero_division=0,
        ),

        # Weighted F1 also considers how many samples each class contains
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=TARGET_LABELS,
            average="weighted",
            zero_division=0,
        ),

        "average_time": total_time / len(data),
        "total_time": total_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "rows": rows,
        "errors": errors,
    }
# save the confusion matrix in both csv and image form
def save_confusion_matrix(result):
    matrix = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=CONFUSION_LABELS,
    )

    pd.DataFrame(
        matrix,
        index=[f"True {x}" for x in CONFUSION_LABELS],
        columns=[f"Pred {x}" for x in CONFUSION_LABELS],
    ).to_csv(
        RESULTS_FOLDER / "selected_audio_model_confusion_matrix.csv"
    )

    # image version is easier to include in the report
    fig, ax = plt.subplots(figsize=(8, 7))

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
        "Selected Audio Model - Test Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_audio_model_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save per class precision, recall and F1
def save_class_metrics(result):
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        RESULTS_FOLDER / "selected_audio_model_classification_report.csv"
    )

    # only the six target classes are shown in the graph
    class_df = report_df.loc[
        TARGET_LABELS,
        ["precision", "recall", "f1-score"],
    ]

    ax = class_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Selected Audio Model - Per-Class Performance"
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Emotion")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(["Precision", "Recall", "F1"])

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_audio_model_class_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# show how many held out recordings belong to each emotion class
def save_class_distribution(data):
    counts = (
        data["true_label"]
        .value_counts()
        .reindex(TARGET_LABELS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        counts.index,
        counts.values,
    )
    ax.set_title(
        "RAVDESS Audio Test Class Distribution"
    )
    ax.set_ylabel("Recordings")
    ax.set_xlabel("Emotion")
    fig.tight_layout()
    fig.savefig(
        RESULTS_FOLDER / "audio_test_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# save the headline held out test results
def save_summary(result, test_size):
    summary_df = pd.DataFrame([{
        "Model": MODEL_NAME,
        "Model ID": SELECTED_MODEL,
        "Test Actors": "05, 06",
        "Test Recordings": test_size,
        "Accuracy": round(result["accuracy"], 4),
        "Macro F1": round(result["macro_f1"], 4),
        "Weighted F1": round(result["weighted_f1"], 4),
        "Average Inference Time (s)": round(result["average_time"], 4),
        "Total Inference Time (s)": round(result["total_time"], 2),
        "Errors": len(result["errors"]),
    }])

    summary_df.to_csv(
        RESULTS_FOLDER / "selected_audio_model_results.csv",
        index=False,
    )

    # smaller table used as a report ready image
    table_df = summary_df[[
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Average Inference Time (s)",
        "Errors",
    ]].copy()

    table_df.columns = [
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Avg Time (s)",
        "Errors",
    ]

    fig, ax = plt.subplots(figsize=(10, 2.5))
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
        "Selected Audio Model - Test Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_audio_model_scores_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return summary_df


# run the complete final audio model test
def main():
    data = collect_test_files()

    print("=" * 60)
    print("SELECTED AUDIO MODEL - TEST")
    print("=" * 60)

    print(
        f"Actors: {', '.join(sorted(TEST_ACTORS))}"
    )
    print(
        f"Recordings: {len(data)}"
    )

    print("\nClass distribution:")
    print(
        data["true_label"]
        .value_counts()
        .reindex(TARGET_LABELS, fill_value=0)
    )

    # save the exact held out files used for this test
    data.to_csv(
        RESULTS_FOLDER / "audio_test_files.csv",
        index=False,
    )

    save_class_distribution(
        data
    )

    # Hugging Face uses gpu index 0 for cuda and -1 for cpu
    device = 0 if torch.cuda.is_available() else -1

    print(
        f"\nLoading {SELECTED_MODEL}..."
    )

    # the same final selected model is used for all held out recordings
    classifier = pipeline(
        "audio-classification",
        model=SELECTED_MODEL,
        device=device,
    )

    result = evaluate_model(
        classifier,
        data,
    )

    # save every prediction for traceability
    pd.DataFrame(
        result["rows"]
    ).to_csv(
        RESULTS_FOLDER / "selected_audio_model_predictions.csv",
        index=False,
    )

    # keep only incorrect predictions in a separate error file
    pd.DataFrame(
        result["errors"]
    ).to_csv(
        RESULTS_FOLDER / "selected_audio_model_errors.csv",
        index=False,
    )

    save_confusion_matrix(
        result
    )

    save_class_metrics(
        result
    )

    summary_df = save_summary(
        result,
        len(data),
    )
    print("\n" + "=" * 60)
    print("FINAL AUDIO TEST RESULTS")
    print("=" * 60)
    print(
        summary_df.to_string(
            index=False
        )
    )
    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the test only when this script is executed directly
if __name__ == "__main__":
    main()