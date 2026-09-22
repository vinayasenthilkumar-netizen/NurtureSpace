# gc helps release each model before the next one is loaded
import gc
import time
from pathlib import Path

# plotting and result tables
import matplotlib.pyplot as plt
import pandas as pd

# torch checks whether a cuda gpu is available
import torch

# standard metrics used to compare the candidate models
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Hugging Face pipeline loads each pretrained audio model
from transformers import pipeline


# main experiment folders
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent.parent
RAVDESS_FOLDER = PROJECT_FOLDER / "ravdess_audio"
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "validation"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


# only Actors 01 to 04 are used during model selection
VALIDATION_ACTORS = {"01", "02", "03", "04"}
# six shared classes used to compare models with different label names
TARGET_LABELS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
]

# other keeps unknown or failed predictions visible in the matrix
CONFUSION_LABELS = TARGET_LABELS + ["other"]
# RAVDESS stores emotion as a number inside each filename
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


# five candidate speech emotion models compared on the same validation set
AUDIO_MODELS = {
    "Wav2Vec2 Base SUPERB": "superb/wav2vec2-base-superb-er",
    "HuBERT Base SUPERB": "superb/hubert-base-superb-er",
    "Wav2Vec2 Large SUPERB": "superb/wav2vec2-large-superb-er",
    "Wav2Vec2 Khoa SER": "Khoa/w2v-speech-emotion-recognition",
    "AST CREMA-D": "forwarder1121/ast-finetuned-model",
}


# turn the display name into a simple filename
def safe_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


# read the actor and emotion codes from one RAVDESS filename
def get_ravdess_info(audio_path):
    parts = audio_path.stem.split("-")

    # valid RAVDESS filenames contain seven coded sections
    if len(parts) != 7:
        return None

    emotion = RAVDESS_EMOTION_MAP.get(parts[2])
    actor = parts[6]

    # calm and surprised are outside this six class evaluation
    if emotion not in TARGET_LABELS:
        return None

    return {
        "file_name": audio_path.name,
        "file_path": str(audio_path),
        "actor": actor,
        "true_label": emotion,
    }


# collect only recordings from the validation actors
def collect_validation_files():
    if not RAVDESS_FOLDER.exists():
        raise FileNotFoundError(
            f"RAVDESS folder not found: {RAVDESS_FOLDER}"
        )
    rows = []
    for audio_path in RAVDESS_FOLDER.rglob("*.wav"):
        info = get_ravdess_info(audio_path)

        if not info:
            continue

        # Actors 05 to 06 remain untouched for the final test
        if info["actor"] not in VALIDATION_ACTORS:
            continue

        rows.append(info)

    if not rows:
        raise FileNotFoundError(
            "No usable validation recordings were found."
        )

    # sorting makes the saved validation set easier to inspect
    return (
        pd.DataFrame(rows)
        .sort_values(["actor", "file_name"])
        .reset_index(drop=True)
    )


# map different model label names into the same six emotion classes
def normalise_audio_label(raw_label):
    label = (
        str(raw_label or "")
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )

    # the selected Khoa model uses generic LABEL numbers
    khoa_map = {
        "label 0": "sad",
        "label 1": "angry",
        "label 2": "disgust",
        "label 3": "fearful",
        "label 4": "happy",
        "label 5": "neutral",
    }

    if label in khoa_map:
        return khoa_map[label]

    # other candidate models use names or short label forms
    mappings = {
        "neutral": "neutral",
        "neu": "neutral",
        "neutrality": "neutral",
        "calm": "neutral",

        "happy": "happy",
        "hap": "happy",
        "happiness": "happy",
        "joy": "happy",
        "positive": "happy",

        "sad": "sad",
        "sadness": "sad",

        "angry": "angry",
        "ang": "angry",
        "anger": "angry",

        "fear": "fearful",
        "fearful": "fearful",
        "fea": "fearful",

        "disgust": "disgust",
        "dis": "disgust",
        "disgusted": "disgust",
    }

    # unsupported labels stay visible rather than being silently dropped
    return mappings.get(label, "other")
# extract the highest scoring prediction from different pipeline output shapes
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

    # choose the model label with the highest confidence
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


# run one model on every validation recording
def evaluate_model(classifier, data):
    y_true = []
    y_pred = []
    rows = []
    errors = []

    start = time.perf_counter()

    for index, row in data.iterrows():
        try:
            # request all class scores before choosing the strongest label
            output = classifier(
                row["file_path"],
                top_k=None,
            )

            predicted, raw_label, confidence = get_prediction(output)
            processing_error = ""

        except Exception as error:
            # failed files stay in the evaluation as other
            predicted = "other"
            raw_label = ""
            confidence = 0.0
            processing_error = str(error)

        true_label = row["true_label"]

        y_true.append(true_label)
        y_pred.append(predicted)

        # keep one detailed row for every recording
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
            errors.append(result.copy())

        # print occasional progress during long model runs
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(data)}")

    total_time = time.perf_counter() - start

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        # Macro F1 gives each emotion equal importance
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=TARGET_LABELS,
            average="macro",
            zero_division=0,
        ),

        # Weighted F1 also reflects the number of samples per class
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


# save the detailed evidence for one candidate model
def save_model_results(model_name, result):
    """Save one model's detailed results."""

    name = safe_filename(model_name)

    # save every prediction
    pd.DataFrame(result["rows"]).to_csv(
        RESULTS_FOLDER / f"predictions_{name}.csv",
        index=False,
    )

    # save mistakes separately for easier inspection
    pd.DataFrame(result["errors"]).to_csv(
        RESULTS_FOLDER / f"errors_{name}.csv",
        index=False,
    )

    # precision, recall and F1 for each target class
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=TARGET_LABELS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        RESULTS_FOLDER / f"classification_report_{name}.csv"
    )

    # include other so unsupported predictions remain visible
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
        RESULTS_FOLDER / f"confusion_matrix_{name}.csv"
    )

    # save an image version for report use
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
        f"Audio Validation Confusion Matrix - {model_name}"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / f"confusion_matrix_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare the three main performance metrics across successful models
def save_metric_graph(df):
    x = list(range(len(df)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        [i - width for i in x],
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
        [i + width for i in x],
        df["Weighted F1"],
        width,
        label="Weighted F1",
    )

    ax.set_title(
        "Audio Emotion Model Performance - Validation"
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
        RESULTS_FOLDER / "audio_validation_metric_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare average inference time across models
def save_time_graph(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        df["Model"],
        df["Average Inference Time (s)"],
    )

    ax.set_title(
        "Audio Model Inference Time - Validation"
    )
    ax.set_ylabel("Seconds per recording")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "audio_validation_inference_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# create a compact comparison table for the report
def save_results_table(df):
    table_df = df[[
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Average Inference Time (s)",
    ]].copy()

    table_df.columns = [
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Avg Time (s)",
    ]

    fig, ax = plt.subplots(figsize=(11, 3.5))
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
        "Audio Emotion Model Validation Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "audio_validation_scores_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save the validation class balance as a simple graph
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
        "RAVDESS Validation Class Distribution"
    )
    ax.set_ylabel("Recordings")
    ax.set_xlabel("Emotion")

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "audio_validation_class_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the full five model validation comparison
def main():
    data = collect_validation_files()

    print("=" * 60)
    print("RAVDESS AUDIO VALIDATION")
    print("=" * 60)

    print(
        f"Actors: {', '.join(sorted(VALIDATION_ACTORS))}"
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

    # save the exact files used during model selection
    data.to_csv(
        RESULTS_FOLDER / "audio_validation_files.csv",
        index=False,
    )

    save_class_distribution(data)

    # Hugging Face uses gpu index 0 when cuda is available, otherwise -1 for cpu
    device = 0 if torch.cuda.is_available() else -1

    comparison_rows = []

    # every candidate is tested on the same Actors 01-04 recordings
    for model_name, model_id in AUDIO_MODELS.items():
        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        classifier = None

        try:
            # keep model loading time separate from inference time
            load_start = time.perf_counter()

            classifier = pipeline(
                "audio-classification",
                model=model_id,
                device=device,
            )

            load_time = time.perf_counter() - load_start

            result = evaluate_model(
                classifier,
                data,
            )

            save_model_results(
                model_name,
                result,
            )

            # one row is kept for the final model-selection table
            comparison_rows.append({
                "Model": model_name,
                "Model ID": model_id,
                "Accuracy": round(result["accuracy"], 4),
                "Macro F1": round(result["macro_f1"], 4),
                "Weighted F1": round(result["weighted_f1"], 4),
                "Average Inference Time (s)": round(
                    result["average_time"], 4
                ),
                "Total Inference Time (s)": round(
                    result["total_time"], 2
                ),
                "Model Load Time (s)": round(load_time, 2),
                "Errors": len(result["errors"]),
                "Status": "Completed",
            })

            print(
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f}"
            )

        except Exception as error:
            # failed models remain visible in the comparison
            print(f"Model failed: {error}")

            comparison_rows.append({
                "Model": model_name,
                "Model ID": model_id,
                "Accuracy": None,
                "Macro F1": None,
                "Weighted F1": None,
                "Average Inference Time (s)": None,
                "Total Inference Time (s)": None,
                "Model Load Time (s)": None,
                "Errors": None,
                "Status": f"Failed: {error}",
            })

        finally:
            # release the current model before loading the next candidate
            if classifier is not None:
                del classifier

            gc.collect()

            # clear unused gpu memory when cuda is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(comparison_rows)

    # Macro F1 is the main selection measure, with Accuracy as the tie-breaker
    comparison_df = comparison_df.sort_values(
        by=["Macro F1", "Accuracy"],
        ascending=False,
        na_position="last",
    )

    comparison_df.to_csv(
        RESULTS_FOLDER / "audio_validation_model_comparison.csv",
        index=False,
    )

    # only completed models should appear in the graphs
    completed_df = comparison_df[
        comparison_df["Status"] == "Completed"
    ].copy()

    if not completed_df.empty:
        save_metric_graph(completed_df)
        save_time_graph(completed_df)
        save_results_table(completed_df)

    print("\n" + "=" * 60)
    print("FINAL AUDIO VALIDATION RESULTS")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run validation only when this file is executed directly
if __name__ == "__main__":
    main()