
from __future__ import annotations

# argparse is used so validation and final testing stay as separate commands
import argparse
import sys
import time
from pathlib import Path

# plotting and table libraries used for the evaluation outputs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# standard classification metrics used for the final model evaluation
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)


# keep the dataset and generated results relative to this experiment script
SCRIPT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = SCRIPT_FOLDER / "data"
RESULTS_ROOT = SCRIPT_FOLDER / "results" / "deployment_aligned"


# find the main project folder so this experiment can import the real app code
def find_project_root(start: Path) -> Path:

    # check this directory first, then each parent directory
    candidates = [start, *start.parents]

    for candidate in candidates:
        # these folders are enough to identify the main project root
        if (
            (candidate / "modules").is_dir()
            and (candidate / "model_adapters").is_dir()
            and (candidate / "config").is_dir()
        ):
            return candidate

    # fail clearly if the experiment was placed outside the project structure
    raise RuntimeError(
        "Could not find the project root. Place this script inside the "
        "project's video-expression experiment folder and try again."
    )


# add the detected project root so the experiment uses the deployed code
PROJECT_FOLDER = find_project_root(SCRIPT_FOLDER)
sys.path.insert(0, str(PROJECT_FOLDER))


# application settings are imported after the project path has been added
from config.settings import (
    MIN_VIDEO_EXPRESSION_FACES,
    SELECTED_VIDEO_EXPRESSION_MODEL,
    VIDEO_EXPRESSION_SAMPLE_FRAMES,
)

# use the same visible-expression adapter as the Flask application
from model_adapters.video_expression import (
    normalise_raw_expression,
    predict_visible_expression,
)

# use the same frame extraction, face detection and cleanup functions as deployment
from modules.video_utils import (
    cleanup_video_files,
    extract_face_crops,
    extract_representative_frames,
)


# RAVDESS emotion codes are converted to the labels expected by the app adapter
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happiness",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}


# these are the model classes included in the evaluation metrics
TARGET_LABELS = [
    "neutral",
    "happiness",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

# unclear is added only to the confusion matrix because the deployed adapter can abstain
CONFUSION_LABELS = TARGET_LABELS + ["unclear"]


# Actors 01-03 are used for choosing the frame count
VALIDATION_ACTORS = {"01", "02", "03"}

# Actor 04 stays held out until the frame setting has been frozen
TEST_ACTORS = {"04"}

# only these two application frame-count options are compared
FRAME_COUNTS = [5, 8]

# read the RAVDESS filename and keep only videos needed for this experiment
def parse_video(path: Path, allowed_actors: set[str]):

    parts = path.stem.split("-")

    # a valid RAVDESS filename should contain seven coded sections
    if len(parts) != 7:
        return None

    # RAVDESS code 02 is video-only and 01 is speech
    # other modalities are not part of this deployment-aligned evaluation
    if parts[0] != "02" or parts[1] != "01":
        return None

    actor = parts[6]
    true_label = RAVDESS_EMOTION_MAP.get(parts[2])

    # keep only the actors assigned to the current validation/test phase
    if actor not in allowed_actors or true_label is None:
        return None

    return {
        "video_path": path,
        "filename": path.name,
        "actor": actor,
        "true_label": true_label,
    }


# collect all valid videos for either the validation actors or held-out actor
def collect_videos(actors: set[str]) -> list[dict]:
    records = []

    # search recursively because RAVDESS files may be stored inside actor folders
    for path in DATA_FOLDER.rglob("*.mp4"):
        record = parse_video(path, actors)

        if record:
            records.append(record)

    # fixed ordering makes repeated evaluation runs easier to compare
    records.sort(key=lambda item: str(item["video_path"]))

    if not records:
        raise FileNotFoundError(
            f"No matching RAVDESS video-only files were found in {DATA_FOLDER}."
        )

    return records


# run one video through the same visual pipeline used by the application
def evaluate_one_video(record: dict, frame_count: int) -> dict:
    temporary_files: list[str] = []
    start = time.perf_counter()

    try:
        # sample representative frames across the video
        frame_paths = extract_representative_frames(
            video_path=record["video_path"],
            number_of_frames=frame_count,
        )
        temporary_files.extend(frame_paths)

        # run the deployed Haar-based face crop step on those frames
        face_paths = extract_face_crops(frame_paths)
        temporary_files.extend(face_paths)

        # send the usable face crops through the selected expression adapter
        expression = predict_visible_expression(face_paths)

        # normalise the adapter's raw dominant label before scoring it
        dominant = normalise_raw_expression(
            expression.get("dominant_raw_label", "")
        )

        # anything outside the supported evaluation labels is treated as unclear
        predicted = dominant if dominant in TARGET_LABELS else "unclear"

        # keep detailed evidence so accuracy and abstention behaviour can both be inspected
        return {
            "Frame Count": frame_count,
            "Filename": record["filename"],
            "Actor": record["actor"],
            "True Label": record["true_label"],
            "Predicted Label": predicted,
            "Raw Dominant Label": dominant,
            "Correct": predicted == record["true_label"],
            "Requested Frames": frame_count,
            "Extracted Frames": len(frame_paths),
            "Usable Face Crops": len(face_paths),
            "Analysed Face Count": expression.get("analysed_face_count", 0),
            "Enough Face Evidence": bool(
                expression.get("enough_face_evidence", False)
            ),
            "Strict Majority": bool(
                expression.get("has_strict_majority", False)
            ),
            "Clear Result": bool(
                expression.get("clear_result", False)
            ),
            "Frame Agreement": expression.get("frame_agreement", 0.0),
            "Evidence Reason": expression.get("evidence_reason", ""),
            "Adapter Error": expression.get("error", ""),
            "Processing Seconds": round(
                time.perf_counter() - start,
                4,
            ),
        }

    except Exception as error:
        # failed videos are kept in the evaluation instead of silently disappearing
        return {
            "Frame Count": frame_count,
            "Filename": record["filename"],
            "Actor": record["actor"],
            "True Label": record["true_label"],
            "Predicted Label": "unclear",
            "Raw Dominant Label": "",
            "Correct": False,
            "Requested Frames": frame_count,
            "Extracted Frames": 0,
            "Usable Face Crops": 0,
            "Analysed Face Count": 0,
            "Enough Face Evidence": False,
            "Strict Majority": False,
            "Clear Result": False,
            "Frame Agreement": 0.0,
            "Evidence Reason": "Processing failed.",
            "Adapter Error": str(error),
            "Processing Seconds": round(
                time.perf_counter() - start,
                4,
            ),
        }

    finally:
        # frames and face crops are temporary and should not remain after evaluation
        cleanup_video_files(temporary_files)


# evaluate every video using one fixed frame-count setting
def evaluate_setting(records: list[dict], frame_count: int) -> pd.DataFrame:

    rows = []

    for index, record in enumerate(records, start=1):
        row = evaluate_one_video(record, frame_count)
        rows.append(row)

        # print occasional progress without flooding the console
        if index % 10 == 0 or index == len(records):
            print(f"Frames={frame_count}: processed {index}/{len(records)}")

    return pd.DataFrame(rows)


# summarise one complete validation or test run
def summarise(df: pd.DataFrame, phase: str) -> dict:
    y_true = df["True Label"].tolist()
    y_pred = df["Predicted Label"].tolist()

    # raw accuracy counts unclear predictions as incorrect
    raw_accuracy = accuracy_score(y_true, y_pred)

    # macro F1 gives every emotion class equal importance
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        average="macro",
        zero_division=0,
    )

    # weighted F1 also considers how many examples each class contains
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=TARGET_LABELS,
        average="weighted",
        zero_division=0,
    )

    # deployed logic can mark some outputs as unclear
    # this subset lets us separately inspect accuracy when evidence was considered clear
    clear = df[df["Clear Result"] == True].copy()  # noqa: E712

    clear_accuracy = (
        accuracy_score(clear["True Label"], clear["Predicted Label"])
        if not clear.empty
        else None
    )

    return {
        "Phase": phase,
        "Model": SELECTED_VIDEO_EXPRESSION_MODEL,
        "Frame Count": int(df["Frame Count"].iloc[0]),
        "Videos": len(df),
        "Raw Dominant Accuracy": round(raw_accuracy, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),

        # evidence rates show how often the deployment rules had enough support
        "Sufficient Face Evidence Rate": round(
            df["Enough Face Evidence"].mean(),
            4,
        ),
        "Strict Majority Rate": round(
            df["Strict Majority"].mean(),
            4,
        ),
        "Clear Result Rate": round(
            df["Clear Result"].mean(),
            4,
        ),

        # accuracy among only the outputs accepted as clear
        "Accuracy Among Clear Results": (
            round(clear_accuracy, 4)
            if clear_accuracy is not None
            else None
        ),

        # these help compare whether 8 frames gives useful evidence for the extra cost
        "Average Usable Face Crops": round(
            df["Usable Face Crops"].mean(),
            3,
        ),
        "Average Frame Agreement": round(
            df["Frame Agreement"].mean(),
            4,
        ),
        "Average Processing Seconds": round(
            df["Processing Seconds"].mean(),
            4,
        ),
        "Median Processing Seconds": round(
            df["Processing Seconds"].median(),
            4,
        ),

        # keep processing failures visible in the final evidence
        "Processing Failures": int(
            (df["Adapter Error"].fillna("") != "").sum()
        ),
    }

# save both CSV and image versions of the confusion matrix
def save_confusion(
    df: pd.DataFrame,
    output_folder: Path,
    suffix: str,
) -> None:
    # unclear is included because the deployed adapter is allowed to abstain
    matrix = confusion_matrix(
        df["True Label"],
        df["Predicted Label"],
        labels=CONFUSION_LABELS,
    )

    # CSV is useful for exact numerical reporting
    pd.DataFrame(
        matrix,
        index=[f"True {label}" for label in CONFUSION_LABELS],
        columns=[f"Pred {label}" for label in CONFUSION_LABELS],
    ).to_csv(
        output_folder / f"video_confusion_{suffix}.csv"
    )

    # image version is easier to include in the report
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
        f"Deployment-aligned Video Confusion Matrix - {suffix}"
    )

    fig.tight_layout()
    fig.savefig(
        output_folder / f"video_confusion_{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# create simple graphs comparing the two validation frame counts
def save_validation_graphs(
    summary: pd.DataFrame,
    output_folder: Path,
) -> None:

    x = np.arange(len(summary))
    width = 0.22

    # first graph compares accuracy, macro F1 and how often a clear result was produced
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width,
        summary["Raw Dominant Accuracy"],
        width,
        label="Accuracy",
    )
    ax.bar(
        x,
        summary["Macro F1"],
        width,
        label="Macro F1",
    )
    ax.bar(
        x + width,
        summary["Clear Result Rate"],
        width,
        label="Clear-result rate",
    )

    ax.set_title("5 vs 8 Frames - Deployment-aligned Validation")
    ax.set_ylabel("Score / rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{value} frames" for value in summary["Frame Count"]]
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_folder / "video_frame_count_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # second graph focuses on the runtime cost of processing more frames
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        [f"{value} frames" for value in summary["Frame Count"]],
        summary["Average Processing Seconds"],
    )

    ax.set_title("5 vs 8 Frames - Visual Processing Time")
    ax.set_ylabel("Average seconds per video")

    fig.tight_layout()
    fig.savefig(
        output_folder / "video_frame_count_processing_time.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# run the model-selection phase using validation actors only
def run_compare() -> None:

    output_folder = RESULTS_ROOT / "validation"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Actor 04 is deliberately not loaded in this phase
    records = collect_videos(VALIDATION_ACTORS)

    print("=" * 68)
    print("VIDEO FRAME-COUNT SENSITIVITY - VALIDATION ONLY")
    print("=" * 68)
    print(f"Model: {SELECTED_VIDEO_EXPRESSION_MODEL}")
    print(f"Validation actors: {sorted(VALIDATION_ACTORS)}")
    print(f"Videos: {len(records)}")
    print(f"App's current frame count: {VIDEO_EXPRESSION_SAMPLE_FRAMES}")
    print(f"Minimum usable faces: {MIN_VIDEO_EXPRESSION_FACES}\n")

    detailed_frames = []
    summary_rows = []

    # both frame counts are allowed here because this is the selection set
    for frame_count in FRAME_COUNTS:
        df = evaluate_setting(records, frame_count)

        detailed_frames.append(df)
        summary_rows.append(
            summarise(df, "Validation")
        )

        # save separate confusion evidence for each frame-count option
        save_confusion(
            df,
            output_folder,
            f"validation_{frame_count}_frames",
        )

    # combine both settings into one detailed output file
    detailed = pd.concat(
        detailed_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values("Frame Count")

    detailed.to_csv(
        output_folder / "video_frame_sensitivity_detailed.csv",
        index=False,
    )

    summary.to_csv(
        output_folder / "video_frame_sensitivity_summary.csv",
        index=False,
    )

    save_validation_graphs(
        summary,
        output_folder,
    )

    print("\n" + "=" * 68)
    print("VALIDATION COMPARISON")
    print("=" * 68)
    print(summary.to_string(index=False))

    # this warning is important because Actor 04 must remain untouched during selection
    print(
        "\nDo not use Actor 04 to choose between frame counts. "
        "Freeze the setting from this validation comparison first."
    )

    print(f"\nResults saved in:\n{output_folder}")


# run the final held-out test using only the already selected frame count
def run_test(frame_count: int) -> None:

    output_folder = RESULTS_ROOT / "test"
    output_folder.mkdir(parents=True, exist_ok=True)

    # only the held-out actor is loaded here
    records = collect_videos(TEST_ACTORS)

    print("=" * 68)
    print("FINAL DEPLOYMENT-ALIGNED VIDEO TEST - HELD-OUT ACTOR 04")
    print("=" * 68)
    print(f"Frozen frame count: {frame_count}")
    print(f"Model: {SELECTED_VIDEO_EXPRESSION_MODEL}")
    print(f"Held-out actors: {sorted(TEST_ACTORS)}")
    print(f"Videos: {len(records)}\n")

    # unlike validation, only one frozen setting is evaluated
    df = evaluate_setting(records, frame_count)

    summary = pd.DataFrame([
        summarise(df, "Held-out Test")
    ])

    # detailed predictions make the final result auditable
    df.to_csv(
        output_folder
        / f"selected_video_deployment_predictions_{frame_count}_frames.csv",
        index=False,
    )

    # compact file contains the headline test metrics
    summary.to_csv(
        output_folder
        / f"selected_video_deployment_results_{frame_count}_frames.csv",
        index=False,
    )

    save_confusion(
        df,
        output_folder,
        f"test_{frame_count}_frames",
    )

    print("\n" + "=" * 68)
    print("HELD-OUT DEPLOYMENT TEST RESULTS")
    print("=" * 68)
    print(summary.to_string(index=False))
    print(f"\nResults saved in:\n{output_folder}")


# define separate command-line modes so validation and test are not mixed accidentally
def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compare 5 vs 8 frames on validation data, "
            "then test one frozen setting."
        )
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    # compare mode does not need a frame argument because it evaluates both options
    subparsers.add_parser(
        "compare",
        help="Compare 5 vs 8 frames using Actors 01-03 only.",
    )

    # test mode requires the chosen frame count explicitly
    test_parser = subparsers.add_parser(
        "test",
        help="Evaluate one already-frozen frame count on held-out Actor 04.",
    )

    test_parser.add_argument(
        "--frames",
        type=int,
        choices=FRAME_COUNTS,
        required=True,
        help="The frame count selected BEFORE inspecting Actor 04.",
    )

    return parser.parse_args()


# send the command to the correct evaluation phase
def main() -> None:

    args = parse_args()

    if args.mode == "compare":
        run_compare()
    else:
        run_test(args.frames)


# only execute the experiment when this file is run directly
if __name__ == "__main__":
    main()