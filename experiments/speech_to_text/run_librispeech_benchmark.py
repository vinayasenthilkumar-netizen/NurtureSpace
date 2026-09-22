
# gc is used to clear model objects between Whisper runs
import gc
import random
import re
import time
from pathlib import Path

# plotting and dataframe libraries used for the evaluation outputs
import matplotlib.pyplot as plt
import pandas as pd
import torch
import whisper

# jiwer provides the standard WER and CER calculations
from jiwer import wer, cer


# keep dataset and result paths relative to this experiment folder
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent.parent

# local LibriSpeech test-clean data used for this benchmark
LIBRISPEECH_FOLDER = (
    PROJECT_FOLDER
    / "librispeech_data"
    / "LibriSpeech"
    / "test-clean"
)

# validation outputs are kept separate from the later held-out test results
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "validation"

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# fixed values make the speaker split and sample reproducible
RANDOM_STATE = 42

# 32 speakers are used for model selection
VALIDATION_SPEAKER_COUNT = 32

# all five models are compared on the same 100 recordings
SAMPLE_SIZE = 100


# candidate Whisper models compared before selecting the final one
WHISPER_MODELS = [
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "turbo",
]


# clean transcripts in the same way before calculating WER and CER
def normalise_text(text):
    text = str(text or "").lower()

    # remove punctuation while keeping letters, numbers and apostrophes
    text = re.sub(
        r"[^a-z0-9'\s]",
        " ",
        text,
    )

    # reduce repeated whitespace to a single space
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# create simple filenames from Whisper model names such as small.en
def make_safe_filename(model_name):
    return (
        model_name
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


# collect LibriSpeech audio files together with their reference transcripts
def collect_librispeech_files():
    if not LIBRISPEECH_FOLDER.exists():
        raise FileNotFoundError(
            f"LibriSpeech folder not found: "
            f"{LIBRISPEECH_FOLDER}"
        )

    rows = []

    # each transcript file contains multiple recording IDs and transcripts
    for transcript_file in LIBRISPEECH_FOLDER.rglob("*.trans.txt"):
        with open(
            transcript_file,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(" ", 1)

                if len(parts) != 2:
                    continue

                recording_id = parts[0]
                transcript = parts[1]

                # LibriSpeech audio uses the same recording ID as the transcript line
                audio_path = (
                    transcript_file.parent
                    / f"{recording_id}.flac"
                )

                if not audio_path.exists():
                    continue

                # the first section of the ID identifies the speaker
                speaker_id = recording_id.split("-")[0]

                rows.append({
                    "recording_id": recording_id,
                    "speaker_id": speaker_id,
                    "audio_path": str(audio_path),
                    "reference_transcript": transcript,
                    "normalised_reference": normalise_text(transcript),
                })

    if not rows:
        raise FileNotFoundError(
            "No matching LibriSpeech files found."
        )

    return pd.DataFrame(rows)


# split speakers before sampling recordings so validation and test speakers do not overlap
def create_speaker_split(dataset_df):
    speakers = sorted(
        dataset_df["speaker_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    # at least one speaker must remain outside the validation group
    if len(speakers) <= VALIDATION_SPEAKER_COUNT:
        raise ValueError(
            "Not enough speakers to create "
            "validation and test splits."
        )

    # fixed shuffle keeps the same speaker split across repeated runs
    rng = random.Random(RANDOM_STATE)
    rng.shuffle(speakers)

    validation_speakers = speakers[:VALIDATION_SPEAKER_COUNT]
    test_speakers = speakers[VALIDATION_SPEAKER_COUNT:]

    return (
        validation_speakers,
        test_speakers,
    )


# take one fixed recording sample from validation speakers only
def create_validation_sample(
    dataset_df,
    validation_speakers,
):
    # remove all recordings belonging to reserved test speakers
    validation_df = dataset_df[
        dataset_df["speaker_id"].isin(
            validation_speakers
        )
    ].copy()

    # avoid asking for more samples than are available
    sample_size = min(
        SAMPLE_SIZE,
        len(validation_df),
    )

    # every Whisper model receives this exact same fixed sample
    sample_df = validation_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    return sample_df.reset_index(drop=True)


# load and evaluate one Whisper model on the shared validation sample
def evaluate_model(
    model_name,
    sample_df,
    device,
):
    print("\n" + "=" * 60)
    print(f"Whisper model: {model_name}")
    print("=" * 60)

    model = None

    try:
        # measure loading separately from transcription time
        load_start = time.perf_counter()

        model = whisper.load_model(
            model_name,
            device=device,
        )

        load_time = (
            time.perf_counter()
            - load_start
        )

    except Exception as error:
        # keep failed models in the comparison rather than removing them
        return {
            "summary": {
                "Model": model_name,
                "Status": f"Load failed: {error}",
                "WER": None,
                "CER": None,
                "Average Transcription Time (s)": None,
                "Total Transcription Time (s)": None,
                "Model Load Time (s)": None,
                "Failed Files": None,
            },
            "predictions": [],
            "model": model,
        }

    references = []
    predictions = []
    rows = []
    failed_files = 0

    total_start = time.perf_counter()

    for index, row in sample_df.iterrows():
        start = time.perf_counter()
        error_message = ""

        try:
            # all models use the same deterministic transcription settings
            result = model.transcribe(
                row["audio_path"],
                language="en",
                task="transcribe",
                fp16=(device == "cuda"),
                verbose=False,
                temperature=0,
                condition_on_previous_text=False,
            )

            predicted_text = result.get(
                "text",
                "",
            )

        except Exception as error:
            # failed recordings remain part of the evidence
            predicted_text = ""
            error_message = str(error)
            failed_files += 1

        transcription_time = (
            time.perf_counter()
            - start
        )

        reference = row[
            "normalised_reference"
        ]

        # clean Whisper output using exactly the same rules as the reference
        prediction = normalise_text(
            predicted_text
        )

        # calculate error rates for this individual recording
        file_wer = wer(
            reference,
            prediction,
        )

        file_cer = cer(
            reference,
            prediction,
        )

        references.append(reference)
        predictions.append(prediction)

        # keep detailed evidence for each recording and model
        rows.append({
            "Model": model_name,
            "Recording ID": row["recording_id"],
            "Speaker ID": row["speaker_id"],
            "Reference Transcript": row["reference_transcript"],
            "Predicted Transcript": predicted_text.strip(),
            "WER": round(file_wer, 4),
            "CER": round(file_cer, 4),
            "Transcription Time (s)": round(
                transcription_time,
                4,
            ),
            "Processing Error": error_message,
        })

        # occasional progress output is useful because Whisper evaluation can take time
        if (
            (index + 1) % 10 == 0
            or index + 1 == len(sample_df)
        ):
            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(sample_df)}"
            )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # overall WER and CER are calculated across all transcripts together
    overall_wer = wer(
        " ".join(references),
        " ".join(predictions),
    )

    overall_cer = cer(
        " ".join(references),
        " ".join(predictions),
    )

    average_time = (
        total_time
        / len(sample_df)
    )

    print(
        f"WER: {overall_wer:.4f} | "
        f"CER: {overall_cer:.4f} | "
        f"Avg time: {average_time:.4f}s"
    )

    return {
        "summary": {
            "Model": model_name,
            "Status": "Completed",
            "WER": round(
                overall_wer,
                4,
            ),
            "CER": round(
                overall_cer,
                4,
            ),
            "Average Transcription Time (s)": round(
                average_time,
                4,
            ),
            "Total Transcription Time (s)": round(
                total_time,
                2,
            ),
            "Model Load Time (s)": round(
                load_time,
                2,
            ),
            "Failed Files": failed_files,
        },
        "predictions": rows,
        "model": model,
    }


# compare overall WER and CER across completed models
def save_error_graph(df):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = range(len(df))
    width = 0.35

    # lower bars indicate better transcription accuracy
    ax.bar(
        [i - width / 2 for i in x],
        df["WER"],
        width,
        label="WER",
    )

    ax.bar(
        [i + width / 2 for i in x],
        df["CER"],
        width,
        label="CER",
    )

    ax.set_title(
        "Whisper Validation Error Rates"
    )

    ax.set_ylabel("Error Rate")
    ax.set_xticks(list(x))

    ax.set_xticklabels(
        df["Model"],
        rotation=25,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "whisper_validation_wer_cer.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare average transcription time for the candidate models
def save_time_graph(df):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        df["Model"],
        df["Average Transcription Time (s)"],
    )

    ax.set_title(
        "Whisper Validation Transcription Time"
    )

    ax.set_ylabel(
        "Seconds per recording"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "whisper_validation_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# create a compact table image showing the main validation metrics
def save_results_table(df):
    table_df = df[[
        "Model",
        "WER",
        "CER",
        "Average Transcription Time (s)",
        "Failed Files",
    ]].copy()

    # shorter headings make the table easier to fit into the report
    table_df.columns = [
        "Model",
        "WER",
        "CER",
        "Avg Time (s)",
        "Failed",
    ]

    fig, ax = plt.subplots(
        figsize=(9, 3)
    )

    # normal chart axes are not needed for a table
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
        "Whisper Model Validation Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "whisper_validation_scores_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# show how WER varies across individual recordings for each model
def save_wer_distribution(
    predictions_df,
):
    if predictions_df.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    model_names = (
        predictions_df["Model"]
        .unique()
        .tolist()
    )

    # one WER array is created for each model
    data = [
        predictions_df[
            predictions_df["Model"]
            == model
        ]["WER"].values
        for model in model_names
    ]

    # boxplots show spread and outliers rather than only the overall average
    ax.boxplot(
        data,
        tick_labels=model_names,
    )

    ax.set_title(
        "Per-Recording WER Distribution"
    )

    ax.set_ylabel("WER")

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER
        / "whisper_validation_wer_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the full speaker-disjoint Whisper validation comparison
def main():
    print(
        "Collecting LibriSpeech files..."
    )

    # first build one dataframe containing all usable test-clean recordings
    dataset_df = collect_librispeech_files()

    print(
        f"Total recordings: "
        f"{len(dataset_df)}"
    )

    print(
        f"Total speakers: "
        f"{dataset_df['speaker_id'].nunique()}"
    )

    # split speakers before creating the recording sample
    # this prevents the same speaker appearing in validation and final testing
    validation_speakers, test_speakers = (
        create_speaker_split(dataset_df)
    )

    # save both speaker groups so the split is documented and reproducible
    pd.DataFrame({
        "Validation Speaker ID":
            validation_speakers
    }).to_csv(
        RESULTS_FOLDER
        / "validation_speakers.csv",
        index=False,
    )

    pd.DataFrame({
        "Reserved Test Speaker ID":
            test_speakers
    }).to_csv(
        RESULTS_FOLDER
        / "reserved_test_speakers.csv",
        index=False,
    )

    # sample recordings only from the 32 validation speakers
    sample_df = create_validation_sample(
        dataset_df,
        validation_speakers,
    )

    # save the exact 100 recordings used for comparing the models
    sample_df.to_csv(
        RESULTS_FOLDER
        / "librispeech_validation_sample.csv",
        index=False,
    )

    print(
        f"Validation speakers: "
        f"{len(validation_speakers)}"
    )

    print(
        f"Reserved test speakers: "
        f"{len(test_speakers)}"
    )

    print(
        f"Validation recordings used: "
        f"{len(sample_df)}"
    )

    # use CUDA if available, otherwise evaluate on CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    comparison_rows = []
    all_predictions = []

    # every candidate Whisper model sees the same speaker-disjoint sample
    for model_name in WHISPER_MODELS:
        result = evaluate_model(
            model_name,
            sample_df,
            device,
        )

        comparison_rows.append(
            result["summary"]
        )

        # save detailed predictions separately for each model
        prediction_df = pd.DataFrame(
            result["predictions"]
        )

        prediction_df.to_csv(
            RESULTS_FOLDER
            / (
                "predictions_"
                f"{make_safe_filename(model_name)}"
                ".csv"
            ),
            index=False,
        )

        # also keep all model predictions together for the WER distribution plot
        all_predictions.extend(
            result["predictions"]
        )

        # release the current Whisper model before loading the next one
        model = result.get("model")

        if model is not None:
            del model

        del result

        gc.collect()

        # clear unused GPU memory between models when CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # lowest WER is placed first because lower transcription error is better
    comparison_df = comparison_df.sort_values(
        by="WER",
        ascending=True,
        na_position="last",
    )

    # save the main five-model comparison
    comparison_df.to_csv(
        RESULTS_FOLDER
        / "whisper_validation_model_comparison.csv",
        index=False,
    )

    # only successfully completed models should appear in comparison graphs
    completed_df = comparison_df[
        comparison_df["Status"]
        == "Completed"
    ].copy()

    predictions_df = pd.DataFrame(
        all_predictions
    )

    if not completed_df.empty:
        save_error_graph(
            completed_df
        )

        save_time_graph(
            completed_df
        )

        save_results_table(
            completed_df
        )

    if not predictions_df.empty:
        save_wer_distribution(
            predictions_df
        )

    print("\n" + "=" * 60)
    print("WHISPER VALIDATION RESULTS")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the validation experiment only when this script is executed directly
if __name__ == "__main__":
    main()