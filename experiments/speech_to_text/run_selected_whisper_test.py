
# re is used to clean transcripts before WER and CER are calculated
import re
import time
from pathlib import Path

# plotting and table libraries used for the final evaluation outputs
import matplotlib.pyplot as plt
import pandas as pd
import torch
import whisper

# jiwer provides the Word Error Rate and Character Error Rate calculations
from jiwer import wer, cer


# keep the experiment paths relative to this script and project folder
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent.parent

# local LibriSpeech test-clean dataset used for the speech evaluation
LIBRISPEECH_FOLDER = (
    PROJECT_FOLDER
    / "librispeech_data"
    / "LibriSpeech"
    / "test-clean"
)

# validation results contain the speaker IDs that were reserved for final testing
VALIDATION_FOLDER = SCRIPT_FOLDER / "results" / "validation"

# final held-out test outputs are kept separate from validation results
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "test"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


# final Whisper model selected before this held-out test
SELECTED_MODEL = "small.en"

# fixed sampling settings keep repeated runs reproducible
RANDOM_STATE = 42
RECORDINGS_PER_SPEAKER = 12


# clean reference and predicted transcripts in the same way before scoring
def normalise_text(text):
    text = str(text or "").lower()

    # remove punctuation while keeping letters, numbers, apostrophes and spaces
    text = re.sub(r"[^a-z0-9'\s]", " ", text)

    # reduce repeated spaces so formatting differences do not affect the metrics
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# collect LibriSpeech recordings together with their reference transcripts
def collect_librispeech_files():
    if not LIBRISPEECH_FOLDER.exists():
        raise FileNotFoundError(
            f"LibriSpeech folder not found: {LIBRISPEECH_FOLDER}"
        )

    rows = []

    # each LibriSpeech transcript file contains recording IDs followed by text
    for transcript_file in LIBRISPEECH_FOLDER.rglob("*.trans.txt"):
        with open(transcript_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(" ", 1)

                if len(parts) != 2:
                    continue

                recording_id, transcript = parts
                audio_path = transcript_file.parent / f"{recording_id}.flac"

                # skip transcript entries where the matching audio file is missing
                if not audio_path.exists():
                    continue

                # the first part of the LibriSpeech ID identifies the speaker
                speaker_id = recording_id.split("-")[0]

                rows.append({
                    "recording_id": recording_id,
                    "speaker_id": str(speaker_id),
                    "audio_path": str(audio_path),
                    "reference_transcript": transcript,
                    "normalised_reference": normalise_text(transcript),
                })

    if not rows:
        raise FileNotFoundError(
            "No LibriSpeech recordings were found."
        )

    return pd.DataFrame(rows)


# load the speaker IDs that were kept out of the earlier model-selection stage
def load_reserved_speakers():
    path = VALIDATION_FOLDER / "reserved_test_speakers.csv"

    if not path.exists():
        raise FileNotFoundError(
            "reserved_test_speakers.csv was not found. "
            "Run the validation benchmark first."
        )

    # read speaker IDs as strings so formatting is kept consistent
    df = pd.read_csv(
        path,
        dtype=str,
    )

    column = "Reserved Test Speaker ID"

    if column not in df.columns:
        raise ValueError(
            f"Expected column '{column}' was not found."
        )

    return (
        df[column]
        .dropna()
        .astype(str)
        .tolist()
    )


# create one fixed held-out sample from the reserved speakers
def create_test_sample(dataset_df, test_speakers):
    # remove any recordings belonging to speakers outside the reserved test list
    test_df = dataset_df[
        dataset_df["speaker_id"].isin(test_speakers)
    ].copy()

    samples = []

    for speaker in test_speakers:
        speaker_df = test_df[
            test_df["speaker_id"] == speaker
        ]

        # use up to 12 recordings without exceeding what the speaker has available
        sample_size = min(
            RECORDINGS_PER_SPEAKER,
            len(speaker_df),
        )

        # the fixed random state makes the chosen recordings reproducible
        speaker_sample = speaker_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

        samples.append(speaker_sample)

    if not samples:
        raise ValueError(
            "No recordings matched the reserved test speakers."
        )

    # sort after sampling so the saved test file is easy to inspect
    return (
        pd.concat(samples)
        .sort_values(["speaker_id", "recording_id"])
        .reset_index(drop=True)
    )


# transcribe every held-out recording and calculate per-file errors
def evaluate_model(model, sample_df, device):
    references = []
    predictions = []
    rows = []
    failed_files = 0

    total_start = time.perf_counter()

    for index, row in sample_df.iterrows():
        start = time.perf_counter()
        error_message = ""

        try:
            # use deterministic English transcription settings for the benchmark
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
            # failed files stay in the results instead of being silently removed
            predicted_text = ""
            error_message = str(error)
            failed_files += 1

        elapsed = time.perf_counter() - start

        # both transcripts are normalised before comparing them
        reference = row["normalised_reference"]
        prediction = normalise_text(predicted_text)

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

        # keep one detailed row per recording for later checking
        rows.append({
            "Recording ID": row["recording_id"],
            "Speaker ID": row["speaker_id"],
            "Reference Transcript": row["reference_transcript"],
            "Predicted Transcript": predicted_text.strip(),
            "WER": round(file_wer, 4),
            "CER": round(file_cer, 4),
            "Transcription Time (s)": round(elapsed, 4),
            "Processing Error": error_message,
        })

        # print occasional progress during the longer transcription run
        if (
            (index + 1) % 10 == 0
            or index + 1 == len(sample_df)
        ):
            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(sample_df)}"
            )

    total_time = time.perf_counter() - total_start

    # overall WER and CER are calculated over all transcripts together
    overall_wer = wer(
        " ".join(references),
        " ".join(predictions),
    )

    overall_cer = cer(
        " ".join(references),
        " ".join(predictions),
    )

    return {
        "wer": overall_wer,
        "cer": overall_cer,
        "average_time": total_time / len(sample_df),
        "total_time": total_time,
        "failed_files": failed_files,
        "rows": rows,
    }


# save the headline Whisper results as CSV and a report-friendly table image
def save_summary(result, sample_df, speakers):
    summary = pd.DataFrame([{
        "Model": SELECTED_MODEL,
        "Test Speakers": len(speakers),
        "Test Recordings": len(sample_df),
        "WER": round(result["wer"], 4),
        "CER": round(result["cer"], 4),
        "Average Transcription Time (s)": round(
            result["average_time"],
            4,
        ),
        "Total Transcription Time (s)": round(
            result["total_time"],
            2,
        ),
        "Failed Files": result["failed_files"],
    }])

    # save exact headline results as CSV
    summary.to_csv(
        RESULTS_FOLDER / "selected_whisper_model_results.csv",
        index=False,
    )

    # keep only the main fields needed for the visual table
    table_df = summary[[
        "Model",
        "Test Recordings",
        "WER",
        "CER",
        "Average Transcription Time (s)",
        "Failed Files",
    ]].copy()

    table_df.columns = [
        "Model",
        "Recordings",
        "WER",
        "CER",
        "Avg Time (s)",
        "Failed",
    ]

    fig, ax = plt.subplots(figsize=(9, 2.5))

    # hide normal chart axes because this figure is a table only
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
        "Selected Whisper Model - Test Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_whisper_scores_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return summary


# create a simple visual comparison of the two transcription error rates
def save_error_graph(result):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(
        ["WER", "CER"],
        [result["wer"], result["cer"]],
    )

    ax.set_title(
        "Selected Whisper Model - Test Error Rates"
    )
    ax.set_ylabel("Error Rate")

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_whisper_wer_cer.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# show how WER varies across the individual held-out recordings
def save_wer_distribution(predictions_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        predictions_df["WER"],
        bins=15,
    )
    ax.set_title(
        "Selected Whisper Model - "
        "Per-Recording WER Distribution"
    )

    ax.set_xlabel("WER")
    ax.set_ylabel("Number of recordings")

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_whisper_wer_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# summarise transcription performance separately for each reserved speaker
def save_speaker_results(predictions_df):
    # averaging by speaker helps show whether performance is consistent
    speaker_results = (
        predictions_df
        .groupby("Speaker ID")
        .agg(
            Recordings=("Recording ID", "count"),
            Mean_WER=("WER", "mean"),
            Mean_CER=("CER", "mean"),
            Mean_Time=(
                "Transcription Time (s)",
                "mean",
            ),
        )
        .reset_index()
    )

    # rename the output columns to be easier to read in the CSV
    speaker_results.columns = [
        "Speaker ID",
        "Recordings",
        "Mean WER",
        "Mean CER",
        "Mean Time (s)",
    ]

    speaker_results.to_csv(
        RESULTS_FOLDER / "selected_whisper_speaker_results.csv",
        index=False,
    )

    # WER by speaker gives a quick view of differences across speakers
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        speaker_results["Speaker ID"],
        speaker_results["Mean WER"],
    )

    ax.set_title(
        "Selected Whisper Model - WER by Speaker"
    )

    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Mean WER")

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "selected_whisper_wer_by_speaker.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the complete held-out Whisper test
def main():
    print("=" * 60)
    print("SELECTED WHISPER MODEL - TEST")
    print("=" * 60)

    # collect all available LibriSpeech files first
    dataset_df = collect_librispeech_files()

    # load only the speakers that were kept out of model selection
    test_speakers = load_reserved_speakers()

    # create a fixed sample from those reserved speakers
    sample_df = create_test_sample(
        dataset_df,
        test_speakers,
    )

    print(
        f"Reserved speakers: "
        f"{len(test_speakers)}"
    )

    print(
        f"Test recordings: "
        f"{len(sample_df)}"
    )

    print(
        "Speakers: "
        + ", ".join(test_speakers)
    )

    # save the exact held-out sample used for this final test
    sample_df.to_csv(
        RESULTS_FOLDER / "librispeech_test_sample.csv",
        index=False,
    )

    # use CUDA when available, otherwise run Whisper on CPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Loading Whisper {SELECTED_MODEL}...")

    # keep model-loading time separate from transcription time
    load_start = time.perf_counter()

    model = whisper.load_model(
        SELECTED_MODEL,
        device=device,
    )

    load_time = time.perf_counter() - load_start

    print(
        f"Model loaded in "
        f"{load_time:.2f}s"
    )

    # run transcription and calculate the final held-out metrics
    result = evaluate_model(
        model,
        sample_df,
        device,
    )

    predictions_df = pd.DataFrame(
        result["rows"]
    )

    # save every recording result for traceability
    predictions_df.to_csv(
        RESULTS_FOLDER / "selected_whisper_predictions.csv",
        index=False,
    )

    # recordings with any word-level error are saved separately
    errors_df = predictions_df[
        predictions_df["WER"] > 0
    ]

    errors_df.to_csv(
        RESULTS_FOLDER / "selected_whisper_errors.csv",
        index=False,
    )

    # create the headline summary and supporting graphs
    summary = save_summary(
        result,
        sample_df,
        test_speakers,
    )

    save_error_graph(result)
    save_wer_distribution(predictions_df)
    save_speaker_results(predictions_df)

    print("\n" + "=" * 60)
    print("FINAL WHISPER TEST RESULTS")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the test only when this file is executed directly
if __name__ == "__main__":
    main()