
# gc and torch cleanup help release model memory between comparison runs
import gc
import os
import time

# plotting and dataframe libraries used for evaluation outputs
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Hugging Face is used for the DAIR-AI dataset and pretrained classifiers
from datasets import load_dataset
from transformers import pipeline

# standard classification metrics used to compare the candidate models
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# shared mapping converts different model outputs into the common benchmark labels
from model_label_mapping import (
    BENCHMARK_LABELS,
    get_top_benchmark_prediction,
)


# use a fixed validation sample so every model sees exactly the same examples
SAMPLE_SIZE = 200
RANDOM_STATE = 42


# keep generated validation evidence inside this experiment folder
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(
    SCRIPT_FOLDER,
    "results",
    "validation",
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# five pretrained emotion models are compared before selecting the final model
MODELS = {
    "DistilRoBERTa j-hartmann":
        "j-hartmann/emotion-english-distilroberta-base",

    "DistilBERT bhadresh":
        "bhadresh-savani/distilbert-base-uncased-emotion",

    "RoBERTa GoEmotions SamLowe":
        "SamLowe/roberta-base-go_emotions",

    "Twitter RoBERTa CardiffNLP":
        "cardiffnlp/twitter-roberta-base-emotion-multilabel-latest",

    "DistilBERT GoEmotions joeddav":
        "joeddav/distilbert-base-uncased-go-emotions-student",
}


# DAIR-AI stores the six emotion classes as integer values
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# other is included in the confusion matrix for predictions
# that do not map directly to one of the six benchmark classes
REPORT_LABELS = BENCHMARK_LABELS + ["other"]


# create a simple safe filename from the display name of each model
def safe_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


# load one reproducible sample from the official validation split
def load_validation_sample():
    print("Loading DAIR-AI validation set...")

    # validation is used for model selection, not the final test split
    dataset = load_dataset("dair-ai/emotion")

    validation_df = (
        dataset["validation"]
        .to_pandas()[["text", "label"]]
        .copy()
    )

    # convert the numeric dataset labels into readable emotion names
    validation_df["true_label"] = validation_df["label"].map(
        LABEL_MAP
    )

    # avoid requesting more rows than are actually available
    sample_size = min(
        SAMPLE_SIZE,
        len(validation_df),
    )

    # fixed random state means every candidate model gets the same sample
    sample = validation_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(
        f"Using {len(sample)} validation examples."
    )

    return sample


# run one candidate model over the fixed validation sample
def evaluate_model(classifier, data):
    y_true = []
    y_pred = []
    predictions = []
    errors = []

    start = time.perf_counter()

    for index, row in data.iterrows():

        # return the model scores needed by the shared benchmark mapping
        raw_output = classifier(
            row["text"],
            truncation=True,
        )

        # reduce model-specific output to one common benchmark prediction
        prediction = get_top_benchmark_prediction(
            raw_output
        )

        true_label = row["true_label"]
        predicted_label = prediction["label"]

        y_true.append(true_label)
        y_pred.append(predicted_label)

        # keep the full row so individual predictions can be inspected later
        result = {
            "Row": index + 1,
            "Text": row["text"],
            "True Label": true_label,
            "Predicted Label": predicted_label,
            "Score": round(
                prediction["score"],
                4,
            ),
            "Correct": predicted_label == true_label,
        }

        predictions.append(result)

        # incorrect cases are also stored separately for error analysis
        if predicted_label != true_label:
            errors.append(
                result.copy()
            )

    total_time = time.perf_counter() - start

    # return both headline metrics and the detailed prediction evidence
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        # Macro F1 gives each benchmark emotion equal importance
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=BENCHMARK_LABELS,
            average="macro",
            zero_division=0,
        ),

        # Weighted F1 also considers the number of examples in each class
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=BENCHMARK_LABELS,
            average="weighted",
            zero_division=0,
        ),

        "average_time": (
            total_time
            / len(data)
        ),

        "total_time": total_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "predictions": predictions,
        "errors": errors,
    }


# save the detailed evidence produced by one candidate model
def save_model_results(model_name, result):
    name = safe_filename(
        model_name
    )

    # per-class precision, recall and F1 make class-specific weaknesses visible
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=BENCHMARK_LABELS,
        zero_division=0,
    )

    with open(
        os.path.join(
            RESULTS_FOLDER,
            f"classification_report_{name}.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    # include other in the confusion matrix so mapped out-of-set outputs remain visible
    matrix = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=REPORT_LABELS,
    )

    # save the exact confusion counts as CSV
    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"True {x}"
            for x in REPORT_LABELS
        ],
        columns=[
            f"Pred {x}"
            for x in REPORT_LABELS
        ],
    )

    matrix_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"confusion_matrix_{name}.csv",
        )
    )

    # also create an image version for easier use in the report
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=REPORT_LABELS,
    )

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=45,
    )

    ax.set_title(
        f"Validation Confusion Matrix - {model_name}"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            f"confusion_matrix_{name}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # save every prediction for traceability
    pd.DataFrame(
        result["predictions"]
    ).to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"predictions_{name}.csv",
        ),
        index=False,
    )

    # save only incorrect examples separately so they are easier to review
    pd.DataFrame(
        result["errors"]
    ).to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"errors_{name}.csv",
        ),
        index=False,
    )

    return report


# compare Accuracy, Macro F1 and Weighted F1 across successful models
def save_metric_graph(comparison_df):

    # failed models are not meaningful on a performance graph
    completed = comparison_df[
        comparison_df["Status"] == "Completed"
    ].copy()

    if completed.empty:
        return

    names = completed["Model"].tolist()
    x = list(range(len(names)))
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    # three neighbouring bars are shown for each model
    ax.bar(
        [i - width for i in x],
        completed["Accuracy"],
        width,
        label="Accuracy",
    )

    ax.bar(
        x,
        completed["Macro F1"],
        width,
        label="Macro F1",
    )

    ax.bar(
        [i + width for i in x],
        completed["Weighted F1"],
        width,
        label="Weighted F1",
    )

    ax.set_title(
        "Text Emotion Model Performance - Validation Set"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)

    ax.set_xticklabels(
        names,
        rotation=25,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "validation_model_metric_comparison.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare the average processing time of each successful model
def save_time_graph(comparison_df):
    completed = comparison_df[
        comparison_df["Status"] == "Completed"
    ].copy()

    if completed.empty:
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        completed["Model"],
        completed["Average Inference Time (s)"],
    )

    ax.set_title(
        "Text Model Inference Time - Validation Set"
    )

    ax.set_ylabel(
        "Seconds per example"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "validation_model_inference_time.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# create a compact image table of the main validation results
def save_results_table(comparison_df):
    # only the most useful comparison fields are shown in the image
    table_df = comparison_df[[
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Average Inference Time (s)",
    ]].copy()

    # shorter heading keeps the table readable
    table_df.columns = [
        "Model",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Avg Time (s)",
    ]

    fig, ax = plt.subplots(
        figsize=(12, 3.5)
    )

    # hide normal plotting axes because this figure contains a table only
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
        "Text Emotion Model Validation Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "validation_model_scores_table.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run all five candidate models on the same validation sample
def main():
    # draw the fixed validation sample once before loading any model
    sample = load_validation_sample()

    # save the exact sample so the comparison can be reproduced later
    sample.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "validation_sample.csv",
        ),
        index=False,
    )

    comparison_rows = []

    for model_name, model_id in MODELS.items():

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        # defined before try so cleanup can safely check it in finally
        classifier = None

        try:
            # CPU is used so all five models are compared under the same setup
            classifier = pipeline(
                "text-classification",
                model=model_id,
                top_k=None,
                device=-1,
            )

            # every model is tested on the same 200 validation examples
            result = evaluate_model(
                classifier,
                sample,
            )

            report = save_model_results(
                model_name,
                result,
            )

            # keep one summary row for the final comparison
            comparison_rows.append({
                "Model": model_name,
                "Model ID": model_id,

                "Accuracy": round(
                    result["accuracy"],
                    4,
                ),

                "Macro F1": round(
                    result["macro_f1"],
                    4,
                ),

                "Weighted F1": round(
                    result["weighted_f1"],
                    4,
                ),

                "Average Inference Time (s)": round(
                    result["average_time"],
                    4,
                ),

                "Total Evaluation Time (s)": round(
                    result["total_time"],
                    2,
                ),

                "Errors": len(
                    result["errors"]
                ),

                "Status": "Completed",
            })

            # print the class report as well as the two main selection metrics
            print(report)

            print(
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f}"
            )

        except Exception as error:
            # keep failed models in the comparison rather than removing them
            print(
                f"Model failed: {error}"
            )

            comparison_rows.append({
                "Model": model_name,
                "Model ID": model_id,
                "Accuracy": None,
                "Macro F1": None,
                "Weighted F1": None,
                "Average Inference Time (s)": None,
                "Total Evaluation Time (s)": None,
                "Errors": None,
                "Status": f"Failed: {error}",
            })

        finally:
            # delete the loaded model before moving to the next candidate
            if classifier is not None:
                del classifier

            gc.collect()

            # clear GPU memory as well if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # rank mainly by Macro F1, then Accuracy
    # failed models are placed at the bottom
    comparison_df = comparison_df.sort_values(
        by=[
            "Macro F1",
            "Accuracy",
        ],
        ascending=False,
        na_position="last",
    )

    # save the final model-selection comparison
    comparison_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "validation_model_comparison_results.csv",
        ),
        index=False,
    )

    # generate report-friendly comparison figures
    save_metric_graph(
        comparison_df
    )

    save_time_graph(
        comparison_df
    )

    save_results_table(
        comparison_df
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the validation experiment only when this file is executed directly
if __name__ == "__main__":
    main()