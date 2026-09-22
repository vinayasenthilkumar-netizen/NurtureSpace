
# os is used for keeping result paths relative to this experiment file
import os
import time

# plotting and table libraries used for evaluation outputs
import matplotlib.pyplot as plt
import pandas as pd

# Hugging Face dataset loader provides the DAIR-AI emotion test split
from datasets import load_dataset

# standard classification metrics used for the final evaluation
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# transformer pipeline runs the selected pretrained text-emotion model
from transformers import pipeline

# shared mapping helper converts GoEmotions outputs into benchmark labels
from model_label_mapping import (
    BENCHMARK_LABELS,
    get_top_benchmark_prediction,
)


# final text model chosen before running this test
SELECTED_MODEL = (
    "joeddav/"
    "distilbert-base-uncased-go-emotions-student"
)

# shorter name used in result tables and figures
MODEL_NAME = "DistilBERT GoEmotions joeddav"


# keep all test outputs inside this experiment folder
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(SCRIPT_FOLDER, "results", "test")

# create the output folder if it does not already exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# DAIR-AI stores emotion classes as integer labels
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# other is kept in the confusion matrix because some model outputs
# may not map directly into the benchmark emotion classes
REPORT_LABELS = BENCHMARK_LABELS + ["other"]


# load the held-out DAIR-AI test split used for final evaluation
def load_test_data():
    print("Loading DAIR-AI test set...")

    # use the official test split rather than taking another sample
    dataset = load_dataset("dair-ai/emotion")

    # only text and numeric label are required for this experiment
    test_df = dataset["test"].to_pandas()[["text", "label"]].copy()

    # convert the dataset's integer label into a readable emotion name
    test_df["true_label"] = test_df["label"].map(LABEL_MAP)

    print(f"Test examples: {len(test_df)}")

    return test_df


# run the selected classifier over every example in the test set
def evaluate_model(classifier, test_df):
    y_true = []
    y_pred = []
    predictions = []
    errors = []

    # timing starts before the first prediction so total evaluation time is recorded
    start = time.perf_counter()

    for index, row in test_df.iterrows():

        # top_k=None returns the scores needed by the shared benchmark mapping
        raw_output = classifier(
            row["text"],
            truncation=True,
        )

        # map the model's GoEmotions-style output into the benchmark classes
        prediction = get_top_benchmark_prediction(
            raw_output
        )

        true_label = row["true_label"]
        predicted_label = prediction["label"]

        y_true.append(true_label)
        y_pred.append(predicted_label)

        # keep each prediction so individual successes and mistakes can be checked
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

        # save incorrect examples separately for later error analysis
        if predicted_label != true_label:
            errors.append(result.copy())

        # print progress every 200 examples
        if (index + 1) % 200 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(test_df)}"
            )

    total_time = time.perf_counter() - start

    # overall proportion of correct test predictions
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # macro F1 gives each benchmark class equal importance
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=BENCHMARK_LABELS,
        average="macro",
        zero_division=0,
    )

    # weighted F1 also considers how common each class is in the test set
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=BENCHMARK_LABELS,
        average="weighted",
        zero_division=0,
    )

    # return both headline metrics and detailed predictions
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "average_time": total_time / len(test_df),
        "total_time": total_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "predictions": predictions,
        "errors": errors,
    }


# save the confusion matrix in both CSV and image form
def save_confusion_matrix(result):
    # include other so unsupported mapped predictions remain visible
    matrix = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=REPORT_LABELS,
    )

    # dataframe version keeps the exact confusion counts
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"True {x}" for x in REPORT_LABELS],
        columns=[f"Pred {x}" for x in REPORT_LABELS],
    )

    matrix_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_confusion_matrix.csv",
        )
    )

    # create a report-friendly visual version
    fig, ax = plt.subplots(figsize=(8, 7))

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
        "Selected Text Model - Test Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# calculate and save precision, recall and F1 for each benchmark class
def save_classification_report(result):
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=BENCHMARK_LABELS,
        output_dict=True,
        zero_division=0,
    )

    # transpose makes each emotion class appear as one row
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_classification_report.csv",
        )
    )

    # only the three main class metrics are needed for the graph
    class_df = report_df.loc[
        BENCHMARK_LABELS,
        ["precision", "recall", "f1-score"],
    ]

    ax = class_df.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "Selected Text Model - Per-Class Performance"
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Emotion")
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis="x",
        rotation=25,
    )
    ax.legend(
        ["Precision", "Recall", "F1"]
    )

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_class_metrics.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return report_df


# save the headline final-test results as CSV and a report table image
def save_summary(result, test_size):
    summary_df = pd.DataFrame([{
        "Model": MODEL_NAME,
        "Model ID": SELECTED_MODEL,
        "Test Examples": test_size,
        "Accuracy": round(result["accuracy"], 4),
        "Macro F1": round(result["macro_f1"], 4),
        "Weighted F1": round(result["weighted_f1"], 4),
        "Average Inference Time (s)": round(
            result["average_time"],
            4,
        ),
        "Total Evaluation Time (s)": round(
            result["total_time"],
            2,
        ),
        "Errors": len(result["errors"]),
    }])

    # CSV keeps the exact final test numbers
    summary_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_results.csv",
        ),
        index=False,
    )

    # use a smaller set of metrics in the image so the table stays readable
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

    fig, ax = plt.subplots(figsize=(11, 2.5))

    # normal chart axes are not needed for a table figure
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    # keep the text size fixed so matplotlib does not shrink it too much
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title(
        "Selected Text Model - Test Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_scores_table.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return summary_df


# run the complete final text-model evaluation
def main():
    # load the full DAIR-AI test split first
    test_df = load_test_data()

    # CPU is used here so the experiment has a fixed deployment setting
    classifier = pipeline(
        "text-classification",
        model=SELECTED_MODEL,
        top_k=None,
        device=-1,
    )

    print(f"\nTesting: {MODEL_NAME}")
    print(f"Model: {SELECTED_MODEL}")

    # run inference and calculate the main evaluation metrics
    result = evaluate_model(
        classifier,
        test_df,
    )

    # save every prediction for traceability
    pd.DataFrame(
        result["predictions"]
    ).to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_predictions.csv",
        ),
        index=False,
    )

    # save only incorrect predictions separately for error analysis
    pd.DataFrame(
        result["errors"]
    ).to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "selected_text_model_errors.csv",
        ),
        index=False,
    )

    # generate the main evaluation evidence
    save_confusion_matrix(result)
    save_classification_report(result)

    summary_df = save_summary(
        result,
        len(test_df),
    )

    # print headline results at the end of the run
    print("\n" + "=" * 60)
    print("SELECTED TEXT MODEL - TEST RESULTS")
    print("=" * 60)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the evaluation only when this script is executed directly
if __name__ == "__main__":
    main()