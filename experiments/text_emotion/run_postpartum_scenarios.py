
# gc is used to clear model objects between tests
import gc
import os
import time

# plotting and table libraries for the evaluation outputs
import matplotlib.pyplot as plt
import pandas as pd
import torch

# standard classification metrics used to compare the models
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Hugging Face pipeline is used to load each pretrained classifier
from transformers import pipeline

# shared mapping converts each model's original labels
# into the smaller set of labels used by the application
from model_label_mapping import (
    APPLICATION_LABELS,
    rank_application_predictions,
)


# keep all experiment outputs inside the local results/postpartum folder
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(
    SCRIPT_FOLDER,
    "results",
    "postpartum",
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# five pretrained models are compared using the same scenario set
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


# fictional postpartum-style reflections are grouped across six application labels
# scenarios also include clear, mixed and ambiguous wording
SCENARIOS = [

    # Positive
    {
        "text": "I felt cheerful and hopeful after getting a little rest.",
        "expected": "Positive",
        "type": "Clear",
    },
    {
        "text": "Today went better than I expected and I feel relieved.",
        "expected": "Positive",
        "type": "Clear",
    },
    {
        "text": "I felt proud that I managed the morning routine.",
        "expected": "Positive",
        "type": "Clear",
    },
    {
        "text": "I am tired, but I still feel optimistic about tomorrow.",
        "expected": "Positive",
        "type": "Mixed",
    },
    {
        "text": "I enjoyed a quiet moment and felt genuinely happy.",
        "expected": "Positive",
        "type": "Clear",
    },

    # Warmth
    {
        "text": "My partner listened to me and I felt cared for.",
        "expected": "Warmth",
        "type": "Clear",
    },
    {
        "text": "My sister checked in and I felt less alone.",
        "expected": "Warmth",
        "type": "Clear",
    },
    {
        "text": "I felt close to my baby during feeding today.",
        "expected": "Warmth",
        "type": "Clear",
    },
    {
        "text": "I was stressed, but my family made me feel supported.",
        "expected": "Warmth",
        "type": "Mixed",
    },
    {
        "text": "A friend sat with me and I felt understood.",
        "expected": "Warmth",
        "type": "Clear",
    },

    # Low Mood
    {
        "text": "I felt down and tearful for most of the day.",
        "expected": "Low Mood",
        "type": "Clear",
    },
    {
        "text": "Even with help around me, I still felt very low.",
        "expected": "Low Mood",
        "type": "Mixed",
    },
    {
        "text": "Nothing felt enjoyable today and I felt empty.",
        "expected": "Low Mood",
        "type": "Clear",
    },
    {
        "text": "I felt discouraged when the day became difficult.",
        "expected": "Low Mood",
        "type": "Clear",
    },
    {
        "text": "I have been feeling sad and withdrawn today.",
        "expected": "Low Mood",
        "type": "Clear",
    },

    # Worry
    {
        "text": "I keep worrying that I will not manage tomorrow.",
        "expected": "Worry",
        "type": "Clear",
    },
    {
        "text": "I felt nervous about whether the baby was feeding enough.",
        "expected": "Worry",
        "type": "Clear",
    },
    {
        "text": "I slept well, but I am anxious about returning to work.",
        "expected": "Worry",
        "type": "Mixed",
    },
    {
        "text": "My mind kept going over everything that might go wrong.",
        "expected": "Worry",
        "type": "Clear",
    },
    {
        "text": "I feel uneasy and uncertain about the next few days.",
        "expected": "Worry",
        "type": "Mixed",
    },

    # Frustration
    {
        "text": "I felt frustrated when nothing went as planned.",
        "expected": "Frustration",
        "type": "Clear",
    },
    {
        "text": "The constant interruptions made me irritated.",
        "expected": "Frustration",
        "type": "Clear",
    },
    {
        "text": "I love my baby, but I felt angry with how difficult today was.",
        "expected": "Frustration",
        "type": "Mixed",
    },
    {
        "text": "I became annoyed because I could not get a moment to myself.",
        "expected": "Frustration",
        "type": "Clear",
    },
    {
        "text": "I felt upset and impatient when the same problem happened again.",
        "expected": "Frustration",
        "type": "Clear",
    },

    # Neutral
    {
        "text": "I am not sure how I feel today.",
        "expected": "Neutral",
        "type": "Ambiguous",
    },
    {
        "text": "Today felt different, but I cannot explain why.",
        "expected": "Neutral",
        "type": "Ambiguous",
    },
    {
        "text": "Some moments felt good and others difficult, so my feelings are unclear.",
        "expected": "Neutral",
        "type": "Mixed",
    },
    {
        "text": "I feel a mixture of things and cannot name one emotion.",
        "expected": "Neutral",
        "type": "Ambiguous",
    },
    {
        "text": "I noticed a change in how I felt, but I do not know what it means.",
        "expected": "Neutral",
        "type": "Ambiguous",
    },
]


# convert model names into safe filenames for CSV and image outputs
def safe_filename(name):
    """Create a safe filename."""

    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


# run one classifier across all 30 fictional postpartum scenarios
def evaluate_model(classifier):
    y_true = []
    y_pred = []
    rows = []
    top_three_matches = 0

    # timing starts before the first scenario is processed
    start = time.perf_counter()

    for number, scenario in enumerate(SCENARIOS, start=1):

        # top_k=None returns all model labels and scores
        raw_output = classifier(
            scenario["text"],
            truncation=True,
        )

        # map the model-specific labels into the application's six labels
        ranked = rank_application_predictions(raw_output)

        # first ranked application label is treated as the main prediction
        top_one = ranked[0][0]

        # Top-3 coverage checks whether the expected label appears anywhere
        # among the three highest-ranked mapped predictions
        top_three = [
            label
            for label, _ in ranked[:3]
        ]

        top_three_match = (
            scenario["expected"]
            in top_three
        )

        if top_three_match:
            top_three_matches += 1

        y_true.append(
            scenario["expected"]
        )

        y_pred.append(
            top_one
        )

        # keep detailed scenario-level evidence for later inspection
        rows.append({
            "Scenario": number,
            "Type": scenario["type"],
            "Text": scenario["text"],
            "Expected": scenario["expected"],
            "Predicted": top_one,
            "Correct": top_one == scenario["expected"],
            "Top 3": " | ".join(top_three),
            "Expected in Top 3": top_three_match,
            "Top 1 Score": round(
                ranked[0][1],
                4,
            ),
        })

    total_time = time.perf_counter() - start

    # return both overall metrics and individual scenario predictions
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "top3": (
            top_three_matches
            / len(SCENARIOS)
        ),

        # Macro F1 gives all six application labels equal importance
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=APPLICATION_LABELS,
            average="macro",
            zero_division=0,
        ),

        # Weighted F1 also reflects the number of examples for each label
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=APPLICATION_LABELS,
            average="weighted",
            zero_division=0,
        ),

        "average_time": (
            total_time
            / len(SCENARIOS)
        ),

        "total_time": total_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "rows": rows,
    }


# save detailed predictions, classification report and confusion matrix
# for one candidate model
def save_model_results(model_name, result):
    name = safe_filename(
        model_name
    )

    # one row per postpartum scenario
    pd.DataFrame(
        result["rows"]
    ).to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"predictions_{name}.csv",
        ),
        index=False,
    )

    # save precision, recall and F1 for each application label
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=APPLICATION_LABELS,
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

    # confusion matrix shows which application labels are mixed up
    matrix = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=APPLICATION_LABELS,
    )

    # CSV keeps the exact confusion counts
    pd.DataFrame(
        matrix,
        index=[
            f"True {x}"
            for x in APPLICATION_LABELS
        ],
        columns=[
            f"Pred {x}"
            for x in APPLICATION_LABELS
        ],
    ).to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"confusion_matrix_{name}.csv",
        )
    )

    # image version is useful as report evidence
    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=APPLICATION_LABELS,
    )

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=45,
    )

    ax.set_title(
        f"Postpartum Scenario Confusion Matrix - {model_name}"
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


# compare the four main performance measures across all completed models
def save_metric_graph(df):
    x = list(range(len(df)))
    width = 0.2

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    # each model gets four neighbouring bars
    ax.bar(
        [i - 1.5 * width for i in x],
        df["Top 1 Accuracy"],
        width,
        label="Accuracy",
    )

    ax.bar(
        [i - 0.5 * width for i in x],
        df["Top 3 Coverage"],
        width,
        label="Top-3 Coverage",
    )

    ax.bar(
        [i + 0.5 * width for i in x],
        df["Macro F1"],
        width,
        label="Macro F1",
    )

    ax.bar(
        [i + 1.5 * width for i in x],
        df["Weighted F1"],
        width,
        label="Weighted F1",
    )

    ax.set_title(
        "Postpartum Text Model Performance"
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
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_metric_comparison.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare average processing speed for each completed model
def save_time_graph(df):
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        df["Model"],
        df["Average Inference Time (s)"],
    )

    ax.set_title(
        "Postpartum Text Model Inference Time"
    )

    ax.set_ylabel(
        "Seconds per scenario"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_inference_time.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# check whether the models behave differently on clear, mixed
# and ambiguous reflection wording
def save_type_graph(predictions_df):
    # average the Correct boolean for each model and scenario type
    # this gives the accuracy for that group
    type_results = (
        predictions_df
        .groupby(
            ["Model", "Type"]
        )["Correct"]
        .mean()
        .unstack(fill_value=0)
    )

    ax = type_results.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_title(
        "Accuracy by Postpartum Scenario Type"
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    ax.legend(
        title="Scenario Type"
    )

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_scenario_type_accuracy.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# create a compact image table with the main comparison scores
def save_results_table(df):
    # only the main metrics are included so the table stays readable
    table_df = df[[
        "Model",
        "Top 1 Accuracy",
        "Top 3 Coverage",
        "Macro F1",
        "Weighted F1",
    ]].copy()

    fig, ax = plt.subplots(
        figsize=(11, 3.5)
    )

    # hide normal graph axes because this output is only a table
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    # set a fixed font size for report readability
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title(
        "Postpartum Text Model Comparison",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_model_scores_table.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run all five models on exactly the same postpartum scenario set
def main():
    comparison_rows = []
    all_predictions = []

    for model_name, model_id in MODELS.items():

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        # keep this outside try so it can be removed safely in finally
        classifier = None

        try:
            # all models are tested through the same Hugging Face pipeline
            # CPU is used here for a consistent comparison environment
            classifier = pipeline(
                "text-classification",
                model=model_id,
                top_k=None,
                device=-1,
            )

            # evaluate all 30 scenarios and save model-specific evidence
            result = evaluate_model(
                classifier
            )

            save_model_results(
                model_name,
                result,
            )

            # keep one summary row for the final model comparison
            comparison_rows.append({
                "Model": model_name,
                "Model ID": model_id,

                "Top 1 Accuracy": round(
                    result["accuracy"],
                    4,
                ),

                "Top 3 Coverage": round(
                    result["top3"],
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

                "Total Time (s)": round(
                    result["total_time"],
                    2,
                ),

                "Status": "Completed",
            })

            # combine all scenario predictions so performance by scenario type
            # can be compared later
            for row in result["rows"]:
                row["Model"] = model_name
                all_predictions.append(row)

            # print the main measures after each model finishes
            print(
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Top-3: {result['top3']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f}"
            )

        except Exception as error:
            # failed models stay in the results instead of being silently removed
            print(
                f"Model failed: {error}"
            )

            comparison_rows.append({
                "Model": model_name,
                "Model ID": model_id,
                "Top 1 Accuracy": None,
                "Top 3 Coverage": None,
                "Macro F1": None,
                "Weighted F1": None,
                "Average Inference Time (s)": None,
                "Total Time (s)": None,
                "Status": f"Failed: {error}",
            })

        finally:
            # remove the previous classifier before loading the next model
            if classifier is not None:
                del classifier

            gc.collect()

            # clear GPU cache as well if CUDA happens to be available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # order completed models mainly by Macro F1, then Top-1 Accuracy
    comparison_df = comparison_df.sort_values(
        by=[
            "Macro F1",
            "Top 1 Accuracy",
        ],
        ascending=False,
        na_position="last",
    )

    # one dataframe contains every model/scenario prediction
    predictions_df = pd.DataFrame(
        all_predictions
    )

    # save the overall model comparison
    comparison_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_model_comparison.csv",
        ),
        index=False,
    )

    # detailed file makes individual scenario behaviour easier to inspect
    predictions_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_all_predictions.csv",
        ),
        index=False,
    )

    # graphs should only include models that completed successfully
    completed_df = comparison_df[
        comparison_df["Status"] == "Completed"
    ].copy()

    if not completed_df.empty:
        save_metric_graph(
            completed_df
        )
        save_time_graph(
            completed_df
        )
        save_results_table(
            completed_df
        )

    # scenario-type comparison needs at least one successful prediction set
    if not predictions_df.empty:
        save_type_graph(
            predictions_df
        )

    print("\n" + "=" * 60)
    print("POSTPARTUM MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print("\nResults saved in:")
    print(RESULTS_FOLDER)


# run the experiment only when this file is executed directly
if __name__ == "__main__":
    main()