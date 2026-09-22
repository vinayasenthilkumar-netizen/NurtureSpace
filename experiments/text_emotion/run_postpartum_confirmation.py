# gc and torch cleanup help free memory between model runs
import gc
import os
import time

# plotting and table libraries for the evaluation outputs
import matplotlib.pyplot as plt
import pandas as pd
import torch

# standard classification metrics used for the confirmation comparison
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Hugging Face pipeline loads each pretrained text-emotion model
from transformers import pipeline

# shared mapping converts raw model labels into the application's six categories
from model_label_mapping import (
    APPLICATION_LABELS,
    rank_application_predictions,
)


# keep all confirmation outputs in a separate results folder
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(
    SCRIPT_FOLDER,
    "results",
    "postpartum_confirmation",
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)


# only the two strongest models from the earlier comparison are tested here
MODELS = {
    "DistilBERT bhadresh":
        "bhadresh-savani/distilbert-base-uncased-emotion",

    "DistilBERT GoEmotions joeddav":
        "joeddav/distilbert-base-uncased-go-emotions-student",
}


# 60 new postpartum-style examples are used so this is separate
# from the first smaller model-comparison scenario set
SCENARIOS = [

    # Positive
    ("I felt refreshed after getting enough time to eat and rest.", "Positive", "Clear"),
    ("I was proud of how I managed the baby's routine today.", "Positive", "Clear"),
    ("I felt cheerful after taking a short walk outside.", "Positive", "Clear"),
    ("The morning went smoothly and I felt hopeful about the rest of the day.", "Positive", "Clear"),
    ("I felt more capable today than I did yesterday.", "Positive", "Clear"),
    ("I was exhausted, but I still felt pleased with how feeding went.", "Positive", "Mixed"),
    ("The night was difficult, but I felt good about getting through it.", "Positive", "Mixed"),
    ("There were stressful moments, although I still felt optimistic.", "Positive", "Mixed"),
    ("Nothing special happened, but overall I felt a little lighter.", "Positive", "Ambiguous"),
    ("I cannot really explain it, but today felt encouraging.", "Positive", "Ambiguous"),

    # Warmth
    ("My partner took time to listen and I felt cared for.", "Warmth", "Clear"),
    ("Holding my baby quietly made me feel very close to them.", "Warmth", "Clear"),
    ("My mother helped with dinner and I felt supported.", "Warmth", "Clear"),
    ("Talking with a friend made me feel understood.", "Warmth", "Clear"),
    ("I felt comforted when my sister checked on me.", "Warmth", "Clear"),
    ("I was having a hard day, but my partner stayed beside me.", "Warmth", "Mixed"),
    ("I felt tired, although spending time with my baby felt comforting.", "Warmth", "Mixed"),
    ("I was stressed, but having family around made me feel less alone.", "Warmth", "Mixed"),
    ("We did not talk much, but having someone nearby felt reassuring.", "Warmth", "Ambiguous"),
    ("I cannot describe it clearly, but I felt connected during feeding.", "Warmth", "Ambiguous"),

    # Low Mood
    ("I felt sad and tearful for much of today.", "Low Mood", "Clear"),
    ("I did not enjoy anything today and felt very low.", "Low Mood", "Clear"),
    ("I felt discouraged and withdrawn from everyone around me.", "Low Mood", "Clear"),
    ("The whole day felt heavy and I struggled to feel interested in anything.", "Low Mood", "Clear"),
    ("I felt empty even though people were around me.", "Low Mood", "Clear"),
    ("My family helped today, but I still felt down.", "Low Mood", "Mixed"),
    ("The baby slept well, although I still felt unusually sad.", "Low Mood", "Mixed"),
    ("Some parts of the day were fine, but I kept feeling low.", "Low Mood", "Mixed"),
    ("I cannot put my finger on it, but everything felt a bit heavy.", "Low Mood", "Ambiguous"),
    ("Nothing was particularly wrong, yet I did not feel like myself today.", "Low Mood", "Ambiguous"),

    # Worry
    ("I kept worrying about whether the baby was getting enough milk.", "Worry", "Clear"),
    ("I feel anxious about managing everything tomorrow.", "Worry", "Clear"),
    ("I was nervous about the baby's appointment.", "Worry", "Clear"),
    ("My mind kept thinking about things that could go wrong.", "Worry", "Clear"),
    ("I feel uneasy about returning to work soon.", "Worry", "Clear"),
    ("I had some rest, but I still worried about the baby's feeding.", "Worry", "Mixed"),
    ("My partner reassured me, although I remained anxious.", "Worry", "Mixed"),
    ("Today went well, but I am nervous about tonight.", "Worry", "Mixed"),
    ("I am not sure why, but I keep feeling uneasy about tomorrow.", "Worry", "Ambiguous"),
    ("Nothing specific is wrong, but I cannot stop thinking about what might happen.", "Worry", "Ambiguous"),

    # Frustration
    ("I felt irritated because I could not get anything done.", "Frustration", "Clear"),
    ("I became angry when the same problem happened again.", "Frustration", "Clear"),
    ("The constant interruptions frustrated me all afternoon.", "Frustration", "Clear"),
    ("I felt annoyed that I could not get even a few minutes alone.", "Frustration", "Clear"),
    ("I was impatient when everything seemed to take longer than expected.", "Frustration", "Clear"),
    ("I love spending time with my baby, but today I felt very frustrated.", "Frustration", "Mixed"),
    ("I was grateful for the help, although I still felt irritated.", "Frustration", "Mixed"),
    ("Things eventually worked out, but I was angry about how difficult they became.", "Frustration", "Mixed"),
    ("I am not exactly angry, but everything seemed to bother me today.", "Frustration", "Ambiguous"),
    ("I cannot explain why, but small things kept getting on my nerves.", "Frustration", "Ambiguous"),

    # Neutral
    ("Today was fairly ordinary and I do not feel strongly about it.", "Neutral", "Clear"),
    ("I feel neither particularly good nor particularly bad today.", "Neutral", "Clear"),
    ("The day has been normal so far.", "Neutral", "Clear"),
    ("I noticed how I felt today, but there was no strong emotion.", "Neutral", "Clear"),
    ("Things happened as usual and I feel quite neutral about them.", "Neutral", "Clear"),
    ("There were good and difficult moments, so I do not have one clear feeling.", "Neutral", "Mixed"),
    ("I felt different things at different times and none really stood out.", "Neutral", "Mixed"),
    ("Part of the day was pleasant and part was stressful, so it feels mixed.", "Neutral", "Mixed"),
    ("I am not really sure what emotion describes today.", "Neutral", "Ambiguous"),
    ("Something felt different today, but I cannot tell what I was feeling.", "Neutral", "Ambiguous"),
]


# create shorter file-safe names for model-specific outputs
def safe_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


# run one model across all 60 confirmation scenarios
def evaluate_model(classifier):
    y_true = []
    y_pred = []
    rows = []
    top3_matches = 0

    start = time.perf_counter()

    for number, (text, expected, scenario_type) in enumerate(
        SCENARIOS,
        start=1,
    ):
        # return all model scores so they can be mapped and ranked
        output = classifier(
            text,
            truncation=True,
        )

        # map model-specific labels into the six application categories
        ranked = rank_application_predictions(output)

        # first mapped label is used for Top-1 evaluation
        predicted = ranked[0][0]

        # also keep the three highest-ranked categories
        top3 = [
            label
            for label, _ in ranked[:3]
        ]

        top3_match = expected in top3

        if top3_match:
            top3_matches += 1

        y_true.append(expected)
        y_pred.append(predicted)

        # save each scenario result so mistakes can be inspected later
        rows.append({
            "Scenario": number,
            "Type": scenario_type,
            "Text": text,
            "Expected": expected,
            "Predicted": predicted,
            "Correct": predicted == expected,
            "Top 3": " | ".join(top3),
            "Expected in Top 3": top3_match,
            "Top 1 Score": round(ranked[0][1], 4),
        })

    total_time = time.perf_counter() - start

    # return both overall metrics and detailed scenario results
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "top3": (
            top3_matches
            / len(SCENARIOS)
        ),

        # Macro F1 gives each application category equal importance
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=APPLICATION_LABELS,
            average="macro",
            zero_division=0,
        ),

        # Weighted F1 also considers category frequency
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


# save detailed predictions, per-class results and confusion matrix for one model
def save_model_results(model_name, result):
    name = safe_filename(model_name)

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

    # output_dict makes the per-class metrics easier to graph later
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=APPLICATION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"classification_report_{name}.csv",
        )
    )

    # confusion matrix shows which application categories are being mixed up
    matrix = confusion_matrix(
        result["y_true"],
        result["y_pred"],
        labels=APPLICATION_LABELS,
    )

    # save exact matrix values as CSV
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

    # create a visual version for the report
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
        f"Postpartum Confirmation - {model_name}"
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

    # report dataframe is returned so per-category F1 can be compared later
    return report_df


# compare the main performance measures for the two models
def save_metric_graph(df):
    x = list(range(len(df)))
    width = 0.2

    fig, ax = plt.subplots(
        figsize=(9, 5)
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
        "Postpartum Confirmation Performance"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Model"]
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "confirmation_metric_comparison.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare the average processing speed of the two models
def save_time_graph(df):
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        df["Model"],
        df["Average Inference Time (s)"],
    )

    ax.set_title(
        "Postpartum Confirmation Inference Time"
    )

    ax.set_ylabel(
        "Seconds per scenario"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "confirmation_inference_time.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare how both models handle clear, mixed and ambiguous examples
def save_type_graph(predictions_df):
    # mean of the boolean Correct column gives accuracy for each group
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
        figsize=(9, 5),
    )

    ax.set_title(
        "Accuracy by Scenario Type"
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(
        axis="x",
        rotation=0,
    )

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "confirmation_scenario_type_accuracy.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# compare F1 separately for each of the six application categories
def save_class_f1_graph(reports):
    rows = []

    for model_name, report_df in reports.items():
        for label in APPLICATION_LABELS:
            rows.append({
                "Model": model_name,
                "Category": label,
                "F1": report_df.loc[
                    label,
                    "f1-score",
                ],
            })

    class_df = pd.DataFrame(rows)

    # pivot gives one model column beside the other for each category
    pivot = class_df.pivot(
        index="Category",
        columns="Model",
        values="F1",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "F1 Score by Postpartum Category"
    )
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "confirmation_per_class_f1.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# create a compact table image for the main confirmation results
def save_results_table(df):
    table_df = df[[
        "Model",
        "Top 1 Accuracy",
        "Top 3 Coverage",
        "Macro F1",
        "Weighted F1",
    ]]

    fig, ax = plt.subplots(
        figsize=(9, 2.5)
    )

    # normal graph axes are not needed for a table
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
        "Postpartum Confirmation Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            RESULTS_FOLDER,
            "confirmation_scores_table.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the full two-model confirmation experiment
def main():
    # save the scenario set separately so the exact test data is documented
    scenario_df = pd.DataFrame([
        {
            "Scenario": i,
            "Text": text,
            "Expected": expected,
            "Type": scenario_type,
        }
        for i, (text, expected, scenario_type)
        in enumerate(SCENARIOS, start=1)
    ])

    scenario_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "confirmation_scenarios.csv",
        ),
        index=False,
    )

    comparison_rows = []
    all_predictions = []
    reports = {}

    # both models are tested using exactly the same 60 scenarios
    for model_name, model_id in MODELS.items():

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)

        classifier = None

        try:
            # CPU is used so both models run under the same setup
            classifier = pipeline(
                "text-classification",
                model=model_id,
                top_k=None,
                device=-1,
            )

            result = evaluate_model(
                classifier
            )

            # keep each model's per-class report for the F1 comparison graph
            reports[model_name] = save_model_results(
                model_name,
                result,
            )

            # add the main metrics to the final comparison table
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
            })

            # combine prediction rows from both models for scenario-type analysis
            for row in result["rows"]:
                row["Model"] = model_name
                all_predictions.append(row)

            print(
                f"Accuracy: {result['accuracy']:.4f} | "
                f"Macro F1: {result['macro_f1']:.4f} | "
                f"Top-3: {result['top3']:.4f}"
            )

        finally:
            # release the current model before loading the next one
            if classifier is not None:
                del classifier

            gc.collect()

            # clear GPU cache as well if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # rank mainly by Macro F1, then Top-1 Accuracy
    comparison_df = comparison_df.sort_values(
        by=[
            "Macro F1",
            "Top 1 Accuracy",
        ],
        ascending=False,
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    # save the headline confirmation comparison
    comparison_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_confirmation_results.csv",
        ),
        index=False,
    )

    # save every scenario prediction from both models
    predictions_df.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "postpartum_confirmation_predictions.csv",
        ),
        index=False,
    )

    # generate the figures used to compare the two finalists
    save_metric_graph(comparison_df)
    save_time_graph(comparison_df)
    save_type_graph(predictions_df)
    save_class_f1_graph(reports)
    save_results_table(comparison_df)

    print("\n" + "=" * 60)
    print("POSTPARTUM CONFIRMATION RESULTS")
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