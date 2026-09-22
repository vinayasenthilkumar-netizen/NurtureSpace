
# pathlib keeps the output folder relative to this script
from pathlib import Path

# plotting and data libraries used for the analysis outputs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# keep all generated files beside this script inside a results folder
SCRIPT_FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = SCRIPT_FOLDER / "results"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


# compare the chosen 70:30 weighting against two nearby alternatives
# reflection always keeps the larger share in these tests
WEIGHTS = {
    "60:40": (0.60, 0.40),
    "70:30": (0.70, 0.30),
    "80:20": (0.80, 0.20),
}


# thresholds used to turn the internal combined value into a category
STEADY_THRESHOLD = 3.67
STRAIN_THRESHOLD = 2.34


# convert one internal combined value into the user-facing indicator
def overall_checkin_indicator(value):
    if value >= STEADY_THRESHOLD:
        return "Relatively steady"

    if value >= STRAIN_THRESHOLD:
        return "Some strain"

    return "Significant strain"


# calculate the weighted value before it is converted into a category
def internal_combined_value(reflection, questions, weights):
    reflection_weight, question_weight = weights

    return (
        reflection * reflection_weight
        + questions * question_weight
    )


# test every possible Reflection and Question score pairing used in this analysis
def build_results():

    # scores move from 1 to 5 in steps of 0.25
    # this gives a wide range of matching and conflicting score combinations
    values = np.arange(1.0, 5.01, 0.25)
    rows = []

    # test each Reflection score against every Question score
    for reflection in values:
        for questions in values:

            # keep the original scores and their distance apart
            # the difference is useful later when grouping disagreement levels
            row = {
                "Reflection Score": round(reflection, 2),
                "Question Score": round(questions, 2),
                "Score Difference": round(
                    abs(reflection - questions),
                    2,
                ),
            }

            # calculate the internal value and category for every weighting
            for name, weights in WEIGHTS.items():
                value = internal_combined_value(
                    reflection,
                    questions,
                    weights,
                )

                row[f"{name} Internal Combined Value"] = round(
                    value,
                    4,
                )

                row[f"{name} Overall Check-In Indicator"] = (
                    overall_checkin_indicator(value)
                )

            # measure how far 60:40 moves away from the selected 70:30 value
            row["60:40 vs 70:30 Difference"] = round(
                abs(
                    row["60:40 Internal Combined Value"]
                    - row["70:30 Internal Combined Value"]
                ),
                4,
            )

            # do the same comparison for 80:20
            row["80:20 vs 70:30 Difference"] = round(
                abs(
                    row["80:20 Internal Combined Value"]
                    - row["70:30 Internal Combined Value"]
                ),
                4,
            )

            # a weighting change matters more when it changes the final category
            row["60:40 Changed Indicator"] = (
                row["60:40 Overall Check-In Indicator"]
                != row["70:30 Overall Check-In Indicator"]
            )

            row["80:20 Changed Indicator"] = (
                row["80:20 Overall Check-In Indicator"]
                != row["70:30 Overall Check-In Indicator"]
            )

            rows.append(row)

    # dataframe makes the later summaries and exports easier to calculate
    return pd.DataFrame(rows)


# reduce the detailed combinations into one overall row per alternative weighting
def create_summary(results):

    rows = []

    # 70:30 is the reference, so only the two alternatives need summary rows
    for name in ["60:40", "80:20"]:
        difference = results[f"{name} vs 70:30 Difference"]
        changed = results[f"{name} Changed Indicator"]

        rows.append({
            "Weighting": name,
            "Compared With": "70:30",

            # average amount the internal value shifts
            "Mean Absolute Internal Value Difference": round(
                difference.mean(),
                4,
            ),

            # largest shift found across any tested combination
            "Maximum Internal Value Difference": round(
                difference.max(),
                4,
            ),

            # proportion of cases where the user-facing indicator stays the same
            "Indicator Agreement": round(
                1 - changed.mean(),
                4,
            ),

            # raw number of combinations where the category changes
            "Indicator Changes": int(
                changed.sum()
            ),

            # proportion of all combinations where the category changes
            "Indicator Change Rate": round(
                changed.mean(),
                4,
            ),
        })

    return pd.DataFrame(rows)


# check whether weighting matters more when Reflection and Question scores disagree
def disagreement_summary(results):

    # score pairs are grouped by the distance between Reflection and Questions
    groups = [
        (
            "Similar scores",
            results["Score Difference"] < 1,
        ),
        (
            "Moderate disagreement",
            (results["Score Difference"] >= 1)
            & (results["Score Difference"] < 2),
        ),
        (
            "Large disagreement",
            results["Score Difference"] >= 2,
        ),
    ]

    rows = []

    for label, mask in groups:
        subset = results[mask]

        # this shows whether weighting choice becomes more important
        # when the two input scores are further apart
        rows.append({
            "Score Relationship": label,
            "Cases": len(subset),

            "60:40 Indicator Change Rate": round(
                subset["60:40 Changed Indicator"].mean(),
                4,
            ),

            "80:20 Indicator Change Rate": round(
                subset["80:20 Changed Indicator"].mean(),
                4,
            ),

            "60:40 Mean Difference": round(
                subset["60:40 vs 70:30 Difference"].mean(),
                4,
            ),

            "80:20 Mean Difference": round(
                subset["80:20 vs 70:30 Difference"].mean(),
                4,
            ),
        })

    return pd.DataFrame(rows)


# save a small set of easy-to-explain score pairs for the report
def save_example_cases(results):

    # these include cases where one score is much higher than the other
    # as well as cases where the two scores are fairly close
    examples = pd.DataFrame([
        {"Reflection Score": 4.5, "Question Score": 2.0},
        {"Reflection Score": 2.0, "Question Score": 4.5},
        {"Reflection Score": 4.0, "Question Score": 3.5},
        {"Reflection Score": 3.5, "Question Score": 4.0},
        {"Reflection Score": 2.5, "Question Score": 2.0},
        {"Reflection Score": 2.0, "Question Score": 2.5},
        {"Reflection Score": 3.75, "Question Score": 3.0},
        {"Reflection Score": 3.0, "Question Score": 3.75},
    ])

    # pull the full calculated results for only these selected combinations
    selected = results.merge(
        examples,
        on=["Reflection Score", "Question Score"],
        how="inner",
    )

    # keep these examples separate from the much larger detailed results file
    selected.to_csv(
        RESULTS_FOLDER / "weighting_example_cases.csv",
        index=False,
    )


# create the figures used to visually compare the weighting options
def save_graphs(results, summary):

    # use the two alternative weighting names on the x-axis
    labels = summary["Weighting"]


    # graph 1: average numerical movement away from 70:30
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        labels,
        summary["Mean Absolute Internal Value Difference"],
    )

    ax.set_title(
        "Average Internal Value Difference from 70:30"
    )
    ax.set_ylabel(
        "Mean Absolute Difference"
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "weighting_score_difference.png",
        dpi=300,
        bbox_inches="tight",
    )

    # close each figure after saving so repeated runs do not keep it in memory
    plt.close(fig)


    # graph 2: percentage of combinations that change wellbeing category
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        labels,
        summary["Indicator Change Rate"] * 100,
    )

    ax.set_title(
        "Overall Check-In Indicator Changes"
    )
    ax.set_ylabel(
        "Cases Changing Indicator (%)"
    )

    # percentage scale is kept consistent from 0 to 100
    ax.set_ylim(0, 100)

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "weighting_category_changes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # selected examples make it easier to see how the weighting behaves
    # when Reflection and Question scores agree or conflict
    scenarios = [
        (1.5, 4.5),
        (2.5, 4.0),
        (3.0, 3.0),
        (4.0, 2.5),
        (4.5, 1.5),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for reflection, questions in scenarios:

        # calculate this example under 60:40, 70:30 and 80:20
        values = [
            internal_combined_value(
                reflection,
                questions,
                WEIGHTS[name],
            )
            for name in WEIGHTS
        ]

        # one line represents one fixed Reflection/Question pair
        ax.plot(
            list(WEIGHTS.keys()),
            values,
            marker="o",
            label=f"R={reflection}, Q={questions}",
        )

    # threshold lines show where a numerical shift could change category
    ax.axhline(
        STRAIN_THRESHOLD,
        linestyle="--",
    )

    ax.axhline(
        STEADY_THRESHOLD,
        linestyle="--",
    )

    ax.set_title(
        "Effect of Weighting on Internal Combined Values"
    )
    ax.set_ylabel(
        "Internal Combined Value"
    )

    # all component and combined values use the same 1 to 5 range
    ax.set_ylim(1, 5)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "weighting_example_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# create a report-friendly image version of the main summary table
def save_table(summary):

    # only include the most useful measures in the visual table
    table_df = summary[[
        "Weighting",
        "Mean Absolute Internal Value Difference",
        "Maximum Internal Value Difference",
        "Indicator Agreement",
        "Indicator Change Rate",
    ]].copy()

    # shorter headings make the figure easier to fit into a report
    table_df.columns = [
        "Weight",
        "Mean Diff",
        "Max Diff",
        "Agreement",
        "Change Rate",
    ]

    fig, ax = plt.subplots(figsize=(8, 2.5))

    # hide normal chart axes because this figure contains only a table
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    # set the table size manually so it remains readable when exported
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title(
        "Weighting Sensitivity Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "weighting_sensitivity_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the complete sensitivity analysis and save every result file
def main():
    # simple console heading makes the script output easier to read
    print("=" * 60)
    print("WEIGHTING SENSITIVITY ANALYSIS")
    print("=" * 60)

    # first calculate every score combination
    results = build_results()

    # then reduce the detailed results into the two main summaries
    summary = create_summary(results)
    disagreement = disagreement_summary(results)


    # full output keeps every Reflection/Question combination
    results.to_csv(
        RESULTS_FOLDER / "weighting_detailed_results.csv",
        index=False,
    )

    # smaller file gives the overall comparison against 70:30
    summary.to_csv(
        RESULTS_FOLDER / "weighting_summary.csv",
        index=False,
    )

    # this file shows how sensitivity changes as the two scores move apart
    disagreement.to_csv(
        RESULTS_FOLDER / "weighting_disagreement_summary.csv",
        index=False,
    )


    # save selected examples and report-ready visuals
    save_example_cases(results)
    save_graphs(results, summary)
    save_table(summary)


    # print the main findings so they can be checked immediately after a run
    print(
        f"\nReflection-Question combinations tested: {len(results)}"
    )

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print("\nDISAGREEMENT ANALYSIS")
    print(disagreement.to_string(index=False))

    print(
        f"\nResults saved in:\n{RESULTS_FOLDER}"
    )


# run main only when this file is executed directly
if __name__ == "__main__":
    main()