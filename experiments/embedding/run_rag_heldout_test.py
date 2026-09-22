"""
Held-out evaluation of the FINAL resource retriever.

Purpose
-------
The earlier embedding benchmark compared candidate embedding models on 30
labelled queries and selected BAAI/bge-small-en-v1.5. This script must be run
AFTER that selection is frozen. It evaluates only the final deployed
retrieval configuration (selected BGE model + FAISS retriever) on 15 NEW,
pre-labelled queries.

Important methodology rule
--------------------------
Do not change the query wording or relevant resource IDs after seeing the
retrieval results. If a relevance judgement is genuinely wrong, document the
change and rerun the whole held-out test as a new version.

Recommended location
--------------------
Place this file in the same experiment folder as run_embedding_benchmark.py.
The script automatically locates the project root.
"""

from __future__ import annotations

# system and timing helpers used by the experiment
import sys
import time
from pathlib import Path

# plotting and dataframe libraries used for the final evidence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# folder containing this experiment script
SCRIPT_FOLDER = Path(__file__).resolve().parent


# find the project root without depending on one fixed machine path
def find_project_root(start: Path) -> Path: 
    candidates = [start, *start.parents]

    for candidate in candidates:
        # these folders identify the main application project
        if (
            (candidate / "database").is_dir()
            and (candidate / "rag").is_dir()
            and (candidate / "config").is_dir()
        ):
            return candidate

    raise RuntimeError(
        "Could not find the project root. Place this script inside the "
        "project (for example experiments/embedding/) and try again."
    )


# add the project root so the real application modules can be imported
PROJECT_FOLDER = find_project_root(SCRIPT_FOLDER)
sys.path.insert(0, str(PROJECT_FOLDER))

# use the deployed embedding setting, live resource catalogue and final retriever
from config.settings import SELECTED_EMBEDDING_MODEL
from database.resources import get_active_resources
from rag.retriever import load_embedding_model, retrieve_resources


# keep held out results separate from the earlier model selection experiment
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "heldout_test"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# user facing retrieval normally shows the top three resources
TOP_K = 3

# version label helps document any future rerun of the frozen test
HELDOUT_VERSION = "heldout_v1"


# these queries were kept separate from the 30 model selection queries
# relevant IDs were assigned before running retrieval
CASES = [
    (
        "H01",
        "My newborn's skin looks yellow. Where can I read about jaundice and when to get help?",
        [22, 96, 357],
    ),
    (
        "H02",
        "Why does my newborn need a hearing screening test and what happens if the result is abnormal?",
        [354],
    ),
    (
        "H03",
        "What is newborn metabolic screening and why is it done soon after birth?",
        [356],
    ),
    (
        "H04",
        "What contraception or family-planning options can I consider after giving birth, including while breastfeeding?",
        [39, 41, 64, 81, 84, 233, 355],
    ),
    (
        "H05",
        "I want information about my newborn's vaccinations, including BCG and routine immunisations.",
        [104, 119, 360],
    ),
    (
        "H06",
        "How should I clean and look after my baby's umbilical cord stump?",
        [248, 351, 372],
    ),
    (
        "H07",
        "How is my baby's growth measured and monitored during the first year?",
        [103, 118, 373],
    ),
    (
        "H08",
        "When can I gradually start exercising again after giving birth?",
        [26, 31, 37, 57],
    ),
    (
        "H09",
        "Is bleeding after childbirth expected and what physical changes should I know about?",
        [30, 54, 87],
    ),
    (
        "H10",
        "I want practical ideas for bonding and connecting with my newborn in the early days.",
        [24, 91, 320],
    ),
    (
        "H11",
        "How can I make my home and everyday travel safer for my young baby?",
        [19, 89],
    ),
    (
        "H12",
        "Can I drink coffee while breastfeeding and what food or hydration guidance should I follow?",
        [361],
    ),
    (
        "H13",
        "I want to stop breastfeeding gradually. How can I wean and replace feeds safely?",
        [364],
    ),
    (
        "H14",
        "When might my periods and fertility return while I am breastfeeding?",
        [39, 64, 81, 84, 355],
    ),
    (
        "H15",
        "My breast is painful while breastfeeding and I am worried about a blocked duct or mastitis.",
        [70, 80, 85],
    ),
]


# make sure every frozen relevance label still exists in the final resource database
def validate_cases(resources: list[dict]) -> None:
    """Fail early if a frozen relevance ID is absent from the final corpus."""

    available_ids = {
        int(resource["id"])
        for resource in resources
    }

    for case_id, _query, relevant_ids in CASES:
        missing = set(relevant_ids) - available_ids

        if missing:
            raise ValueError(
                f"{case_id} contains missing resource IDs: {sorted(missing)}"
            )


# save the frozen queries separately so they are not mixed with retrieval results
def save_case_manifest() -> None:
    manifest = [
        {
            "Case ID": case_id,
            "Query": query,
            "Relevant Resource IDs": ", ".join(map(str, relevant_ids)),
            "Held-out Version": HELDOUT_VERSION,
        }
        for case_id, query, relevant_ids in CASES
    ]

    pd.DataFrame(manifest).to_csv(
        RESULTS_FOLDER / "rag_heldout_cases.csv",
        index=False,
    )


# run every held out query through the final retriever
def evaluate(resources: list[dict], embedding_model) -> pd.DataFrame:
    rows = []

    # search the full corpus so first relevant rank and MRR are meaningful
    full_k = len(resources)

    for case_id, query, relevant_ids in CASES:
        start = time.perf_counter()

        ranked = retrieve_resources(
            query=query,
            resources=resources,
            embedding_model=embedding_model,
            top_k=full_k,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # keep the full ranking for MRR but only the top three for displayed metrics
        ranked_ids = [int(item["id"]) for item in ranked]
        top_three = ranked[:TOP_K]
        top_three_ids = ranked_ids[:TOP_K]

        # find where the first relevant resource appears anywhere in the ranking
        first_rank = next(
            (
                rank
                for rank, resource_id in enumerate(ranked_ids, start=1)
                if resource_id in relevant_ids
            ),
            None,
        )

        # Hit@1 checks only the first result
        hit1 = int(
            bool(top_three_ids)
            and top_three_ids[0] in relevant_ids
        )

        # Hit@3 checks whether at least one of the first three is relevant
        hit3 = int(
            any(resource_id in relevant_ids for resource_id in top_three_ids)
        )

        # Precision@3 measures how many of the three returned resources are relevant
        precision3 = (
            sum(resource_id in relevant_ids for resource_id in top_three_ids)
            / TOP_K
        )

        # reciprocal rank rewards relevant resources appearing near the top
        reciprocal_rank = 1 / first_rank if first_rank else 0.0

        # small helper avoids repeated index checks for top three evidence
        def item_value(index: int, key: str, default=""):
            if index >= len(top_three):
                return default

            return top_three[index].get(key, default)

        rows.append({
            "Case ID": case_id,
            "Query": query,
            "Relevant IDs": ", ".join(map(str, relevant_ids)),

            "Top 1 ID": item_value(0, "id"),
            "Top 1 Title": item_value(0, "title"),
            "Top 1 Score": item_value(0, "retrieval_score"),

            "Top 2 ID": item_value(1, "id"),
            "Top 2 Title": item_value(1, "title"),
            "Top 2 Score": item_value(1, "retrieval_score"),

            "Top 3 ID": item_value(2, "id"),
            "Top 3 Title": item_value(2, "title"),
            "Top 3 Score": item_value(2, "retrieval_score"),

            "First Relevant Rank": first_rank,
            "Hit@1": hit1,
            "Hit@3": hit3,
            "Precision@3": round(precision3, 4),
            "Reciprocal Rank": round(reciprocal_rank, 4),
            "Query Time (ms)": round(elapsed_ms, 3),
        })

        # print a short result line while the experiment runs
        print(
            f"{case_id}: top={top_three_ids} | "
            f"hit1={hit1} hit3={hit3} rr={reciprocal_rank:.3f}"
        )

    return pd.DataFrame(rows)


# combine the 15 query results into one final retriever summary
def make_summary(
    results: pd.DataFrame,
    resource_count: int,
) -> pd.DataFrame:

    return pd.DataFrame([{
        "Model": SELECTED_EMBEDDING_MODEL,
        "Retriever": "FAISS IndexFlatIP with normalised embeddings",
        "Held-out Version": HELDOUT_VERSION,
        "Resource Count": resource_count,
        "Held-out Queries": len(results),

        # means across all held-out queries form the main retrieval metrics
        "Hit@1": round(results["Hit@1"].mean(), 4),
        "Hit@3": round(results["Hit@3"].mean(), 4),
        "Precision@3": round(results["Precision@3"].mean(), 4),
        "MRR": round(results["Reciprocal Rank"].mean(), 4),

        # keep both mean and median query time because latency can vary
        "Average Query Time (ms)": round(
            results["Query Time (ms)"].mean(), 3
        ),
        "Median Query Time (ms)": round(
            results["Query Time (ms)"].median(), 3
        ),
    }])


# graph the main held out retrieval metrics
def save_metric_graph(summary: pd.DataFrame) -> None:
    metrics = [
        "Hit@1",
        "Hit@3",
        "Precision@3",
        "MRR",
    ]

    values = [
        float(summary.iloc[0][metric])
        for metric in metrics
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metrics, values)

    ax.set_title(
        "Selected BGE + FAISS Held out Retrieval Performance"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "rag_heldout_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# show how far down the ranking the first relevant resource appeared
def save_rank_graph(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        results["Case ID"],
        results["First Relevant Rank"],
    )

    ax.set_title(
        "Held-out RAG - First Relevant Resource Rank"
    )
    ax.set_xlabel("Held-out query")
    ax.set_ylabel("Rank (lower is better)")
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "rag_heldout_first_relevant_rank.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save a compact report ready table of the main metrics
def save_summary_table(summary: pd.DataFrame) -> None:
    table_df = summary[[
        "Model",
        "Held-out Queries",
        "Hit@1",
        "Hit@3",
        "Precision@3",
        "MRR",
    ]].copy()

    # remove the Hugging Face organisation prefix for a cleaner table
    table_df["Model"] = (
        table_df["Model"]
        .str.split("/")
        .str[-1]
    )

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
        "Selected Retriever - Held-out Test Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "rag_heldout_results_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the complete final retriever test
def main() -> None:
    print("=" * 68)
    print("FINAL BGE + FAISS HELD-OUT RETRIEVAL TEST")
    print("=" * 68)
    print(
        f"Selected model: "
        f"{SELECTED_EMBEDDING_MODEL}"
    )
    # use the final active resource catalogue from the application database
    resources = [
        dict(row)
        for row in get_active_resources()
    ]
    print(
        f"Active resources: "
        f"{len(resources)}"
    )
    print(
        f"Held-out queries: "
        f"{len(CASES)}"
    )

    # check the frozen relevance labels before retrieving anything
    validate_cases(resources)
    # save the cases independently from their results
    save_case_manifest()
    print("\nLoading selected embedding model...")
    # model loading is timed separately from query retrieval
    load_start = time.perf_counter()
    embedding_model = load_embedding_model(
        SELECTED_EMBEDDING_MODEL
    )
    load_time = time.perf_counter() - load_start
    print(
        f"Model ready in "
        f"{load_time:.2f}s\n"
    )
    # run all 15 held-out queries through the final deployed retriever
    results = evaluate(
        resources,
        embedding_model,
    )
    summary = make_summary(
        results,
        len(resources),
    )
    summary["Model Load Time (s)"] = round(
        load_time,
        3,
    )
    # detailed file keeps the retrieved evidence for every query
    results.to_csv(
        RESULTS_FOLDER / "rag_heldout_detailed_results.csv",
        index=False,
    )

    # summary file contains the final headline retrieval metrics
    summary.to_csv(
        RESULTS_FOLDER / "rag_heldout_summary.csv",
        index=False,
    )

    # create report evidence figures
    save_metric_graph(summary)
    save_rank_graph(results)
    save_summary_table(summary)

    print("\n" + "=" * 68)
    print("HELD-OUT RETRIEVAL RESULTS")
    print("=" * 68)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved in:\n"
        f"{RESULTS_FOLDER}"
    )


# run the held out experiment only when this script is executed directly
if __name__ == "__main__":
    main()