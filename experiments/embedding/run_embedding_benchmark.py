"""
Compare five embedding models on the application's resource corpus.
"""

# ast/json handle stored topic formats, while gc helps release models between runs
import ast
import gc
import json
import sys
import time
from pathlib import Path

# plotting, numerical work and dataframe outputs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# torch selects gpu/cpu and sentence-transformers loads the embedding models
import torch
from sentence_transformers import SentenceTransformer


# add the project folder so the real resource database can be imported
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent.parent
sys.path.insert(0, str(PROJECT_FOLDER))

from database.resources import get_active_resources


# keep corrected validation results separate from other embedding experiments
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "validation_corrected"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


# five candidate embedding models compared before final selection
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
]


# evaluate the first three retrieved resources
TOP_K = 3

# automatically use cuda when available, otherwise fall back to cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 30 labelled application-style queries used only for model selection
CASES = [
    (
        "Q01",
        "I barely get any sleep because my newborn keeps waking up and I am exhausted.",
        [1, 28, 59, 75, 190, 333, 336],
    ),
    (
        "Q02",
        "How can I make sure my baby sleeps safely?",
        [15, 17, 73, 75, 298, 351],
    ),
    (
        "Q03",
        "My nipples are cracked and breastfeeding is painful.",
        [70, 85, 88, 134, 137, 138, 145, 317, 363],
    ),
    (
        "Q04",
        "How do I know if I am making enough breast milk?",
        [40, 66, 72, 78, 86, 352, 353],
    ),
    (
        "Q05",
        "Can I combine breastfeeding with bottle or formula feeding?",
        [68, 69, 125, 127, 175, 337, 344],
    ),
    (
        "Q06",
        "I am breastfeeding and want to know whether my medicines are safe.",
        [40, 48, 94, 232, 306, 362],
    ),
    (
        "Q07",
        "What should I eat after giving birth to help my recovery?",
        [3, 58, 290, 292, 318, 361],
    ),
    (
        "Q08",
        "What should I expect while recovering from a vaginal birth?",
        [30, 38, 54, 58, 83, 87, 135],
    ),
    (
        "Q09",
        "I am recovering from a C-section and still have pain.",
        [38, 55, 135, 144, 359],
    ),
    (
        "Q10",
        "I leak urine after giving birth and want help with my pelvic floor.",
        [34, 37, 57, 79, 83, 135, 277],
    ),
    (
        "Q11",
        "What happens at my six-week postnatal check?",
        [35, 43, 77, 165],
    ),
    (
        "Q12",
        "How can I tell the difference between baby blues and postnatal depression?",
        [21, 56, 74, 181, 230, 332, 358],
    ),
    (
        "Q13",
        "I keep worrying and feeling anxious since having my baby.",
        [63, 182, 184, 185, 262, 270, 348],
    ),
    (
        "Q14",
        "How can my partner support me after the baby arrives?",
        [46, 65, 101, 168, 212, 301, 334],
    ),
    (
        "Q15",
        "I am parenting mostly on my own and need some support.",
        [33, 36, 46, 53, 100, 124, 170],
    ),
    (
        "Q16",
        "I never seem to have any time to look after myself anymore.",
        [2, 8, 12, 13, 31, 200, 231, 315],
    ),
    (
        "Q17",
        "I am going back to work and need advice about pumping and breastfeeding.",
        [40, 67, 314, 316, 353, 369, 377],
    ),
    (
        "Q18",
        "I am worried about returning to work after having my baby.",
        [6, 10, 262, 293, 369, 376, 377],
    ),
    (
        "Q19",
        "My baby will not stop crying and I do not know how to soothe them.",
        [18, 76, 92, 95, 98, 251, 342],
    ),
    (
        "Q20",
        "My baby has colic and cries for long periods.",
        [76, 95, 98, 275, 342, 370],
    ),
    (
        "Q21",
        "I need basic guidance on caring for my newborn at home.",
        [16, 24, 29, 61, 90, 91, 219, 351, 365, 372],
    ),
    (
        "Q22",
        "How can I support my baby's development and milestones?",
        [102, 103, 118, 166, 213, 250, 254, 328],
    ),
    (
        "Q23",
        "What warning signs after childbirth mean I should seek medical help?",
        [23, 44, 45, 50, 54, 159],
    ),
    (
        "Q24",
        "I feel like becoming a mother has changed who I am.",
        [162, 164, 179, 194, 195, 260, 297, 308, 347],
    ),
    (
        "Q25",
        "I keep having unwanted frightening thoughts about my baby.",
        [169, 188, 189, 259, 341],
    ),
    (
        "Q26",
        "I feel guilty and ashamed that I am not being a good enough mother.",
        [13, 198, 197, 259, 337],
    ),
    (
        "Q27",
        "I feel overwhelmed carrying most of the household and parenting mental load.",
        [6, 10, 13, 196, 204, 324],
    ),
    (
        "Q28",
        "I am struggling with breastfeeding and want support from a lactation professional.",
        [36, 52, 100, 126, 128, 192, 208, 340, 353],
    ),
    (
        "Q29",
        "I am nervous about introducing solid food to my baby.",
        [20, 177, 220, 367],
    ),
    (
        "Q30",
        "I am still sore and in pain while recovering after childbirth.",
        [30, 38, 54, 55, 83, 135, 144, 359],
    ),
]


# topics may be stored as a list, JSON string or Python-style list string
def parse_topics(value):
    if isinstance(value, list):
        return value

    if not value:
        return []

    try:
        parsed = json.loads(value)

    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return [str(value)]

    return parsed if isinstance(parsed, list) else [str(parsed)]


# build the text representation of every active resource
def load_resources():
    resources = [
        dict(row)
        for row in get_active_resources()
    ]

    docs = []

    for resource in resources:
        topics = ", ".join(
            parse_topics(resource.get("topics"))
        )

        # title, topics and content are all included in the resource embedding
        docs.append(
            f"{resource.get('title', '')}\n"
            f"Topics: {topics}\n"
            f"{resource.get('content', '') or ''}"
        )

    return resources, docs


# E5 expects passage prefixes for corpus documents
def format_docs(model_name, docs):
    if model_name.startswith("intfloat/e5"):
        return [
            f"passage: {doc}"
            for doc in docs
        ]

    return docs


# apply the query format recommended for E5 and BGE models
def format_query(model_name, query):
    if model_name.startswith("intfloat/e5"):
        return f"query: {query}"

    if model_name.startswith("BAAI/bge"):
        return (
            "Represent this sentence for searching relevant passages: "
            + query
        )

    return query


# encode one query and rank the full resource corpus by similarity
def rank_query(model, model_name, query, corpus_embeddings):
    query = format_query(
        model_name,
        query,
    )

    start = time.perf_counter()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # normalised embeddings allow dot product to act as cosine similarity
    scores = corpus_embeddings @ query_embedding
    ranking = np.argsort(-scores)

    query_time = time.perf_counter() - start

    return ranking, scores, query_time


# evaluate one candidate embedding model on all 30 queries
def evaluate_model(model_name, resources, docs):
    print(f"\n{'=' * 65}")
    print(model_name)
    print("=" * 65)

    # model load time is recorded separately from retrieval time
    start = time.perf_counter()

    model = SentenceTransformer(
        model_name,
        device=DEVICE,
    )

    load_time = time.perf_counter() - start

    # encode the resource corpus once for this candidate model
    start = time.perf_counter()

    corpus_embeddings = model.encode(
        format_docs(model_name, docs),
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    encode_time = time.perf_counter() - start
    rows = []

    for case_id, query, relevant_ids in CASES:
        ranking, scores, query_time = rank_query(
            model,
            model_name,
            query,
            corpus_embeddings,
        )

        # convert embedding positions back into resource database IDs
        ranked_ids = [
            int(resources[index]["id"])
            for index in ranking
        ]

        top_indices = ranking[:TOP_K]
        top_ids = ranked_ids[:TOP_K]

        # search the full ranking for the first relevant resource
        first_rank = next(
            (
                rank
                for rank, resource_id in enumerate(ranked_ids, start=1)
                if resource_id in relevant_ids
            ),
            None,
        )

        # Hit@1 checks whether the first result is relevant
        hit1 = int(
            top_ids[0] in relevant_ids
        )

        # Hit@3 checks whether at least one top-three result is relevant
        hit3 = int(
            any(resource_id in relevant_ids for resource_id in top_ids)
        )

        # Precision@3 measures how many of the top three are relevant
        precision3 = (
            sum(resource_id in relevant_ids for resource_id in top_ids)
            / TOP_K
        )

        # reciprocal rank rewards the first relevant result appearing near the top
        rr = 1 / first_rank if first_rank else 0

        rows.append({
            "Model": model_name,
            "Case ID": case_id,
            "Query": query,
            "Relevant IDs": ", ".join(map(str, relevant_ids)),

            "Top 1 ID": top_ids[0],
            "Top 1 Title": resources[top_indices[0]]["title"],
            "Top 2 ID": top_ids[1],
            "Top 2 Title": resources[top_indices[1]]["title"],
            "Top 3 ID": top_ids[2],
            "Top 3 Title": resources[top_indices[2]]["title"],

            "Top 1 Similarity": round(
                float(scores[top_indices[0]]),
                4,
            ),
            "First Relevant Rank": first_rank,
            "Hit@1": hit1,
            "Hit@3": hit3,
            "Precision@3": round(precision3, 4),
            "Reciprocal Rank": round(rr, 4),

            "Query Time (ms)": round(
                query_time * 1000,
                3,
            ),
            "Model Load Time (s)": round(
                load_time,
                3,
            ),
            "Corpus Encode Time (s)": round(
                encode_time,
                3,
            ),
        })

        print(
            f"{case_id}: top={top_ids} "
            f"hit1={hit1} hit3={hit3} rr={rr:.3f}"
        )

    # store embedding size before releasing the model
    dimension = corpus_embeddings.shape[1]

    # free memory before loading the next candidate
    del model, corpus_embeddings
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows, dimension


# combine detailed query results into one summary row per model
def create_comparison(results, dimensions):
    rows = []

    for model_name in MODELS:
        df = results[
            results["Model"] == model_name
        ]

        rows.append({
            "Model": model_name,
            "Embedding Dimension": dimensions[model_name],
            "Hit@1": round(df["Hit@1"].mean(), 4),
            "Hit@3": round(df["Hit@3"].mean(), 4),
            "Precision@3": round(df["Precision@3"].mean(), 4),
            "MRR": round(df["Reciprocal Rank"].mean(), 4),
            "Average Query Time (ms)": round(
                df["Query Time (ms)"].mean(),
                3,
            ),
            "Corpus Encode Time (s)": round(
                df["Corpus Encode Time (s)"].iloc[0],
                3,
            ),
            "Model Load Time (s)": round(
                df["Model Load Time (s)"].iloc[0],
                3,
            ),
        })

    comparison = pd.DataFrame(rows)

    # prioritise retrieval quality, then use query time as the final tie-breaker
    return comparison.sort_values(
        [
            "MRR",
            "Hit@3",
            "Precision@3",
            "Hit@1",
            "Average Query Time (ms)",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )


# save the labelled validation cases separately from model results
def save_queries():
    rows = [
        {
            "Case ID": case_id,
            "Query": query,
            "Relevant Resource IDs": ", ".join(map(str, relevant_ids)),
        }
        for case_id, query, relevant_ids in CASES
    ]

    pd.DataFrame(rows).to_csv(
        RESULTS_FOLDER / "embedding_validation_queries.csv",
        index=False,
    )


# create the main model-comparison graphs
def save_graphs(df):
    # use shorter model names on chart labels
    labels = df["Model"].str.split("/").str[-1]

    x = np.arange(len(df))
    width = 0.22

    # compare the four main retrieval-quality metrics
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        x - 1.5 * width,
        df["Hit@1"],
        width,
        label="Hit@1",
    )

    ax.bar(
        x - 0.5 * width,
        df["Hit@3"],
        width,
        label="Hit@3",
    )

    ax.bar(
        x + 0.5 * width,
        df["Precision@3"],
        width,
        label="Precision@3",
    )

    ax.bar(
        x + 1.5 * width,
        df["MRR"],
        width,
        label="MRR",
    )

    ax.set_title(
        "Embedding Model Validation"
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "embedding_validation_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # compare average query latency
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        labels,
        df["Average Query Time (ms)"],
    )

    ax.set_title(
        "Embedding Query Time"
    )
    ax.set_ylabel("Milliseconds")
    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "embedding_validation_latency.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# save a compact report-ready comparison table
def save_table(df):
    table_df = df[[
        "Model",
        "Hit@1",
        "Hit@3",
        "Precision@3",
        "MRR",
        "Average Query Time (ms)",
    ]].copy()

    # remove organisation prefixes to keep the table readable
    table_df["Model"] = (
        table_df["Model"]
        .str.split("/")
        .str[-1]
    )

    table_df.columns = [
        "Model",
        "Hit@1",
        "Hit@3",
        "Precision@3",
        "MRR",
        "Query ms",
    ]

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    ax.set_title(
        "Embedding Model Validation Results",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "embedding_validation_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the complete five-model validation experiment
def main():
    print("=" * 65)
    print("CORRECTED EMBEDDING MODEL VALIDATION")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    # load the same active resource corpus used by the application
    resources, docs = load_resources()

    print(
        f"Active resources: "
        f"{len(resources)}"
    )

    print(
        f"Validation queries: "
        f"{len(CASES)}"
    )

    # check that every labelled relevant resource still exists
    available_ids = {
        int(resource["id"])
        for resource in resources
    }

    for case_id, _, relevant_ids in CASES:
        missing = set(relevant_ids) - available_ids

        if missing:
            raise ValueError(
                f"{case_id} references missing resources: {missing}"
            )

    # save the labelled cases before running model comparison
    save_queries()

    rows = []
    dimensions = {}

    # every candidate is evaluated on the same corpus and same 30 queries
    for model_name in MODELS:
        model_rows, dimension = evaluate_model(
            model_name,
            resources,
            docs,
        )

        rows.extend(model_rows)
        dimensions[model_name] = dimension

    # detailed results keep every query and retrieved top-three resource
    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_FOLDER / "embedding_validation_detailed_results.csv",
        index=False,
    )

    # create one summary row per candidate model
    comparison = create_comparison(
        results,
        dimensions,
    )

    comparison.to_csv(
        RESULTS_FOLDER / "embedding_validation_model_comparison.csv",
        index=False,
    )

    # produce report-friendly evidence
    save_graphs(comparison)
    save_table(comparison)

    print("\n" + "=" * 65)
    print("CORRECTED EMBEDDING VALIDATION RESULTS")
    print("=" * 65)

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved in:\n"
        f"{RESULTS_FOLDER}"
    )


# run validation only when this script is executed directly
if __name__ == "__main__":
    main()