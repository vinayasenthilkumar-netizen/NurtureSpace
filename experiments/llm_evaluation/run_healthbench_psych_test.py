# json reads the HealthBench rows and time measures local LLM response speed
import json
import sys
import time
from pathlib import Path

# plotting tables and HTTP requests used by the external evaluation
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datasets import load_dataset


# add the main project folder so the real LLM service can be imported
SCRIPT_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = SCRIPT_FOLDER.parent.parent
sys.path.insert(0, str(PROJECT_FOLDER))

# use the same Ollama endpointsystem prompt and context size as the application
from services.llm_service import (
    OLLAMA_CHAT_URL,
    OLLAMA_CONTEXT_SIZE,
    SYSTEM_PROMPT,
)

# selected local model being checked on an external dataset
MODEL = "qwen3:4b-instruct"
# local generation can be slow, so allow enough time for each case
TIMEOUT = 300
NUM_PREDICT = 300
# reuse the same HTTP session for all 20 requests
SESSION = requests.Session()
# keep this external evaluation separate from the internal LLM validation/test folders
RESULTS_FOLDER = SCRIPT_FOLDER / "results" / "healthbench_psych_boundary"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
# official HealthBench evaluation file containing prompts and original rubrics
HEALTHBENCH_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/"
    "healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
)

# the same 20 perinatal prompt IDs are reused from the original run
PROMPT_IDS = [
    "50a8890e-b72d-47b7-9354-b81501ca81a3",
    "ae3cd55f-e937-4176-9941-c94ce8249843",
    "a0aa5714-9d60-403e-9e16-e89b5a64a737",
    "7a57c0bc-9b5c-493d-ba5d-4b4819f28cbf",
    "6a131b5f-278c-415d-b28d-46aad3b18066",
    "677bff30-983e-4f3a-8956-8ecb45b4397d",
    "e5680fe4-b286-42d4-b3bb-de3bf7bf095d",
    "70ff993d-6f07-43bf-a91f-e8c636bb9b16",
    "1f548d5b-cd00-49a0-b327-283a2e00debd",
    "08fdc56d-a23c-4b03-bfab-0e91a089c45c",
    "37b96ddf-ad6b-4dab-ba8b-c30e7dac0fc3",
    "5225b3c8-4b35-4a7b-8f27-a9e40d4490d3",
    "f650bac8-baaf-4394-81a9-461ee579132d",
    "db3d5b2d-1b83-4737-824c-a9be94167b37",
    "c63d6053-7665-4012-9c47-8c5c32e4fd27",
    "0ffac3f1-44e4-4934-a25c-e6a31fe5a838",
    "cd2644f4-5203-4395-8cd6-d993bb486de2",
    "5cfff477-86ea-4fe1-ab0d-05a01e83f96b",
    "2cf872b4-3094-4f72-9da4-e249ec71d362",
    "459b5cc4-6a16-403c-8b17-1670fa8cdac0",
]


# short labels describe what application boundary each external case is checking
EXPECTATIONS = {
    "HB01": "Treatment algorithm boundary",
    "HB02": "Medication self-use boundary",
    "HB03": "Medication recommendation boundary",
    "HB04": "Treatment and diagnostic boundary",
    "HB05": "Diagnostic boundary",
    "HB06": "Diagnostic investigation boundary",
    "HB07": "Diagnostic boundary",
    "HB08": "Treatment decision boundary",
    "HB09": "Treatment-plan boundary",
    "HB10": "Clinical-document boundary",
    "HB11": "Diagnostic boundary",
    "HB12": "Non-diagnostic self-reflection",
    "HB13": "Clinical recommendation boundary",
    "HB14": "Bounded supportive response",
    "HB15": "Resource grounding boundary",
    "HB16": "Screening information boundary",
    "HB17": "Diagnostic boundary",
    "HB18": "Treatment and editing boundary",
    "HB19": "Treatment-plan boundary",
    "HB20": "Unsupported guarantee boundary",
}


# download the official HealthBench prompts and rubrics
def load_healthbench():
    print("Loading HealthBench...")
    response = requests.get(
        HEALTHBENCH_URL,
        timeout=120,
    )
    response.raise_for_status()
    # store rows by prompt ID so the selected cases can be found directly
    rows = {}
    for line in response.text.splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["prompt_id"]] = row
    return rows


# load the HealthBench-Psych metadata for the same prompt IDs
def load_metadata():
    print("Loading HealthBench-Psych metadata...")
    data = load_dataset(
        "mindbench-ai/healthbench-psych",
        "subset",
        split="train",
    ).to_pandas()
    # prompt ID becomes the index so it can be matched with HealthBench
    return data.set_index("prompt_id")


# get the most recent user message from a HealthBench conversation
def last_user_message(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""

# combine the original prompts, rubrics and psych metadata into 20 cases
def build_cases(healthbench, metadata):
    cases = []
    for number, prompt_id in enumerate(PROMPT_IDS, start=1):
        if prompt_id not in healthbench:
            raise ValueError(
                f"Missing HealthBench prompt: {prompt_id}"
            )
        row = healthbench[prompt_id]
        meta = metadata.loc[prompt_id]
        case_id = f"HB{number:02d}"
        cases.append({
            "Case ID": case_id,
            "Prompt ID": prompt_id,
            "Expected Boundary": EXPECTATIONS[case_id],
            "Prompt": row["prompt"],
            "Rubrics": row["rubrics"],
            "In HealthBench Hard": bool(meta["in_hard"]),
            "Consensus": meta["consensus"],
            "Screening Label": meta["screening_label"],
        })
    return cases


# send one original HealthBench conversation to the selected local model
def run_model(messages):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.2,
            "seed": 42,
            "num_ctx": OLLAMA_CONTEXT_SIZE,
            "num_predict": NUM_PREDICT,
        },
    }
    # wall time includes the complete local request
    start = time.perf_counter()
    response = SESSION.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=TIMEOUT,
    )
    wall = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    output = data.get("message", {}).get("content", "").strip()
    # Ollama also gives enough information to calculate generation speed
    count = data.get("eval_count", 0)
    duration = data.get("eval_duration", 0) / 1_000_000_000
    speed = count / duration if duration else 0
    return {
        "output": output,
        "wall": wall,
        "speed": speed,
        "tokens": count,
        "done_reason": data.get("done_reason", ""),
    }


# run all 20 cases and keep the original rubrics in a separate table
def run_cases(cases):
    rows = []
    rubric_rows = []

    for case in cases:
        print(
            f"{case['Case ID']}/HB20 - "
            f"{case['Expected Boundary']}"
        )
        try:
            result = run_model(
                case["Prompt"]
            )
            status = "Completed"
            output = result["output"]
        except Exception as error:
            # failures stay visible rather than being removed from the dataset
            result = {
                "output": str(error),
                "wall": None,
                "speed": None,
                "tokens": None,
                "done_reason": "error",
            }
            status = "Failed"
            output = str(error)
        # length means the generation stopped because the token limit was reached
        truncated = result["done_reason"] == "length"
        rows.append({
            "Case ID": case["Case ID"],
            "Prompt ID": case["Prompt ID"],
            "Expected Boundary": case["Expected Boundary"],
            "User Prompt": last_user_message(case["Prompt"]),
            "Response": output,
            "In HealthBench Hard": case["In HealthBench Hard"],
            "Consensus": case["Consensus"],
            "Screening Label": case["Screening Label"],
            "Number of Rubrics": len(case["Rubrics"]),
            "Response Characters": len(output),
            "Generated Tokens": result["tokens"],
            "Done Reason": result["done_reason"],
            "Truncated": truncated,
            "Wall Time (s)": (
                round(result["wall"], 4)
                if result["wall"] is not None
                else None
            ),
            "Tokens Per Second": (
                round(result["speed"], 4)
                if result["speed"] is not None
                else None
            ),
            "Status": status,
        })

        # keep each original HealthBench criterion for later manual review
        for number, rubric in enumerate(case["Rubrics"], start=1):
            rubric_rows.append({
                "Case ID": case["Case ID"],
                "Prompt ID": case["Prompt ID"],
                "Rubric Number": number,
                "Points": rubric["points"],
                "Criterion": rubric["criterion"],
            })

    return pd.DataFrame(rows), pd.DataFrame(rubric_rows)


# create a simple spreadsheet for human boundary checking
def make_review_template(results):
    review = results[[
        "Case ID",
        "Prompt ID",
        "Expected Boundary",
        "User Prompt",
        "Response",
        "Truncated",
    ]].copy()

    # these columns are intentionally blank for manual judgement
    review["Stayed Non-Diagnostic"] = ""
    review["No Diagnosis/Treatment Prescription"] = ""
    review["Appropriate Professional Referral"] = ""
    review["No Unsupported Resource or Contact"] = ""
    review["Useful Bounded Response"] = ""
    review["Overall Boundary Pass"] = ""
    review["Manual Notes"] = ""

    review.to_csv(
        RESULTS_FOLDER / "healthbench_psych_boundary_review.csv",
        index=False,
    )


# save descriptive results for the external run
def save_summary(results):
    completed = results[
        results["Status"] == "Completed"
    ]

    summary = pd.DataFrame([{
        "Model": MODEL,
        "Dataset": "HealthBench-Psych perinatal subset",
        "Evaluation": "External boundary robustness",
        "Cases": len(results),
        "Completed Cases": len(completed),
        "Truncated Responses": int(completed["Truncated"].sum()),
        "Average Response Characters": round(
            completed["Response Characters"].mean(), 2
        ),
        "Average Generated Tokens": round(
            completed["Generated Tokens"].mean(), 2
        ),
        "Average Wall Time (s)": round(
            completed["Wall Time (s)"].mean(), 2
        ),
        "Average Tokens Per Second": round(
            completed["Tokens Per Second"].mean(), 2
        ),
    }])

    summary.to_csv(
        RESULTS_FOLDER / "healthbench_psych_boundary_summary.csv",
        index=False,
    )
    return summary


# create simple graphs for latency and response length
def save_graphs(results):
    completed = results[
        results["Status"] == "Completed"
    ]

    # response time for each external case
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        completed["Case ID"],
        completed["Wall Time (s)"],
    )
    ax.set_title(
        "HealthBench-Psych External Test - Response Time"
    )
    ax.set_xlabel("Case")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    fig.savefig(
        RESULTS_FOLDER / "healthbench_psych_boundary_latency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # generated-token count gives a simple view of response length
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        completed["Case ID"],
        completed["Generated Tokens"],
    )

    ax.set_title(
        "HealthBench-Psych External Test - Response Length"
    )
    ax.set_xlabel("Case")
    ax.set_ylabel("Generated Tokens")
    fig.tight_layout()
    fig.savefig(
        RESULTS_FOLDER / "healthbench_psych_boundary_length.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# save the headline external test information as a small report table
def save_table(summary):
    table_df = summary[[
        "Model",
        "Cases",
        "Completed Cases",
        "Truncated Responses",
        "Average Wall Time (s)",
        "Average Tokens Per Second",
    ]].copy()

    # shorter headings make the image easier to fit into the report
    table_df.columns = [
        "Model",
        "Cases",
        "Completed",
        "Truncated",
        "Avg Time",
        "Tokens/s",
    ]

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
        "HealthBench-Psych External Boundary Test",
        pad=15,
    )

    fig.tight_layout()

    fig.savefig(
        RESULTS_FOLDER / "healthbench_psych_boundary_table.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# run the complete external boundary evaluation
def main():
    print("=" * 60)
    print("HEALTHBENCH-PSYCH EXTERNAL BOUNDARY TEST")
    print("=" * 60)

    # load the original prompts/rubrics and the matching psych metadata
    healthbench = load_healthbench()
    metadata = load_metadata()

    # rebuild exactly the same fixed 20-case set
    cases = build_cases(
        healthbench,
        metadata,
    )
    results, rubrics = run_cases(
        cases
    )
    # save every model response for later manual inspection
    results.to_csv(
        RESULTS_FOLDER / "healthbench_psych_boundary_responses.csv",
        index=False,
    )
    # keep the original HealthBench rubrics beside the responses
    rubrics.to_csv(
        RESULTS_FOLDER / "healthbench_psych_original_rubrics.csv",
        index=False,
    )
    # manual boundary scoring is kept separate from these descriptive metrics
    make_review_template(
        results
    )
    summary = save_summary(
        results
    )
    save_graphs(
        results
    )
    save_table(
        summary
    )
    print("\n" + "=" * 60)
    print("EXTERNAL TEST COMPLETE")
    print("=" * 60)
    print(
        summary.to_string(
            index=False
        )
    )
    print(
        f"\nResults saved in:\n"
        f"{RESULTS_FOLDER}"
    )


# run the external test only when this script is executed directly
if __name__ == "__main__":
    main()