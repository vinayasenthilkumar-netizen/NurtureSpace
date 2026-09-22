# path, json and timing helpers used by the pipeline experiment
from pathlib import Path
import json
import sys
import time

# plotting and dataframe libraries used for experiment results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# add the main project folder so application modules can be imported
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# reflection modes used by the real application
from core.constants import TEXT_REFLECTION, VOICE_REFLECTION, VIDEO_REFLECTION

# use the actual guided workflow instead of recreating the pipeline for testing
from graphs.guided_workflow_graph import (
    GUIDED_STAGE_REFLECTION,
    GUIDED_STAGE_REVIEW,
    GUIDED_STAGE_STRUCTURED,
    guided_workflow_graph,
)
# temporary voice and video files are cleaned after each case
from services.video_processing import cleanup_video_result
from services.voice_processing import cleanup_voice_result
# experiment input and output folders
DATA = ROOT / "experiments" / "pipeline" / "data"
RESULTS = ROOT / "experiments" / "pipeline" / "results"
AUDIO_ROOT = ROOT / "ravdess_audio"
VIDEO_ROOT = ROOT / "experiments" / "video_expression" / "data"
RESULTS.mkdir(parents=True, exist_ok=True)


# keep context questions neutral so the main test focuses on the pipeline
CONTEXT_ANSWERS = {
    "food": 3,
    "medication": 3,
    "physical_recovery": 3,
    "baby_care": 3,
    "other_responsibilities": 3,
    "personal_care": 3,
}

# rotate through different questionnaire conditions across the test cases
QUESTION_SETS = [
    ("steady", {"sleep": 4, "mood": 4, "stress": 2, "support": 4}),
    ("middle", {"sleep": 3, "mood": 3, "stress": 3, "support": 3}),
    ("strain", {"sleep": 2, "mood": 2, "stress": 4, "support": 2}),
    ("mixed", {"sleep": 4, "mood": 3, "stress": 4, "support": 3}),
]

# selected RAVDESS emotions used for audio and video pipeline cases
RAVDESS_EMOTIONS = {
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
}

# split a RAVDESS filename into its seven coded sections
def ravdess_parts(path):
    parts = path.stem.split("-")
    return parts if len(parts) == 7 else None

# choose one suitable RAVDESS file for each target emotion
def pick_ravdess(root, modality, extensions):
    files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    chosen = []
    used_actors = set()
    for code, emotion in RAVDESS_EMOTIONS.items():
        matches = []
        for path in files:
            parts = ravdess_parts(path)
            if parts and parts[0] == modality and parts[1] == "01" and parts[2] == code:
                matches.append(path)
        matches.sort()
        if not matches:
            continue
        # prefer a different actor for each emotion where possible
        path = next(
            (p for p in matches if p.parent.name not in used_actors),
            matches[0],
        )
        used_actors.add(path.parent.name)
        chosen.append((path, emotion))
    return chosen


# independently recalculate the reflection score to check LangGraph's result
def expected_reflection(text_score, audio_score, video_score):
    t, a, v = text_score, audio_score, video_score
    if t is not None and a is not None and v is not None:
        score = t * 0.40 + a * 0.40 + v * 0.20
    elif t is not None and v is not None:
        score = t * 0.80 + v * 0.20
    elif a is not None and v is not None:
        score = a * 0.80 + v * 0.20
    elif t is not None and a is not None:
        score = t * 0.50 + a * 0.50
    elif t is not None:
        score = t
    elif a is not None:
        score = a
    else:
        return None

    return round(score, 2)

# independently check the final 70:30 reflection/question weighting
def expected_combined(reflection_score, question_score):
    if reflection_score is None or question_score is None:
        return None
    return round(reflection_score * 0.70 + question_score * 0.30, 2)

# allow for a very small floating-point difference
def same_score(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 0.001


# build the full set of text, voice and video pipeline cases
def build_cases():
    cases = []
    # four own-written text reflections
    text_files = sorted((DATA / "own_text").glob("*.txt"))
    if len(text_files) < 4:
        raise RuntimeError("Add at least 4 .txt files to data/own_text.")
    for i, path in enumerate(text_files[:4], 1):
        cases.append({
            "case": f"T{i:02}",
            "mode": TEXT_REFLECTION,
            "source_group": "Own text",
            "path": path,
            "expected_emotion": "",
            "robustness_case": False,
        })
    # four RAVDESS voice cases
    rav_audio = pick_ravdess(AUDIO_ROOT, "03", {".wav"})
    if len(rav_audio) < 4:
        raise RuntimeError("Could not find 4 suitable RAVDESS audio files.")
    for i, (path, emotion) in enumerate(rav_audio[:4], 1):
        cases.append({
            "case": f"RA{i:02}",
            "mode": VOICE_REFLECTION,
            "source_group": "RAVDESS audio",
            "path": path,
            "expected_emotion": emotion,
            "robustness_case": False,
        })
    # two own-recorded voice cases
    own_audio = sorted(
        path for path in (DATA / "own_audio").iterdir()
        if path.suffix.lower() in {
            ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".webm", ".flac", ".aac"
        }
    )
    if len(own_audio) < 2:
        raise RuntimeError("Add at least 2 recordings to data/own_audio.")

    for i, path in enumerate(own_audio[:2], 1):
        cases.append({
            "case": f"OA{i:02}",
            "mode": VOICE_REFLECTION,
            "source_group": "Own audio",
            "path": path,
            "expected_emotion": "",
            "robustness_case": False,
        })

    # prefer full audio-video RAVDESS clips, otherwise use video-only clips
    full_av = pick_ravdess(VIDEO_ROOT, "01", {".mp4", ".avi", ".mov", ".mkv"})
    video_only = pick_ravdess(VIDEO_ROOT, "02", {".mp4", ".avi", ".mov", ".mkv"})
    rav_video = full_av[:4] if len(full_av) >= 4 else video_only[:4]
    is_video_only = len(full_av) < 4

    if len(rav_video) < 4:
        raise RuntimeError("Could not find 4 suitable RAVDESS video files.")
    group = "RAVDESS video-only" if is_video_only else "RAVDESS full-AV video"
    for i, (path, emotion) in enumerate(rav_video, 1):
        cases.append({
            "case": f"RV{i:02}",
            "mode": VIDEO_REFLECTION,
            "source_group": group,
            "path": path,
            "expected_emotion": emotion,
            "robustness_case": is_video_only,
        })
    # two own-recorded video cases
    own_video = sorted(
        path for path in (DATA / "own_video").iterdir()
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    )
    if len(own_video) < 2:
        raise RuntimeError("Add at least 2 recordings to data/own_video.")
    for i, path in enumerate(own_video[:2], 1):
        cases.append({
            "case": f"OV{i:02}",
            "mode": VIDEO_REFLECTION,
            "source_group": "Own video",
            "path": path,
            "expected_emotion": "",
            "robustness_case": False,
        })

    # give each case one of the prepared questionnaire patterns
    for i, case in enumerate(cases):
        name, answers = QUESTION_SETS[i % len(QUESTION_SETS)]
        case["question_set"] = name
        case["core_answers"] = answers
    return cases

# run the structured-question stage through the real guided workflow
def run_questions(core_answers):
    result = guided_workflow_graph.invoke({
        "workflow_stage": GUIDED_STAGE_STRUCTURED,
        "core_answers": core_answers,
        "context_answers": CONTEXT_ANSWERS,
    })
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result.get("question_score"), result.get("checkin_status", "")


# run text, voice or video through the actual reflection stage
def run_reflection(case):
    mode = case["mode"]
    path = case["path"]
    graph_input = {
        "workflow_stage": GUIDED_STAGE_REFLECTION,
        "reflection_mode": mode,
    }
    if mode == TEXT_REFLECTION:
        text = path.read_text(encoding="utf-8").strip()
        graph_input["reflection_text"] = text
        return guided_workflow_graph.invoke(graph_input), text

    key = "voice_input" if mode == VOICE_REFLECTION else "video_input"
    with path.open("rb") as media_file:
        graph_input[key] = media_file
        result = guided_workflow_graph.invoke(graph_input)
    # voice and video both need their transcript for the review stage
    if mode == VOICE_REFLECTION:
        transcript = result.get("voice_processing_result", {}).get("transcript", "")
    else:
        transcript = result.get("video_processing_result", {}).get("transcript", "")
    return result, str(transcript or "").strip()


# pass the approved reflection evidence into the review stage
def run_review(mode, text, question_score, audio_result=None, video_evidence=None):
    if not text:
        return {}
    graph_input = {
        "workflow_stage": GUIDED_STAGE_REVIEW,
        "reflection_mode": mode,
        "question_score": question_score,
    }
    if audio_result:
        graph_input["suggested_audio_observation"] = audio_result
    if video_evidence:
        graph_input["suggested_visible_expression"] = video_evidence
    if mode == TEXT_REFLECTION:
        graph_input["reflection_text"] = text
    else:
        graph_input["confirmed_transcript"] = text
    return guided_workflow_graph.invoke(graph_input)


# convert score lists into CSV-friendly JSON text
def json_value(value):
    return json.dumps(value or [], ensure_ascii=False)

# run one complete questionnaire + reflection + review pipeline case
def run_case(case):
    start = time.perf_counter()
    reflection_state = {}

    try:
        question_score, question_status = run_questions(case["core_answers"])
        reflection_state, analysis_text = run_reflection(case)

        voice_result = reflection_state.get("voice_processing_result", {}) or {}
        video_result = reflection_state.get("video_processing_result", {}) or {}

        # select the evidence relevant to the current reflection type
        if case["mode"] == VOICE_REFLECTION:
            audio_result = voice_result.get("audio_observation", {}) or {}
            video_evidence = {}

        elif case["mode"] == VIDEO_REFLECTION:
            audio_result = video_result.get("audio_observation", {}) or {}
            video_evidence = video_result.get("visible_expression", {}) or {}

        else:
            audio_result = {}
            video_evidence = {}

        review_state = run_review(
            case["mode"],
            analysis_text,
            question_score,
            audio_result,
            video_evidence,
        )

        # scores returned by the real LangGraph review stage
        text_score = review_state.get("text_component_score")
        audio_score = review_state.get("audio_component_score")
        video_score = review_state.get("video_component_score")
        reflection_score = review_state.get("reflection_score")
        combined_score = review_state.get("combined_wellbeing_score")

        # calculate the same scores independently to verify the formulas
        expected_ref = expected_reflection(text_score, audio_score, video_score)
        expected_comb = expected_combined(expected_ref, question_score)

        ref_match = same_score(reflection_score, expected_ref)
        combined_match = same_score(combined_score, expected_comb)

        review_complete = bool(review_state.get("analysis_complete", False))
        reflection_ready = bool(reflection_state.get("review_ready", False))
        fatal_error = str(reflection_state.get("error", "") or "").strip()

        # video-only RAVDESS files are treated as robustness cases
        # because they may not complete the normal audio-video path
        if case["robustness_case"]:
            pipeline_pass = (
                reflection_ready
                and not fatal_error
                and ref_match
                and combined_match
            )
        else:
            pipeline_pass = (
                reflection_ready
                and review_complete
                and combined_score is not None
                and not fatal_error
                and ref_match
                and combined_match
            )

        text_scores = review_state.get("all_text_emotion_scores", []) or []
        top_text = text_scores[0].get("category", "") if text_scores else ""

        processing_errors = (
            reflection_state.get("reflection_processing_errors", []) or []
        )

        if case["mode"] == VIDEO_REFLECTION:
            processing_errors = video_result.get("errors", []) or processing_errors

        return {
            "case": case["case"],
            "source_group": case["source_group"],
            "mode": case["mode"],
            "file": case["path"].name,
            "expected_emotion": case["expected_emotion"],

            "question_set": case["question_set"],
            "question_score": question_score,
            "question_status": question_status,
            "transcript_or_text": analysis_text,

            "top_text_category": top_text,
            "audio_raw_label": audio_result.get("raw_label", ""),
            "audio_observation": audio_result.get("observation", ""),
            "video_raw_label": video_evidence.get("dominant_raw_label", ""),
            "video_observation": video_evidence.get("observation", ""),

            "text_component_score": text_score,
            "audio_component_score": audio_score,
            "video_component_score": video_score,

            "reflection_score": reflection_score,
            "expected_reflection_score": expected_ref,
            "reflection_formula_match": ref_match,
            "reflection_indicator": review_state.get("reflection_indicator"),

            "combined_score": combined_score,
            "expected_combined_score": expected_comb,
            "combined_formula_match": combined_match,
            "combined_indicator": review_state.get("combined_wellbeing_indicator"),

            "review_complete": review_complete,
            "full_end_to_end": combined_score is not None and review_complete,
            "robustness_case": case["robustness_case"],
            "pipeline_pass": pipeline_pass,

            "processing_errors": " | ".join(
                str(error) for error in processing_errors if error
            ),

            "text_model_scores": json_value(text_scores),
            "audio_model_scores": json_value(audio_result.get("all_scores", [])),
            "video_model_scores": json_value(video_evidence.get("all_scores", [])),

            "processing_seconds": round(time.perf_counter() - start, 2),
        }

    finally:
        # remove temporary media created during voice and video processing
        if case["mode"] == VOICE_REFLECTION:
            cleanup_voice_result(
                reflection_state.get("voice_processing_result", {}) or {}
            )
        elif case["mode"] == VIDEO_REFLECTION:
            cleanup_video_result(
                reflection_state.get("video_processing_result", {}) or {}
            )

# keep failed cases in the final dataset instead of dropping them
def failure_row(case, error):
    return {
        "case": case["case"],
        "source_group": case["source_group"],
        "mode": case["mode"],
        "file": case["path"].name,
        "expected_emotion": case["expected_emotion"],
        "question_set": case["question_set"],
        "pipeline_pass": False,
        "full_end_to_end": False,
        "robustness_case": case["robustness_case"],
        "processing_errors": str(error),
    }


# summarise passes, formula checks and processing time by input type
def make_summary(df):
    rows = []
    for group, part in df.groupby("source_group", sort=False):
        rows.append({
            "source_group": group,
            "cases": len(part),
            "pipeline_passes": int(part["pipeline_pass"].fillna(False).sum()),
            "pipeline_pass_rate": round(
                part["pipeline_pass"].fillna(False).mean(), 4
            ),
            "full_end_to_end_cases": int(
                part["full_end_to_end"].fillna(False).sum()
            ),
            "formula_matches": int(
                (
                    part["reflection_formula_match"].fillna(False)
                    & part["combined_formula_match"].fillna(False)
                ).sum()
            ),
            "median_processing_seconds": round(
                part["processing_seconds"].median(), 2
            ),
        })

    summary = pd.DataFrame(rows)
    # add one overall row for the whole experiment
    overall = pd.DataFrame([{
        "source_group": "Overall",
        "cases": len(df),
        "pipeline_passes": int(df["pipeline_pass"].fillna(False).sum()),
        "pipeline_pass_rate": round(
            df["pipeline_pass"].fillna(False).mean(), 4
        ),
        "full_end_to_end_cases": int(
            df["full_end_to_end"].fillna(False).sum()
        ),
        "formula_matches": int(
            (
                df["reflection_formula_match"].fillna(False)
                & df["combined_formula_match"].fillna(False)
            ).sum()
        ),
        "median_processing_seconds": round(
            df["processing_seconds"].median(), 2
        ),
    }])
    return pd.concat([summary, overall], ignore_index=True)


# create the main pipeline result graphs
def save_graphs(df, summary):
    full = df[df["combined_score"].notna()].copy()
    # compare question, reflection and final combined scores
    if not full.empty:
        x = np.arange(len(full))
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width, full["question_score"], width, label="Questions")
        ax.bar(x, full["reflection_score"], width, label="Reflection")
        ax.bar(x + width, full["combined_score"], width, label="Combined")
        ax.set_xticks(x)
        ax.set_xticklabels(full["case"])
        ax.set_ylim(1, 5)
        ax.set_ylabel("Score")
        ax.set_title("End-to-End Pipeline Scores")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "pipeline_scores.png", dpi=300)
        plt.close(fig)

    groups = summary[summary["source_group"] != "Overall"]

    # pass rate by source/input type
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        groups["source_group"],
        groups["pipeline_pass_rate"] * 100,
    )
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Pipeline Pass Rate by Input Type")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(RESULTS / "pipeline_pass_rate.png", dpi=300)
    plt.close(fig)
    # median processing time by source/input type
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        groups["source_group"],
        groups["median_processing_seconds"],
    )
    ax.set_ylabel("Median processing time (s)")
    ax.set_title("Pipeline Processing Time by Input Type")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(RESULTS / "pipeline_processing_time.png", dpi=300)
    plt.close(fig)


# run every prepared case and save the full experiment evidence
def main():
    cases = build_cases()
    rows = []
    print(f"Running {len(cases)} pipeline cases...\n")
    for i, case in enumerate(cases, 1):
        print(
            f"[{i}/{len(cases)}] "
            f"{case['case']} - {case['source_group']}"
        )
        try:
            row = run_case(case)
        except Exception as error:
            # keep failed cases so the overall pass rate remains honest
            row = failure_row(case, error)
            print("  FAILED:", error)
        rows.append(row)
    df = pd.DataFrame(rows)

    # failure rows may not contain these fields, so add them before summarising
    required_columns = [
        "reflection_formula_match",
        "combined_formula_match",
        "processing_seconds",
    ]
    for column in required_columns:
        if column not in df:
            df[column] = None
    summary = make_summary(df)

    # save detailed case results and the smaller summary table
    df.to_csv(
        RESULTS / "pipeline_detailed_results.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS / "pipeline_summary.csv",
        index=False,
    )

    save_graphs(df, summary)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {RESULTS}")


# run the pipeline experiment only when this file is executed directly
if __name__ == "__main__":
    main()