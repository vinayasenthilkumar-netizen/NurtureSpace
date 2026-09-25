# Nurture Space

**A privacy-aware, non-diagnostic multimodal AI system for postpartum wellbeing support**  
University of London · CM3070 Final Project

> **Important:** Nurture Space is a research prototype developed for academic assessment. It does **not** diagnose postpartum depression, anxiety, or any other condition; determine clinical risk; recommend treatment; or replace a healthcare professional. Its scores, thresholds, weights and Personal Pattern rules are project-defined and technically evaluated, not clinically validated.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Selected AI Models](#selected-ai-models)
- [Deterministic Scoring](#deterministic-scoring)
- [Personal Pattern Analysis](#personal-pattern-analysis)
- [Bounded Agentic AI Assistant](#bounded-agentic-ai-assistant)
- [Privacy and Data Handling](#privacy-and-data-handling)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Running the Test Suite](#running-the-test-suite)
- [Evaluation Summary](#evaluation-summary)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Project Scope](#project-scope)
- [Academic Project](#academic-project)

---

## Overview

Nurture Space orchestrates several independently pre-trained AI models across text, speech, vocal-pattern and visible-expression modalities, then combines their outputs with deterministic application logic, human review, longitudinal pattern analysis, curated resource retrieval and a bounded local AI Assistant.

Two related check-in pathways are supported:

- **Full Check-In** — a 10-question questionnaire followed by compulsory text, voice or video reflection. It is required on Mondays as the weekly snapshot and is also available on other days when the questionnaire is selected.
- **Personal Reflection Check-In** — a reflection-only pathway available on non-Mondays when the questionnaire is not selected.

Each modality is processed separately and transformed through deterministic scoring and fusion rules. Model-derived contextual observations are presented for user review before eligible information is persisted. User-facing results are non-clinical descriptive Indicators. A separate bounded Assistant can hold non-diagnostic conversation, retrieve curated resources, and propose edits to the current Personal Summary or Appointment Discussion Points, while application logic—not the language model—controls whether Update or Undo is permitted.

---

## Key Features

- Registered-user authentication with server-side Flask sessions.
- Independent privacy controls for saving approved check-ins, using saved history for personalisation, and saving bookmarked resources.
- Four scoring questions (Sleep, Mood, Stress, Perceived Support) and six contextual questions.
- Text, voice and video reflections, with a 60-second limit on recorded media and an editable Whisper transcript before analysis for voice/video.
- Separate text-emotion, vocal-pattern and visible-expression inference paths, with conservative missing-evidence handling.
- Deterministic Questionnaire, Reflection and Combined Indicators.
- Personal Pattern Analysis (baseline change, recurring themes, emerging themes, variability and reflection–check-in alignment) with Personal Pattern Relevance (PPR) prioritisation.
- Curated resource retrieval via BGE embeddings + FAISS.
- Bounded local AI Assistant (Qwen3 4B via Ollama) with protected narrative Update and one-level Undo.
- History, trend and calendar views for eligible saved entries.
- No persistent storage of raw audio/video, sampled frames, face crops or generated transcripts; these are processed temporarily only.

---

## System Architecture

The authenticated Flask application routes into two LangGraphs: a **Guided Wellbeing Check-In** graph (Structured Check-In → Reflection → Review → Dashboard) and a **Bounded Agentic AI Assistant** graph (safety check → intent routing → conversation / retrieval / Personal Summary / Appointment Points). Protected application logic gates Update/Undo operations before permitted changes can reach persistent state.

The full system architecture is shown in **Figure 3.1 of the accompanying final report**.

Main graph modules:

- `graphs/guided_workflow_graph.py`
- `graphs/guided_checkin_graph.py`
- `graphs/reflection_graph.py`
- `graphs/review_graph.py`
- `graphs/dashboard_graph.py`
- `graphs/assistant_graph.py`

---

## Selected AI Models

| Component | Final model | Role |
|---|---|---|
| Text emotion | `joeddav/distilbert-base-uncased-go-emotions-student` | Maps reflection text into application emotion categories |
| Speech-to-text | Whisper `small.en` | Produces editable transcripts for voice/video reflections |
| Vocal pattern | `Khoa/w2v-speech-emotion-recognition` | Provides tentative vocal-pattern evidence from audio |
| Visible expression | HSEmotion `enet_b0_8_va_mtl` | Analyses sampled face crops from video |
| Resource embeddings | `BAAI/bge-small-en-v1.5` | Embeds resources and queries for retrieval |
| Local generative model | `qwen3:4b-instruct` via Ollama | Bounded conversation, retrieval-grounded responses, and narrative drafting/editing |

The video pipeline samples **5 representative frames** per recording. Visible-expression evidence is only accepted when at least **3 usable face predictions** are available and one mapped label reaches a strict majority; otherwise it is treated as unavailable rather than neutral.

---

## Deterministic Scoring

**Questionnaire Score** (Stress is reversed so all four scoring domains point in the same direction):

```text
Reversed Stress = 6 - Stress
Questionnaire Score = (Sleep + Mood + Reversed Stress + Support) / 4
```

**Reflection Score** — available modalities are fused with fixed fallback weights:

| Available evidence | Fusion |
|---|---:|
| Text only | 100% Text |
| Audio only | 100% Audio |
| Text + Audio | 50% / 50% |
| Text + Visual | 80% / 20% |
| Audio + Visual | 80% / 20% |
| Text + Audio + Visual | 40% / 40% / 20% |
| Visual only | No Reflection Score |

**Combined Score** (for a Full Check-In when both Reflection and Questionnaire Scores are available):

```text
Combined Score = (0.70 × Reflection Score) + (0.30 × Questionnaire Score)
```

Thresholds of **3.67** and **2.34** map scores into **Relatively steady**, **Some strain** and **Significant strain** bands. These are descriptive application-defined bands, not clinical scores or diagnoses.

---

## Personal Pattern Analysis

Eligible confirmed saved history is analysed using five deterministic signals: baseline change, recurring themes, emerging themes, wellbeing variability, and reflection–check-in alignment. PPR prioritises which eligible patterns are surfaced.

If saved-history personalisation is enabled, relevant read-only pattern context may also support Assistant conversation or a user-initiated resource query. The Assistant cannot create, alter or reprioritise PPA/PPR rules or results.

---

## Bounded Agentic AI Assistant

**Can:** hold non-diagnostic conversation; retrieve curated resources on request; generate or edit the Personal Summary and Appointment Discussion Points; propose supported rewrites, shortening, additions or removals.

**Cannot:** diagnose a condition; change questionnaire answers or approved model observations; calculate or overwrite scores or Indicators; alter PPA/PPR rules or results; write arbitrary values to the database; or treat model outputs as a user's true emotional state.

Update/Undo operations are checked by application logic and limited to the **latest eligible check-in on the current calendar day**. An explicit-language safety check runs before normal routing; immediate-harm wording is handled through a fixed urgent-support path. This is an application safety boundary, not a clinical risk assessment.

---

## Privacy and Data Handling

The application has three independent user-controlled privacy settings:

1. **Save Approved Check-Ins**
2. **Use Saved History for Personalisation**
3. **Save Bookmarked Resources**

Persistent records are user-scoped.

**Not retained as persistent content:** raw audio/video, extracted audio, sampled frames, face crops, generated transcripts and raw intermediate model outputs. Temporary media is processed under `data/temp/`, while runtime database/session content is excluded from version control.

---

## Technology Stack

- **App/orchestration:** Python 3.11.9, Flask 3.1.3, Jinja2, Flask-Session, LangGraph 1.2.10, SQLite
- **ML/retrieval:** PyTorch, Hugging Face Transformers, Whisper, Sentence Transformers, FAISS, HSEmotion/HSEmotion-ONNX, MediaPipe, OpenCV, librosa
- **Testing/evaluation:** Pytest, scikit-learn, pandas, NumPy, Matplotlib

---

## Prerequisites

Nurture Space was developed in **Visual Studio Code** using a Python virtual environment (`.venv`) with **Python 3.11.9**. Python 3.11 is recommended; using 3.11.9 most closely reproduces the original development environment.

Before installing, make sure you have:

1. **Python 3.11** (3.11.9 used during development)
2. **Git**
3. **FFmpeg** — required for voice/video media preparation; development used **FFmpeg 8.1.1 essentials build (gyan.dev)**
4. **Ollama** — required for the local Qwen3 Assistant; development used **Ollama 0.34.4**
5. A modern browser; microphone/camera permission is required only for in-browser recording

After installation, these commands should work:

```bash
python --version
ffmpeg -version
ollama --version
```

The original development environment was verified with **Python 3.11.9**, **FFmpeg 8.1.1-essentials_build-www.gyan.dev**, and **Ollama 0.34.4**. Exact patch versions are not required unless you want to reproduce the original setup as closely as possible.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create and activate the virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Confirm the interpreter:

```bash
python --version
```

The original development environment used `Python 3.11.9`.

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install FFmpeg

Nurture Space uses FFmpeg during voice/video media preparation. FFmpeg is a **system dependency**, so installing the Python requirements alone is not enough.

#### Windows

1. Open the official FFmpeg download page: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
2. Under **Get packages & executable files → Windows EXE Files**, choose one of the Windows build providers listed by FFmpeg. The **gyan.dev** build is suitable for this project.
3. From the gyan.dev builds page, download **`ffmpeg-release-essentials.zip`**.
4. Extract the ZIP to a permanent location, for example `C:\ffmpeg`.
5. Locate the extracted `bin` folder containing `ffmpeg.exe`.
6. Add that `bin` folder to your Windows **Path** environment variable:
   - Search Windows for **Edit the system environment variables**.
   - Open **Environment Variables**.
   - Under your user variables, select **Path → Edit → New**.
   - Add the full path to the extracted FFmpeg `bin` folder.
7. Close and reopen PowerShell/VS Code, then verify:

```powershell
ffmpeg -version
```

If a version is printed, FFmpeg is available to the application. The original Windows development environment used `ffmpeg version 8.1.1-essentials_build-www.gyan.dev`.

#### macOS

With Homebrew installed:

```bash
brew install ffmpeg
```

#### Ubuntu / Debian

```bash
sudo apt update
sudo apt install ffmpeg
```

Then verify with:

```bash
ffmpeg -version
```

### 5. Install Ollama and Qwen3

Nurture Space expects Ollama's local chat API at `http://localhost:11434/api/chat` and uses the model tag `qwen3:4b-instruct`.

#### Windows

1. Open the official Ollama download page: [https://ollama.com/download/windows](https://ollama.com/download/windows).
2. Select **Download for Windows** and run `OllamaSetup.exe`.
3. Alternatively, Ollama currently provides this PowerShell installer command:

```powershell
irm https://ollama.com/install.ps1 | iex
```

4. Open a new PowerShell or VS Code terminal and verify:

```powershell
ollama --version
```

The original development environment used `ollama version is 0.34.4`.

5. Download the exact Qwen model used by Nurture Space:

```powershell
ollama pull qwen3:4b-instruct
```

6. Confirm it is installed:

```powershell
ollama list
```

Ollama normally runs its local API in the background on Windows. If the application reports that it cannot connect to Ollama, start the service manually with:

```powershell
ollama serve
```

#### macOS / Linux

Download/install Ollama from [https://ollama.com/download](https://ollama.com/download), then run:

```bash
ollama pull qwen3:4b-instruct
ollama list
```

### 6. Flask development configuration

No additional Flask secret-key command is required for the current local academic prototype because the application configuration already supplies a development `SECRET_KEY`.

> **Deployment note:** the development key is intentionally suitable only for local development/testing. A real public deployment should replace it with a strong secret stored outside the source code.

### Note on the curated resource catalogue

The SQLite schema is created automatically on startup, but the curated resource catalogue is application data rather than schema data. The development database is not committed to GitHub (`data/database/*.db` is git-ignored because it can also contain account, check-in and reflection records).

A fresh clone therefore needs a **sanitised resources-only seed/import source** before Daily Resources and RAG retrieval can use the full curated catalogue. Do not publish the development database simply to provide those resources, because it may also contain user/test records.

---

## Running the Application

With the virtual environment active, FFmpeg available and Ollama running:

```bash
python run_flask.py
```

The launcher creates the Flask app, initialises the SQLite schema, warms the BGE retriever and local Qwen3 model, starts the server and opens the application in the default browser at:

```text
http://127.0.0.1:5000
```

The first run may be slower because Hugging Face, Whisper and embedding model files must be downloaded and cached locally. Later runs are typically faster.

---

## Running the Test Suite

Run the complete submitted test suite with:

```bash
python -m pytest tests -q
```

The final archive contains **73 unit tests** and **6 integration tests** (79 total).

To run one level only:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```

Some integration tests require the local Ollama service and/or downloaded model files.

---

## Evaluation Summary

Full methodology and results are reported in Chapter 5 and Appendices A–C of the accompanying final report.

| Area | Result |
|---|---|
| Text emotion (postpartum-style model-selection set) | 0.8333 Top-1, 0.8228 Macro F1 |
| Whisper `small.en` (reserved-speaker test) | WER 0.0359, CER 0.0200 |
| Vocal pattern (held-out RAVDESS Actors 05–06) | 0.5909 accuracy, 0.5738 Macro F1 |
| Visible expression (deployment-aligned 5-frame held-out test) | 0.5333 accuracy, 0.5037 Macro F1 |
| BGE + FAISS retrieval (15 held-out queries) | Hit@1 1.0, Hit@3 1.0, MRR 1.0 |
| Bounded Assistant (final held-out cases) | 4/6 ordinary tasks, 6/6 boundary checks, 2/2 safety cases |
| End-to-end multimodal pipeline | 16/16 cases passed, 16/16 score recomputations matched |
| Usability (SUS) | Mean 91.17/100, N=15 |

Evaluation/model-selection scripts and their external datasets are **not required to run the application**. Reproducing those experiments may additionally require datasets such as RAVDESS and LibriSpeech, plus the project-specific evaluation materials described in the report.

---

## Repository Structure

```text
Nurture-Space/
├── config/          # application/model settings
├── core/            # shared constants and labels
├── data/            # runtime database and temporary media
├── database/        # schema, accounts, privacy, check-ins and resources
├── flask_app/       # routes, templates, static assets and app factory
├── graphs/          # LangGraph workflow and Assistant graphs
├── model_adapters/  # text, Whisper, audio and visible-expression adapters
├── modules/         # scoring and media-processing utilities
├── patterns/        # Personal Pattern Analysis and PPR logic
├── rag/             # embeddings, FAISS retrieval and personalisation
├── services/        # authentication, processing, Assistant and saving services
├── tests/           # unit and integration tests
├── run_flask.py     # application entry point
├── requirements.txt
└── README.md
```

Evaluation scripts/results may be kept in separate experiment folders; they are not part of the normal application runtime path.

---

## Troubleshooting

**`ollama` is not recognised** — close and reopen PowerShell/VS Code after installing Ollama. Verify with `ollama --version`.

**Assistant reports a connection error / connection refused** — confirm Ollama is running and that `qwen3:4b-instruct` appears in `ollama list`. If necessary, run `ollama serve`.

**FFmpeg is not recognised** — confirm the extracted FFmpeg `bin` directory was added to Windows `Path`, then open a new terminal and run `ffmpeg -version`.

**Voice/video reflection fails during conversion** — first confirm `ffmpeg -version` works from the same terminal environment used to launch Nurture Space.

**First run is slow** — expected while pretrained model files are downloaded, loaded and cached. Subsequent runs are normally faster.

**Port 5000 is already in use** — stop the process using that port or change the local port in `run_flask.py`.

**Daily Resources / retrieval returns nothing on a fresh clone** — the resource catalogue is application data rather than schema data. Populate it from a sanitised resources-only source before testing the resource/RAG features.

---

## Known Limitations

- Nurture Space is a technical research prototype, not a clinical tool; the selected checkpoints were not trained specifically for postpartum wellbeing in Singapore.
- Vocal and visible-expression evaluation used acted RAVDESS data, limiting real-world generalisation.
- Visible expression is the weakest evaluated modality and is intentionally constrained in the fusion rules.
- The 70:30 Reflection/Questionnaire weighting and score thresholds are engineering choices, not clinically validated parameters.
- The local LLM can still make semantic errors even when lexical grounding checks pass; protected Update/Undo and human confirmation therefore remain necessary.
- Usability testing measured general adult usability on a small, non-postpartum-restricted sample rather than target-population acceptability or clinical effectiveness.
- Local CPU-only generation can introduce noticeable response latency.

---

## Project Scope

Nurture Space is intended for **English-speaking postpartum mothers in Singapore during the first year after childbirth**. It supports private self-reflection, non-clinical wellbeing summaries, longitudinal self-observation, curated informational resources and preparation of discussion points for healthcare appointments.

Diagnosing postpartum depression or anxiety, determining clinical risk, prescribing or recommending treatment, and replacing professional care are explicitly outside the project scope.

---

## Academic Project

Developed for the **University of London CM3070 Final Project**, applying the project theme of orchestrating multiple pre-trained models to achieve a goal. Full citations for the literature, model sources and evaluation datasets are provided in the accompanying final report.

Main pretrained/model components include `joeddav/distilbert-base-uncased-go-emotions-student`, Whisper `small.en`, `Khoa/w2v-speech-emotion-recognition`, HSEmotion `enet_b0_8_va_mtl`, `BAAI/bge-small-en-v1.5`, FAISS and `Qwen3` through Ollama.



