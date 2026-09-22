
import json
import time
from typing import Any

import requests

from config.settings import SELECTED_LLM_MODEL

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# Keep the model loaded in Ollama memory.
# This prevents repeated model loading between requests.
OLLAMA_KEEP_ALIVE = -1
# Maximum time allowed for one Ollama request.
REQUEST_TIMEOUT_SECONDS = 120
# Use one persistent HTTP session.
# This avoids repeatedly creating HTTP connections.
HTTP_SESSION = requests.Session()
# Smaller context is enough for the relatively short
# conversations used by this prototype.
OLLAMA_CONTEXT_SIZE = 4096


# assistant tasks
VALID_TASKS = {
    "conversation",
    "summary",
    "appointment_points",
    "dashboard_update",
    "resources",
}


# the main system prompts
SYSTEM_PROMPT = """
You are a private, non-diagnostic postpartum wellbeing assistant.

Your role is to help an adult user reflect on their experience,
organise approved information and prepare possible discussion points
for an appointment.

Follow these rules:

1. Do not diagnose postpartum depression, anxiety or another condition.

2. Do not claim that text, voice or facial-expression models reveal
   the user's true emotions.

3. Question, Reflection and Combined wellbeing results are calculated
   by the application, not by the language model.

4. Never calculate, replace, update or change any wellbeing score,
   status, result or indicator.

5. Keep questionnaire, text, audio and visual observations distinguishable.
   Do not invent or recalculate multimodal results.

6. Only use information contained in the supplied application state
   and user message.

7. Proposed themes, appointment points and dashboard changes must be
   confirmed by the user before they are applied.

8. Do not claim that information has been saved. Database saving is
   controlled separately by the application.

9. Suggest only general resource categories. Do not invent organisation
   names, phone numbers, URLs or emergency contact details.

10. When the user describes an immediate risk of harm to themselves or
    their baby, stop the normal reflection flow and encourage immediate
    contact with emergency services and a trusted person.

11. Follow the requested task. Do not produce outputs belonging to a
    different task.

12. Check the application state's "context_source".

    If context_source is "current_checkin":
    - this is the user's active current check-in;
    - use wording such as "in this check-in" or
      "today's check-in" where appropriate;
    - do not call it a saved check-in unless the application
      explicitly says it has already been saved.

    If context_source is "current_reflection":
    - this is the user's active current reflection;
    - use wording such as "in this reflection";
    - do not call it a saved check-in;
    - do not imply that structured questions were completed;
    - do not invent Today's Check-In Status.

    If context_source is "latest_saved_checkin":
    - treat the information as the user's most recent saved
      approved check-in;
    - do not describe it as "today", "today's check-in" or
      the user's current check-in;
    - use wording such as "your most recent saved check-in"
      or "in your latest saved check-in".


    If context_source is "latest_saved_reflection":
    - treat the information as the user's most recent saved reflection;
    - do not imply that structured questions were completed;
    - do not describe it as a newly completed current reflection;
    - use wording such as "your most recent saved reflection" where appropriate."""


#the task instrcutions
TASK_INSTRUCTIONS = {
    "conversation": """
Give a brief, supportive conversational response.

Requirements:
- Respond naturally to what the user asked.
- Keep the response concise.
- Do not create a summary unless requested separately.
- Do not create appointment points unless requested separately.
- Do not introduce information that is not present in the supplied state,
  user message or retrieved curated resources.

When retrieved resources are supplied:

- The interface displays each retrieved resource separately as a card.
- Do not list or repeat the individual resources in assistant_reply.
- Do not mention resource titles, organisations, sources or URLs.
- Do not write Markdown links.
- Do not write bullet points describing the resources.
- Do not summarise each resource individually.
- Write only one short introductory paragraph explaining why the
  retrieved resources may be useful or relevant.
- Keep the introduction grounded in the user's request or approved
  check-in information.
- Do not claim that a resource is prescribed or clinically recommended.
- Do not turn general resource information into personalised medical advice.
""",

"summary": """
Create or edit a personalised check-in summary.

FOLLOW-UP EDITING RULES:

When an earlier "Summary draft:" exists, treat it as the draft being edited.
The user may add, remove, rewrite, reword, refine, simplify or shorten it.

ADD:
- Preserve the existing draft as closely as possible.
- Keep all existing sentences and supported information unless the new
  information requires a small wording change.
- Add only the information the user explicitly asks to add.
- Do not rewrite unrelated parts of the draft.
- Do not remove existing supported information unless requested.
- If the requested information is already represented in the existing
  draft using the same or equivalent meaning, do not add it again and
  do not rewrite the draft unnecessarily.
- For example, "struggling with eating" and "difficulty eating" represent
  the same concern and should not appear twice.
- If the user names only a topic, such as "add stress", add only a simple
  factual statement about that topic.
- Example: "add stress" may become "You have also been feeling stressed."
- Do not add intentions, goals, coping statements, consequences or advice
  such as "you would like to explore it", "you are trying to manage it",
  or "it has been affecting you" unless the user explicitly said this.

REMOVE:
- Remove only the information the user explicitly asks to remove.
- Preserve the rest of the supported summary.
- Do not remove or alter the underlying approved check-in information.
  You are editing only the Personal Summary wording.

REWRITE / REWORD / REFINE:
- Preserve the supported meaning while changing the wording requested.
- Do not introduce new concerns or relationships.

SHORTEN / SIMPLIFY:
- Genuinely make the existing draft shorter or simpler.
- Preserve the main supported concerns unless the user explicitly asks
  for one to be removed.

Requirements:
- Write 2 to 3 natural sentences unless the user asks for a shorter version.
- Write directly to the user using "you".
- Use only information explicitly present in the supplied application state
  or explicitly stated by the user.
- Prioritise approved or confirmed information.
- Mention supportive factors only when explicitly supplied.
- Keep the wording warm, neutral and descriptive.

STRICT GROUNDING RULES:
- A summary statement must be supported by either approved application-state
  information or information explicitly stated by the user.
- Newly stated chat information may be used in the draft when the user asks
  for it to be added, but do not describe it as saved or approved unless it
  appears in the application state.
- An earlier Assistant statement is not evidence by itself.
- Stay close to the user's or application's wording.
- Do not infer causes, effects, relationships, symptoms or experiences.
- Do not add advice, recommendations or coping suggestions.
- Do not say something is "affecting", "impacting", "taking a toll",
  "making things harder" or similar unless that relationship was explicitly
  supplied by the user or application state.
- Do not diagnose or imply a mental-health condition.
- Do not use technical model terminology.
- Never change or recalculate any wellbeing score, result or indicator.
- Do not use the application's calculated Wellbeing Status, Question result,
  Reflection result or Combined result as descriptive summary content.
- Avoid causal connectors such as "because", "therefore", "which has made",
  "these strains have made", "leading to" or "as a result" unless the user
  explicitly supplied that relationship.

Return the result in summary_draft.
""",

"appointment_points": """
Create or edit possible appointment discussion points.

NEW DRAFT:
- Generate 1 to 3 distinct supported negative concerns. Include each clearly different negative concern when present.
- Prefer fewer points when concerns overlap.
- Combine closely related concerns into one point.
- Do not repeat the same concern using different wording.

FOLLOW-UP EDITING:
When earlier "Appointment points:" exist, treat them as the draft being edited.
The user may add, remove, rewrite, reword, simplify or shorten them.

ADD:
- Preserve all existing supported points.
- Add the new concern as an additional point unless it duplicates an existing point.
- If the user gives only a topic or simple concern, keep the new point equally simple.
- Do not invent effects, consequences, causes, coping difficulties or extra context.
- Do not expand a short request into a more detailed concern.
- Use only what the user explicitly stated.

Examples:
- "add tiredness" -> "I feel tired, and I would like to discuss this."
- "add feeling overwhelmed" -> "I feel overwhelmed, and I would like to discuss this."

REMOVE:
- Remove only the point or concern the user asks to remove.
- Preserve the remaining supported points.

REWRITE / REWORD:
- Preserve the supported concern while changing only the wording requested.

SHORTEN / SIMPLIFY:
- Make the point clearly shorter while preserving its main meaning.

WRITING STYLE:
- Each point must be one short sentence.
- Keep each point to a maximum of about 18 words.
- State only the main concern.
- Do not add background detail that is already clear from the reflection.
- Every point must end exactly with:
  "and I would like to discuss this."
- Do not use alternatives such as:
  "I would like to discuss how I have been managing this."
- Use simple first-person wording.

STRICT GROUNDING RULES:
- Every point must describe a concern explicitly supported by the
  application state or explicitly stated by the user.
- Stay close to the user's wording.
- Do not invent symptoms, consequences or medical problems.
- Do not diagnose.
- Do not create a point solely from emotion-model, vocal-model or
  visible-expression observations.
- Do not use a calculated wellbeing result as an appointment concern.
- Do not infer cause and effect unless explicitly stated.
- When the user supplies only a short topic, do not infer how it affects
  their routine, relationships, sleep, baby care or other areas.
  
SUPPORTIVE INFORMATION:
- Never turn positive or supportive information into a problem.
- Supportive information should not become its own appointment point.

APPOINTMENT POINT SELECTION:
- Use only explicit negative concerns, difficulties, strain or discomfort from confirmed_reflection or context_appointment_points. If the information is only positive or supportive, return no appointment points.
- Combine duplicates and closely related concerns.
- Sleep difficulty and tiredness may be combined when they clearly refer
  to the same issue.
- Feeling overwhelmed and having too much to manage may be combined.
- Do not create several points from different sentences describing the
  same underlying concern.
- If no suitable supported concern exists, return [].

GOOD EXAMPLES:
- "I feel exhausted and overwhelmed, and I would like to discuss this."
- "I have been struggling with sleep, and I would like to discuss this."

BAD EXAMPLES:
- "I have been feeling very exhausted and overwhelmed even though I am
  trying to take care of the baby and my daily needs, and I would like
  to discuss this."
- "I haven't slept in a while and feel very tired, and I would like to
  discuss how I have been managing this."
- Multiple points that repeat tiredness, exhaustion and sleep difficulty
  without adding a different concern.

Return the revised list in appointment_points.
""",

    "dashboard_update": """
Suggest dashboard changes only when they are directly supported by the
supplied information.

Requirements:
- Do not apply changes yourself.
- Do not invent new wellbeing information.
- Any proposed update must require user confirmation.
""",

    "resources": """
Suggest only broad categories of supportive resources relevant to the
supplied information.

Requirements:
- Do not invent organisations, URLs, phone numbers or contact details.
- Do not provide a diagnosis.
- Return only appropriate general resource categories.
"""
}


# task response based on the specific taks
TASK_RESPONSE_SCHEMAS = {

    "summary": {
        "type": "object",
        "properties": {
            "summary_draft": {
                "type": "string",
            }
        },
        "required": [
            "summary_draft",
        ],
        "additionalProperties": False,
    },

    "appointment_points": {
        "type": "object",
        "properties": {
            "appointment_points": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "maxItems": 10,
            }
        },
        "required": [
            "appointment_points",
        ],
        "additionalProperties": False,
    },

    "dashboard_update": {
        "type": "object",
        "properties": {
            "proposed_dashboard_updates": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            }
        },
        "required": [
            "proposed_dashboard_updates",
        ],
        "additionalProperties": False,
    },

    "resources": {
        "type": "object",
        "properties": {
            "resource_categories": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            }
        },
        "required": [
            "resource_categories",
        ],
        "additionalProperties": False,
    },
}


# token
TASK_TOKEN_LIMITS = {
    "conversation": 90,
    "summary": 120,
    "appointment_points": 220,
    "dashboard_update": 60,
    "resources": 50,
}


# safety classification
SAFETY_CATEGORIES = {
    "none",
    "self",
    "baby",
    "other_person",
}


SAFETY_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "urgent_safety_detected": {
            "type": "boolean",
        },
        "safety_category": {
            "type": "string",
            "enum": [
                "none",
                "self",
                "baby",
                "other_person",
            ],
        },
    },
    "required": [
        "urgent_safety_detected",
        "safety_category",
    ],
    "additionalProperties": False,
}

# safety trigger
SAFETY_TRIGGER_TERMS = (
    "hurt myself",
    "harm myself",
    "kill myself",
    "suicid",
    "end my life",
    "want to die",
    "wish i was dead",
    "better off dead",
    "don't want to live",
    "do not want to live",
    "don't want to be alive",
    "do not want to be alive",
    "can't keep myself safe",
    "cannot keep myself safe",

    "hurt my baby",
    "harm my baby",
    "kill my baby",
    "shake my baby",
    "hit my baby",
    "can't keep my baby safe",
    "cannot keep my baby safe",

    "hurt someone",
    "harm someone",
    "kill someone",
    "attack someone",
    "i might hurt",
    "i might harm",
)


# utility function
def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _seconds(value: Any) -> float:
    return round(
        (value or 0) / 1_000_000_000,
        2,
    )


#model warm up
def warm_llm() -> bool:
    
    #loads the selected model into Ollama before the user startsinteracting with the Assistant.call 1 time when app starts
    try:
        started = time.perf_counter()
        response = HTTP_SESSION.post(
            OLLAMA_CHAT_URL,
            json={
                "model": SELECTED_LLM_MODEL,

                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "Reply with only: ready",
                    },
                ],

                "stream": False,

                "keep_alive": OLLAMA_KEEP_ALIVE,

                "options": {
                    "temperature": 0.0,
                    "num_ctx": OLLAMA_CONTEXT_SIZE,
                    "num_predict": 4,
                },
            },

            timeout=300,
        )

        response.raise_for_status()

        elapsed = round(
            time.perf_counter() - started,
            2,
        )

        print(
            f"LLM ready: {SELECTED_LLM_MODEL} "
            f"| warm-up={elapsed}s"
        )

        return True

    except requests.RequestException as error:

        print(
            f"LLM warm-up failed: {error}"
        )

        return False



# fall basck respeonse

def _empty_response(
    error_message: str,
) -> dict[str, Any]:

    return {
        "assistant_reply": error_message,
        "summary_draft": "",
        "appointment_points": [],
        "proposed_dashboard_updates": [],
        "resource_categories": [],
        "needs_user_confirmation": False,
        "urgent_safety_response": False,
    }


#deterministic output filterin g
def _apply_task_rules(
    result: dict[str, Any],
    task: str,
) -> dict[str, Any]:

    safe_result = {

        "assistant_reply": (
            result.get(
                "assistant_reply",
                "",
            )
            if isinstance(
                result.get("assistant_reply"),
                str,
            )
            else ""
        ),

        "summary_draft": (
            result.get(
                "summary_draft",
                "",
            ).strip()
            if isinstance(
                result.get("summary_draft"),
                str,
            )
            else ""
        ),

        "appointment_points": [
            point.strip()
            for point in result.get(
                "appointment_points",
                [],
            )
            if (
                isinstance(point, str)
                and point.strip()
            )
        ][:10]
        if isinstance(
            result.get("appointment_points"),
            list,
        )
        else [],

        "proposed_dashboard_updates": (
            result.get(
                "proposed_dashboard_updates",
                [],
            )
            if isinstance(
                result.get(
                    "proposed_dashboard_updates"
                ),
                list,
            )
            else []
        ),

        "resource_categories": (
            result.get(
                "resource_categories",
                [],
            )
            if isinstance(
                result.get(
                    "resource_categories"
                ),
                list,
            )
            else []
        ),

        "needs_user_confirmation": bool(
            result.get(
                "needs_user_confirmation",
                False,
            )
        ),

        "urgent_safety_response": bool(
            result.get(
                "urgent_safety_response",
                False,
            )
        ),
    }


    #conversation
    if task == "conversation":

        safe_result["summary_draft"] = ""
        safe_result["appointment_points"] = []
        safe_result["proposed_dashboard_updates"] = []
        safe_result["resource_categories"] = []
        safe_result["needs_user_confirmation"] = False

#summary
    elif task == "summary":

        safe_result["assistant_reply"] = ""
        safe_result["appointment_points"] = []
        safe_result["proposed_dashboard_updates"] = []
        safe_result["resource_categories"] = []
        safe_result["needs_user_confirmation"] = False


    #appointment points

    elif task == "appointment_points":

        safe_result["assistant_reply"] = ""
        safe_result["summary_draft"] = ""
        safe_result["proposed_dashboard_updates"] = []
        safe_result["resource_categories"] = []
        safe_result["needs_user_confirmation"] = False


    #dashboard
    elif task == "dashboard_update":

        safe_result["assistant_reply"] = ""
        safe_result["summary_draft"] = ""
        safe_result["appointment_points"] = []
        safe_result["resource_categories"] = []

        safe_result["needs_user_confirmation"] = bool(
            safe_result[
                "proposed_dashboard_updates"
            ]
        )


    # resources
    elif task == "resources":
        safe_result["assistant_reply"] = ""
        safe_result["summary_draft"] = ""
        safe_result["appointment_points"] = []
        safe_result["proposed_dashboard_updates"] = []
        safe_result["needs_user_confirmation"] = False


    #urgent safety response
    if safe_result["urgent_safety_response"]:
        safe_result["summary_draft"] = ""
        safe_result["appointment_points"] = []
        safe_result["proposed_dashboard_updates"] = []
        safe_result["resource_categories"] = []
        safe_result["needs_user_confirmation"] = False

    return safe_result


#prepare the recent convo 
def _prepare_history(
    chat_history,
    limit: int = 4,
) -> list[dict[str, str]]:

    history = (
        chat_history
        if isinstance(chat_history, list)
        else []
    )

    prepared = []

    for message in history[-limit:]:

        if not isinstance(message, dict):
            continue

        role = message.get("role")

        if role not in {
            "user",
            "assistant",
        }:
            continue

        content = str(
            message.get(
                "content",
                "",
            )
            or ""
        ).strip()

        if not content:
            continue

        prepared.append(
            {
                "role": role,
                # Keep history short.
                "content": content[:700],
            }
        )

    return prepared


# prepare the retirved resources
def _prepare_resources(
    retrieved_resources,
) -> list[dict[str, str]]:
    if not retrieved_resources:
        return []

    prepared = []

    for resource in retrieved_resources[:3]:
        content = str(resource.get("content", "") or "").strip()
        topics = resource.get("topics", [])

        if isinstance(topics, list):
            topic_text = ", ".join(
                str(item).strip()
                for item in topics[:4]
                if str(item).strip()
            )
        else:
            topic_text = ""

        prepared.append({
            "topics": topic_text[:120],
            "excerpt": content[:180],
        })

    return prepared


def _prepare_application_state(
    application_state: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    state = (
        application_state
        if isinstance(application_state, dict)
        else {}
    )

    if task == "appointment_points":
        keys = (
            "context_source",
            "confirmed_reflection",
            "context_appointment_points",
            "approved_strain_factors",
            "confirmed_appointment_points",
        )

    elif task == "summary":
        keys = (
            "context_source",
            "confirmed_reflection",
            "context_summary_points",
            "approved_text_emotions",
            "approved_supportive_factors",
            "approved_strain_factors",
            "approved_vocal_observation",
            "approved_visible_observation",
            "confirmed_summary",
        )

    else:
        keys = (
            "context_source",
            "saved_entry_type",
            "saved_entry_created_at",
            "status",
            "core_responses",
            "context_responses",
            "approved_text_emotions",
            "approved_supportive_factors",
            "approved_strain_factors",
            "approved_vocal_observation",
            "approved_visible_observation",
            "confirmed_summary",
            "confirmed_appointment_points",
            "personal_pattern_relevance",
        )

    return {
        key: state[key]
        for key in keys
        if key in state
        and state[key] not in (None, "", [], {})
    }

# build compact user prompt
def _build_user_prompt(
    application_state: dict[str, Any],
    user_message: str,
    task: str,
    retrieved_resources=None,
    chat_history=None,
) -> str:
    state = _prepare_application_state(
        application_state,
        task,
    )
    history = _prepare_history(chat_history, limit=6)
    resources = _prepare_resources(retrieved_resources)
    task_instruction = TASK_INSTRUCTIONS.get(task, "")

    parts = [
        f"Requested task: {task}",
        "Task-specific instructions:\n" + task_instruction.strip(),
        "Application state:\n" + compact_json(state),
    ]

    if history:
        parts.append("Recent conversation:\n" + compact_json(history))

    parts.append("Current user message:\n" + user_message)

    if resources:
        parts.append(
            "Retrieved curated resources:\n"
            + compact_json(resources)
            + "\nThe interface shows these resources as cards. "
              "Do not list, name, link or individually summarise them."
        )

    parts.append(
        "The application state is authoritative for approved check-in "
        "information. Do not treat an instruction such as 'create my "
        "summary' as new wellbeing information. Do not claim chat "
        "information has been saved unless it appears in application state."
    )

    if task == "conversation":
        parts.append(
            "Return only the conversational response text. Do not return JSON."
        )
    else:
        parts.append(
            "Return only valid JSON matching the requested task schema."
        )

    return "\n\n".join(parts)


#safety gate 

def _needs_safety_classifier(
    user_message: str,
    chat_history=None,
) -> bool:

    text = str(
        user_message or ""
    ).strip().lower()

    # Current message gets priority.
    if any(
        term in text
        for term in SAFETY_TRIGGER_TERMS
    ):
        return True


    # Only inspect history for direct follow-up answers.
    follow_up_terms = {
        "yes",
        "yes i do",
        "i do",
        "i might",
        "maybe",
        "right now",
        "still",
        "i still do",
    }

    if text not in follow_up_terms:
        return False


    history = (
        chat_history
        if isinstance(chat_history, list)
        else []
    )

    for message in reversed(
        history[-4:]
    ):

        if not isinstance(
            message,
            dict,
        ):
            continue

        if message.get("role") != "user":
            continue

        previous = str(
            message.get(
                "content",
                "",
            )
            or ""
        ).lower()

        return any(
            term in previous
            for term in SAFETY_TRIGGER_TERMS
        )

    return False


# safety clasifeir

def classify_urgent_safety(
    user_message,
    chat_history=None,
):

    clean_message = str(
        user_message or ""
    ).strip()

    if not clean_message:

        return {
            "urgent_safety_detected": False,
            "safety_category": "none",
        }


    # Performance optimisation:
    # ordinary messages do not need a second LLM request.
    if not _needs_safety_classifier(
        clean_message,
        chat_history,
    ):

        return {
            "urgent_safety_detected": False,
            "safety_category": "none",
        }


    recent_history = _prepare_history(
        chat_history,
        limit=3,
    )


    safety_prompt = f"""
Classify whether the CURRENT user message explicitly indicates
an immediate or current risk of physical harm.

Categories:

none:
No explicit current harm risk.

self:
Current intention, urge, plan or immediate risk of self-harm.

baby:
Current intention, urge, plan or immediate risk of harming baby.

other_person:
Current intention, urge, plan or immediate risk of harming
another person.

Do not infer urgent risk only from sadness, stress, anger,
crying, exhaustion or overwhelm.

Historical, hypothetical and clearly non-current statements
are not urgent.

Recent conversation:
{compact_json(recent_history)}

Current user message:
{clean_message}

Return only the structured result.
"""


    payload = {

        "model": SELECTED_LLM_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a bounded explicit-language "
                    "safety classifier. Do not diagnose."
                ),
            },
            {
                "role": "user",
                "content": safety_prompt,
            },
        ],

        "stream": False,

        "format": SAFETY_CLASSIFICATION_SCHEMA,

        "keep_alive": OLLAMA_KEEP_ALIVE,

        "options": {
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": 40,
        },
    }


    started = time.perf_counter()

    try:

        response = HTTP_SESSION.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        response_data = response.json()

        elapsed = round(
            time.perf_counter() - started,
            2,
        )

        print(
            f"Safety LLM wall={elapsed}s | "
            f"load={_seconds(response_data.get('load_duration'))}s | "
            f"prompt={_seconds(response_data.get('prompt_eval_duration'))}s | "
            f"generate={_seconds(response_data.get('eval_duration'))}s"
        )

        raw_content = (
            response_data
            .get(
                "message",
                {},
            )
            .get(
                "content",
                "",
            )
        )

        parsed = json.loads(
            raw_content
        )

        detected = bool(
            parsed.get(
                "urgent_safety_detected",
                False,
            )
        )

        category = str(
            parsed.get(
                "safety_category",
                "none",
            )
            or "none"
        ).strip()

        if category not in SAFETY_CATEGORIES:

            return {
                "urgent_safety_detected": False,
                "safety_category": "none",
            }

        if not detected:

            category = "none"

        if detected and category == "none":

            return {
                "urgent_safety_detected": False,
                "safety_category": "none",
            }

        return {
            "urgent_safety_detected": detected,
            "safety_category": category,
        }


    except (
        requests.RequestException,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ):

        return {
            "urgent_safety_detected": False,
            "safety_category": "none",
        }


# deterministic intent classifier
def classify_assistant_intent(
    user_message,
    chat_history=None,
):
    clean_message = str(
        user_message or ""
    ).strip()

    if not clean_message:
        return "conversation"

    text = clean_message.lower()
    

    history = _prepare_history(
        chat_history,
        limit=4,
    )


   #follow  up resources
    resource_follow_ups = (
        "anything else",
        "something else",
        "any others",
        "another resource",
        "another article",
        "another podcast",
        "another video",
        "more resources",
        "more articles",
        "more podcasts",
        "more videos",
    )

    if any(
        term in text
        for term in resource_follow_ups
    ):

        for previous in reversed(history):

            if previous["role"] != "assistant":
                continue

            if (
                "Resources shown:"
                in previous["content"]
            ):
                return "resources"

            break
    # edit 

    explicit_summary_terms = (
        "personal summary",
        "wellbeing summary",
        "my summary",
        "summary draft",
    )

    explicit_appointment_terms = (
        "appointment point",
        "appointment points",
        "discussion point",
        "discussion points",
    )

    if any(term in text for term in explicit_summary_terms):
        return "summary"

    if any(term in text for term in explicit_appointment_terms):
        return "appointment_points"

    #summary and appointment point editing

    edit_terms = (
        "shorter",
        "shorten",
        "rewrite",
        "reword",
        "refine",
        "simplify",
        "more concise",
        "improve",
        "add",
        "include",
        "mention",
        "change",
        "edit",
        "update",
        "remove",
        "delete",
        "take out",
        "leave out",
        "drop",
        "get rid of",
    )

    if any(
        term in text
        for term in edit_terms
    ):

        for previous in reversed(history):

            if previous["role"] != "assistant":
                continue

            content = previous["content"]

            if "Appointment points:" in content:
                return "appointment_points"

            if "Summary draft:" in content:
                return "summary"

            if content.strip():
                break


    # summary request
    summary_terms = (
        "summary",
        "summarise",
        "summarize",
    )

    if any(
        term in text
        for term in summary_terms
    ):
        return "summary"


    # appointment request

    appointment_terms = (
        "appointment point",
        "discussion point",
        "what should i tell my doctor",
        "what can i tell my doctor",
        "what should i tell my healthcare",
        "what can i tell my healthcare",
        "what should i discuss with my doctor",
        "what can i discuss with my doctor",
        "what should i mention at my appointment",
        "what can i mention at my appointment",
        "prepare me for my appointment",
        "prepare for my appointment",
        "what should i say at my appointment",
        "what can i say at my appointment",
        "what should i bring up with my doctor",
        "what can i bring up with my doctor",
    )

    if any(
        term in text
        for term in appointment_terms
    ):
        return "appointment_points"


    # resource request
    resource_terms = (
        "give me a resource",
        "give me resources",
        "show me a resource",
        "show me resources",
        "find me a resource",
        "find me resources",
        "recommend a resource",
        "recommend resources",
        "something to read",
        "something to watch",
        "something to listen",
        "find me an article",
        "show me an article",
        "recommend an article",
        "find me a podcast",
        "show me a podcast",
        "recommend a podcast",
        "find me a video",
        "show me a video",
        "recommend a video",
    )

    if any(
        term in text
        for term in resource_terms
    ):
        return "resources"
# default
    return "conversation"


# the main llm reponse 
def generate_llm_response(
    application_state: dict[str, Any],
    user_message: str,
    task: str = "conversation",
    retrieved_resources: list[dict[str, Any]] | None = None,
    chat_history=None,
) -> dict[str, Any]:

    clean_message = str(
        user_message or ""
    ).strip()

    if not clean_message:

        return _empty_response(
            "Please enter a message before continuing."
        )


    if task not in VALID_TASKS:

        return _empty_response(
            "The requested assistant task is not supported."
        )


    # build a good and compact propmt
    user_prompt = _build_user_prompt(
        application_state=application_state,
        user_message=clean_message,
        task=task,
        retrieved_resources=retrieved_resources,
        chat_history=chat_history,
    )


    # debug information
    print(
        "Assistant state keys:",
        list(application_state.keys()),
    )
    print(
        "System prompt chars:",
        len(SYSTEM_PROMPT),
    )
    print(
        "User prompt chars:",
        len(user_prompt),
    )

    print(
        "History messages:",
        len(chat_history or []),
    )

    print(
        "Resources supplied:",
        len(retrieved_resources or []),
    )


    # the Ollama payload
    payload = {

        "model": SELECTED_LLM_MODEL,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        "stream": False,

        "keep_alive": OLLAMA_KEEP_ALIVE,

        "options": {
            "temperature": 0.1,

            # Reduced from 4096.
            # Your prompts are relatively short.
            "num_ctx": OLLAMA_CONTEXT_SIZE,

            # Much smaller than the original 500.
            "num_predict": (
                55
                if retrieved_resources
                else TASK_TOKEN_LIMITS.get(task, 80)
            ),
        },
    }


    # Conversation returns plain text.
    # Other tasks use task-specific structured JSON.
    if task != "conversation":

        payload["format"] = TASK_RESPONSE_SCHEMAS.get(
            task
        )

# send request
    started = time.perf_counter()

    try:

        response = HTTP_SESSION.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        response_data = response.json()


        # diagnostics of Ollams performance
        wall_time = round(
            time.perf_counter() - started,
            2,
        )

        load_time = _seconds(
            response_data.get(
                "load_duration"
            )
        )

        prompt_time = _seconds(
            response_data.get(
                "prompt_eval_duration"
            )
        )

        generation_time = _seconds(
            response_data.get(
                "eval_duration"
            )
        )

        prompt_tokens = response_data.get(
            "prompt_eval_count",
            0,
        )

        generated_tokens = response_data.get(
            "eval_count",
            0,
        )


        print(
            f"LLM [{task}] "
            f"wall={wall_time}s | "
            f"load={load_time}s | "
            f"prompt={prompt_time}s "
            f"({prompt_tokens} tokens) | "
            f"generate={generation_time}s "
            f"({generated_tokens} tokens)"
        )


        # calculate the generation speed
        if generation_time > 0:

            tokens_per_second = round(
                generated_tokens / generation_time,
                2,
            )

            print(
                f"Generation speed: "
                f"{tokens_per_second} tokens/sec"
            )


       # get the response
        raw_content = (
            response_data
            .get(
                "message",
                {},
            )
            .get(
                "content",
                "",
            )
            .strip()
        )


        # conversation
        if task == "conversation":

            result = _empty_response("")

            result["assistant_reply"] = (
                raw_content
            )

            return result
        # structured tasks
        parsed_response = json.loads(
            raw_content
        )

        return _apply_task_rules(
            parsed_response,
            task,
        )


    # connection errror
    except requests.ConnectionError:

        return _empty_response(
            "The local assistant is unavailable. "
            "Please make sure Ollama is running and try again."
        )


    # timeout
    except requests.Timeout:

        return _empty_response(
            "The local assistant took too long to respond. "
            "Please try again."
        )


    # other errors
    except (
        requests.RequestException,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:

        print(
            f"Local LLM error: {error}"
        )

        return _empty_response(
            "The assistant could not generate a response safely. "
            "Please try again."
        )