
# imports the main semantic retrieval function
from rag.retriever import retrieve_resources


# turns one relevance pattern into short text that can be used for retrieval
def format_pattern_for_retrieval(relevance_item):
    if not relevance_item:
        return ""
    category = relevance_item.get("category", "")
    pattern = relevance_item.get("pattern")

    if not pattern:
        return ""

    # some theme patterns are already stored as readable text
    if isinstance(pattern, str):
        return pattern.strip()
    if not isinstance(pattern, dict):
        return str(pattern).strip()
    # use an existing readable description when one is available
    for key in ["description", "summary", "message", "text"]:
        value = pattern.get(key)

        if value:
            return str(value).strip()

    field_name = pattern.get("field")

    if field_name:
        field_text = str(field_name).replace("_", " ").strip()
    else:
        field_text = ""

    # describe a change from the user's own recent baseline
    if category == "recent_change" and field_text:
        direction = pattern.get("direction")

        if direction:
            return (
                f"{field_text} has changed {direction} compared with "
                f"the user's personal baseline"
            )
        return (
            f"{field_text} has changed from the user's personal baseline"
        )

    # describethe  fields that have varied across recent check-ins
    if category == "variable_pattern" and field_text:
        return f"{field_text} has varied across recent check-ins"
    # describe the agreement between reflection and structured answers
    if category == "reflection_alignment":
        theme = pattern.get("theme")
        if theme:
            return (
                f"Reflection and structured check-in responses show a "
                f"similar pattern related to {theme}"
            )
        if field_text:
            return (
                f"Reflection and structured check-in responses are aligned "
                f"around {field_text}"
            )
    # fall back to the field name when no specific format applies
    if field_text:
        return field_text

    return ""


# rread the personal pattern relavance output 

# collects the most useful pattern descriptions from the relevance result
def get_relevance_context(
    relevance_result,
    include_supporting=True,
    max_patterns=5
):
    if not relevance_result:
        return []

    surface_patterns = relevance_result.get("surface_patterns", [])
    supporting_patterns = relevance_result.get("supporting_patterns", [])
    # surface patterns are given priority before supporting patterns
    relevance_items = list(surface_patterns)
    if include_supporting:
        relevance_items.extend(supporting_patterns)

    context = []

    for item in relevance_items:
        pattern_text = format_pattern_for_retrieval(item)
        if pattern_text and pattern_text not in context:
            context.append(pattern_text)
        if len(context) >= max_patterns:
            break

    return context


# build the personalised retival query

# combines the user question with relevant saved patterns
def build_personalised_query(
    user_query,
    relevant_patterns,
    max_patterns=5
):

    user_query = user_query.strip() if user_query else ""
    if not relevant_patterns:
        return user_query
    clean_patterns = []
    # remove empty pattern values before building the query
    for pattern in relevant_patterns:
        if not pattern:
            continue

        pattern_text = str(pattern).strip()

        if pattern_text:
            clean_patterns.append(pattern_text)

    if not clean_patterns:
        return user_query

    # keep the amount of personal context small
    clean_patterns = clean_patterns[:max_patterns]
    pattern_text = "; ".join(clean_patterns)

    if user_query:
        return (
            f"User question: {user_query}. "
            f"Relevant personal patterns from approved check-ins: {pattern_text}. "
            f"Retrieve supportive postpartum wellbeing resources relevant to both "
            f"the question and these patterns."
        )

    return (
        f"Relevant personal patterns from approved check-ins: {pattern_text}. "
        f"Retrieve supportive postpartum wellbeing resources relevant to "
        f"these patterns."
    )

# general personalised retrival 
# retrieves resources using both the user query and relevant personal patterns
def retrieve_personalised_resources(
    user_query,
    relevant_patterns,
    resources,
    embedding_model,
    top_k=3
):

    query = build_personalised_query(
        user_query=user_query,
        relevant_patterns=relevant_patterns
    )

    if not query:
        return []

    # send the combined query through the normal semantic retriever
    return retrieve_resources(
        query=query,
        resources=resources,
        embedding_model=embedding_model,
        top_k=top_k
    )

# Retrieve directly from relevance layer output

# connects the pattern relevance result directly to personalised retrieval
def retrieve_from_relevance(
    user_query,
    relevance_result,
    resources,
    embedding_model,
    top_k=3
):

    relevant_patterns = get_relevance_context(
        relevance_result=relevance_result
    )
    return retrieve_personalised_resources(
        user_query=user_query,
        relevant_patterns=relevant_patterns,
        resources=resources,
        embedding_model=embedding_model,
        top_k=top_k
    )