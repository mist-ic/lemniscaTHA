"""
ClearPath RAG Chatbot — Model Router / Classifier

Deterministic 7-signal weighted scorer for query classification.
Routes queries to llama-3.1-8b-instant (simple) or llama-3.3-70b-versatile (complex).
"""

from typing import Dict, List, Optional, Tuple

from app.config import get_settings

# ─── Greeter Detection ─────────────────────────────────────

GREETINGS = {
    "hi", "hello", "hey", "thanks", "thank you",
    "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "sup", "yo",
}

GREETING_RESPONSE = (
    "Hello! I'm ClearPath's support assistant. "
    "I can help you with questions about ClearPath's features, pricing, "
    "integrations, policies, and more. What would you like to know?"
)


def is_greeting(query: str) -> bool:
    """Check if query is a simple greeting that doesn't need LLM."""
    cleaned = query.lower().strip().rstrip("!.,?")
    return cleaned in GREETINGS


# ─── 7-Signal Weighted Scorer ──────────────────────────────

ANALYTICAL_KEYWORDS = {
    "how", "why", "explain", "compare", "difference", "troubleshoot",
    "debug", "analyze", "evaluate", "versus", "vs", "between",
}

ERROR_KEYWORDS = [
    "error", "cannot", "can't", "failed", "broken",
    "not working", "bug", "issue", "not loading",
    "won't load", "doesn't work", "isn't working",
    "isn't loading", "doesn't load", "can't load",
    "won't start", "crash", "crashing",
]

NEGATION_WORDS = {
    "not", "no", "doesn't", "don't", "won't",
    "without", "except", "never", "isn't", "can't",
    "couldn't", "shouldn't", "wouldn't", "hasn't",
    "haven't", "weren't", "wasn't",
}

SENSITIVE_TOPICS = {
    "price", "pricing", "cost", "billing", "payment",
    "security", "data", "privacy", "compliance", "legal",
}


def classify_query(query: str) -> Tuple[str, str, int, Dict]:
    """
    Classify a query using 7 deterministic signals.

    Returns:
        (classification, model, score, signals)
        classification: "simple" or "complex"
        model: Groq model string
        score: integer complexity score
        signals: dict of triggered signals
    """
    settings = get_settings()
    score = 0
    signals: Dict = {}
    query_lower = query.lower()
    words = query_lower.split()
    word_count = len(words)

    # 1. Length signal
    if word_count > 25:
        score += 2
        signals["long_query"] = True
    elif word_count > 15:
        score += 1
        signals["medium_query"] = True

    # 2. Analytical keywords (+2)
    found_analytical = ANALYTICAL_KEYWORDS & set(words)
    if found_analytical:
        score += 2
        signals["analytical_keywords"] = sorted(found_analytical)

    # 3. Error/troubleshooting keywords (+1)
    found_error = [kw for kw in ERROR_KEYWORDS if kw in query_lower]
    if found_error:
        score += 1
        signals["error_keywords"] = True

    # 4. Negation presence (+1)
    if NEGATION_WORDS & set(words):
        score += 1
        signals["negation"] = True

    # 5. Multiple entities (+2) — words starting with uppercase (skip first word)
    if len(query.split()) > 1:
        entities = [
            w for w in query.split()[1:]
            if w[0].isupper() and len(w) > 1
        ]
        if len(entities) >= 2:
            score += 2
            signals["multi_entity"] = True

    # 6. Compound structure (+1)
    if query.count("?") > 1 or query.count(",") > 2 or ";" in query:
        score += 1
        signals["compound"] = True

    # 7. Sensitive topics (+1)
    if SENSITIVE_TOPICS & set(words):
        score += 1
        signals["sensitive_topic"] = True

    # Decision: score >= 4 → complex
    classification = "complex" if score >= 4 else "simple"
    model = (
        settings.COMPLEX_MODEL if classification == "complex"
        else settings.SIMPLE_MODEL
    )

    return classification, model, score, signals


if __name__ == "__main__":
    # Test cases from Plan.md
    test_queries = [
        "Hello",
        "What is ClearPath?",
        "Compare Pro and Enterprise pricing",
        "My timeline view isn't loading after upgrading",
    ]

    for q in test_queries:
        if is_greeting(q):
            print(f"  '{q}' → GREETER (skip LLM)")
            continue
        classification, model, score, signals = classify_query(q)
        model_short = "8B" if "8b" in model else "70B"
        print(f"  '{q}' → {classification} (score={score}, {model_short}) signals={signals}")
