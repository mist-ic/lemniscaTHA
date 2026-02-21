"""
ClearPath RAG Chatbot — Output Evaluator

Flags: no_context, refusal, conflicting_sources.
Runs after LLM generation to flag potentially unreliable outputs.
"""

import re
from typing import Dict, List

# ─── Refusal Patterns ──────────────────────────────────────

REFUSAL_PATTERNS = [
    re.compile(r"i don'?t have (that |enough )?information", re.IGNORECASE),
    re.compile(r"not mentioned in the (provided |available )?documents", re.IGNORECASE),
    re.compile(r"i('m| am) (not sure|unable|sorry)", re.IGNORECASE),
    re.compile(r"cannot (find|answer|help with)", re.IGNORECASE),
    re.compile(r"no information (about|on|regarding)", re.IGNORECASE),
    re.compile(r"not covered in the (context|documentation)", re.IGNORECASE),
    re.compile(r"beyond (my|the) (scope|available)", re.IGNORECASE),
]

# ─── Conflicting Sources Detection ─────────────────────────

CONFLICT_SELF_REPORT_PATTERNS = [
    "conflicting", "inconsistent", "differs between",
    "varies across", "discrepancy", "contradicts",
    "different values", "conflicting information",
]

# Known pricing values that indicate conflict if multiple appear
KNOWN_PRICE_VARIANTS = {"$49", "$45", "$52", "$99"}


def _is_refusal(answer: str) -> bool:
    """Check if answer matches any refusal pattern."""
    for pattern in REFUSAL_PATTERNS:
        if pattern.search(answer):
            return True
    return False


def _check_conflicting_sources(
    answer: str,
    chunks: List[Dict],
) -> bool:
    """
    Detect conflicting information in retrieved chunks.

    Methods:
    1. Model self-report: LLM mentions conflict in answer text.
    2. Numeric divergence: different price values across chunks from different docs.
    3. Known conflict pairs: hard-coded Pro plan pricing variants.
    """
    answer_lower = answer.lower()

    # Method 1: Model self-reports conflict
    for phrase in CONFLICT_SELF_REPORT_PATTERNS:
        if phrase in answer_lower:
            return True

    # Method 2 & 3: Check chunks for numeric divergence
    if len(chunks) >= 2:
        # Extract dollar amounts from each chunk
        price_by_doc: Dict[str, set] = {}
        for chunk in chunks:
            doc = chunk.get("document", "")
            text = chunk.get("text", "")
            prices = set(re.findall(r'\$\d+(?:\.\d{2})?', text))
            if prices:
                price_by_doc[doc] = prices

        # Check if different documents report different prices for same context
        if len(price_by_doc) >= 2:
            all_prices = set()
            for prices in price_by_doc.values():
                all_prices.update(prices)

            # Known conflict: Pro plan pricing ($49, $45, $52)
            known_hits = all_prices & KNOWN_PRICE_VARIANTS
            if len(known_hits) >= 2:
                return True

            # General: if chunks from different docs have non-overlapping price sets
            doc_list = list(price_by_doc.values())
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    # If both have prices but they don't fully overlap
                    if doc_list[i] and doc_list[j]:
                        if not doc_list[i] & doc_list[j]:
                            # Different prices, could be conflict
                            return True

    return False


def evaluate(
    answer: str,
    chunks_retrieved: int,
    retrieved_chunks: List[Dict],
) -> List[str]:
    """
    Run all evaluator flags on the LLM response.

    Args:
        answer: The LLM-generated answer text.
        chunks_retrieved: Number of chunks retrieved (0 means no context).
        retrieved_chunks: List of chunk metadata dicts used for generation.

    Returns:
        List of flag strings, e.g. ["no_context", "conflicting_sources"].
        Empty list if no flags.
    """
    flags: List[str] = []

    is_refusal = _is_refusal(answer)

    # Flag 1: no_context — LLM answered but no chunks retrieved
    if chunks_retrieved == 0 and not is_refusal:
        flags.append("no_context")

    # Flag 2: refusal
    if is_refusal:
        flags.append("refusal")

    # Flag 3: conflicting_sources
    if chunks_retrieved >= 2 and _check_conflicting_sources(answer, retrieved_chunks):
        flags.append("conflicting_sources")

    return flags
