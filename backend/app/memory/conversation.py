"""
ClearPath RAG Chatbot — Conversation Memory

In-memory conversation history with follow-up detection and query rewriting.
Uses the 8B model to rewrite anaphoric queries into standalone form.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Turn dataclass ────────────────────────────────────────

@dataclass
class Turn:
    """A single conversation turn (user query + assistant response)."""
    user_query: str
    assistant_answer: str  # truncated to ~200 chars for token economy
    timestamp: float = field(default_factory=time.time)


# ─── Follow-up detection patterns ─────────────────────────

PRONOUN_PATTERNS = re.compile(
    r'\b(it|that|they|this|its|their|them|those|these|he|she)\b',
    re.IGNORECASE,
)

REFERRING_PHRASES = [
    "about that",
    "from before",
    "you mentioned",
    "you said",
    "previously",
    "as you said",
    "regarding that",
    "the same",
    "more about",
    "tell me more",
    "go on",
    "continue",
    "what about",
    "and also",
    "follow up",
    "following up",
]

# ─── Query rewriting prompt ───────────────────────────────

REWRITE_PROMPT_TEMPLATE = """Given this conversation history:
{history}

Rewrite the following question to be standalone and self-contained, incorporating context from the conversation. Output ONLY the rewritten question, nothing else.

Question: {question}"""


# ─── ConversationMemory ───────────────────────────────────

class ConversationMemory:
    """
    In-memory conversation history manager.

    - Stores last MAX_TURNS turns per conversation_id.
    - Detects follow-up queries (pronouns, short queries, referring phrases).
    - Rewrites follow-up queries using the 8B model for standalone clarity.
    """

    MAX_TURNS = 5
    HISTORY_WINDOW = 3  # number of recent turns to include in generation prompt

    def __init__(self):
        self.conversations: Dict[str, List[Turn]] = {}

    def add_turn(
        self,
        conversation_id: str,
        user_query: str,
        assistant_answer: str,
    ) -> None:
        """Store a completed turn. Truncates assistant answer and evicts old turns."""
        # Truncate assistant answer for token economy
        truncated = assistant_answer[:200] + "..." if len(assistant_answer) > 200 else assistant_answer

        turn = Turn(user_query=user_query, assistant_answer=truncated)

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append(turn)

        # Evict oldest turns beyond MAX_TURNS
        if len(self.conversations[conversation_id]) > self.MAX_TURNS:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.MAX_TURNS:]

    def get_history(self, conversation_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get recent conversation history formatted for prompt builder.

        Returns list of {"user": ..., "assistant": ...} dicts,
        or None if no history exists.
        """
        turns = self.conversations.get(conversation_id, [])
        if not turns:
            return None

        # Return last HISTORY_WINDOW turns
        recent = turns[-self.HISTORY_WINDOW:]
        return [
            {"user": t.user_query, "assistant": t.assistant_answer}
            for t in recent
        ]

    def has_history(self, conversation_id: str) -> bool:
        """Check if a conversation has any history."""
        return bool(self.conversations.get(conversation_id))

    def is_followup(self, query: str, conversation_id: str) -> bool:
        """
        Detect if a query is a follow-up that needs rewriting.

        A query is a follow-up if:
        1. Contains pronouns (it, that, they, this, etc.)
        2. Is very short (<5 words) AND previous turn exists
        3. Contains referring phrases (about that, you mentioned, etc.)
        """
        if not self.has_history(conversation_id):
            return False

        query_lower = query.lower().strip()
        words = query_lower.split()

        # Check 1: Pronouns
        if PRONOUN_PATTERNS.search(query_lower):
            return True

        # Check 2: Short query with existing history
        if len(words) < 5:
            return True

        # Check 3: Referring phrases
        for phrase in REFERRING_PHRASES:
            if phrase in query_lower:
                return True

        return False

    def build_rewrite_prompt(self, query: str, conversation_id: str) -> List[Dict[str, str]]:
        """
        Build the message list for query rewriting via the 8B model.

        Returns messages formatted for GroqClient.generate().
        """
        turns = self.conversations.get(conversation_id, [])
        recent = turns[-self.HISTORY_WINDOW:]

        # Format history
        history_lines = []
        for turn in recent:
            history_lines.append(f"User: {turn.user_query}")
            history_lines.append(f"Assistant: {turn.assistant_answer}")

        history_text = "\n".join(history_lines)

        prompt = REWRITE_PROMPT_TEMPLATE.format(
            history=history_text,
            question=query,
        )

        return [
            {"role": "system", "content": "You rewrite user questions to be standalone. Output ONLY the rewritten question."},
            {"role": "user", "content": prompt},
        ]

    async def rewrite_query(
        self,
        query: str,
        conversation_id: str,
        groq_client: Any,
        model: str,
    ) -> str:
        """
        Rewrite a follow-up query into a standalone question using the 8B model.

        Args:
            query: The original follow-up query.
            conversation_id: The conversation ID.
            groq_client: GroqClient instance.
            model: Model string (should be 8B for economy).

        Returns:
            Rewritten standalone query string.
        """
        messages = self.build_rewrite_prompt(query, conversation_id)

        try:
            result = groq_client.generate(
                messages=messages,
                model=model,
                max_tokens=128,  # rewritten query should be short
            )
            rewritten = result["content"].strip()

            # Sanity check: if rewrite is empty or crazy long, use original
            if not rewritten or len(rewritten) > 500:
                return query

            print(f"[Memory] Rewrote: \"{query}\" → \"{rewritten}\"")
            return rewritten

        except Exception as e:
            print(f"[Memory] Rewrite failed ({e}), using original query")
            return query
