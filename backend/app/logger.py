"""
ClearPath RAG Chatbot — Structured Logger

JSON-formatted logging using structlog.
Logs every query with all required fields per the assignment spec.
"""

import structlog


def setup_logger():
    """Configure structlog for JSON-formatted output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize on import
setup_logger()
logger = structlog.get_logger("clearpath")


def log_query(
    query: str,
    classification: str,
    model_used: str,
    complexity_score: int,
    signals: dict,
    tokens_input: int,
    tokens_output: int,
    latency_ms: int,
    conversation_id: str,
    evaluator_flags: list,
    chunks_retrieved: int = 0,
):
    """Log a complete query processing event."""
    logger.info(
        "query_processed",
        query=query,
        classification=classification,
        model_used=model_used,
        complexity_score=complexity_score,
        signals=signals,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        latency_ms=latency_ms,
        conversation_id=conversation_id,
        evaluator_flags=evaluator_flags,
        chunks_retrieved=chunks_retrieved,
    )
