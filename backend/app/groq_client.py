"""
ClearPath RAG Chatbot — Groq API Client

Wrapper for Groq SDK with streaming, retry, and token tracking.
"""

import time
from typing import Any, Dict, Generator, List, Optional

from groq import Groq, RateLimitError, APIStatusError

from app.config import get_settings


class GroqClient:
    """Wraps the Groq SDK with retry logic and token tracking."""

    def __init__(self):
        settings = get_settings()
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Non-streaming completion.

        Returns:
            {
                "content": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": int,
            }
        """
        last_error = None
        for attempt in range(3):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    stream=False,
                )
                latency_ms = int((time.perf_counter() - start) * 1000)

                return {
                    "content": response.choices[0].message.content or "",
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "latency_ms": latency_ms,
                }

            except RateLimitError as e:
                last_error = e
                wait = _backoff_wait(attempt, e)
                print(f"[GroqClient] Rate limited (attempt {attempt+1}/3), waiting {wait:.1f}s")
                time.sleep(wait)

            except APIStatusError as e:
                if e.status_code in (503, 529):
                    last_error = e
                    wait = _backoff_wait(attempt, e)
                    print(f"[GroqClient] Service unavailable (attempt {attempt+1}/3), waiting {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error  # type: ignore

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 512,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming completion — yields token deltas.

        Yields dicts:
            {"token": "partial text"}     — for each token
            {"done": True, "prompt_tokens": int, "completion_tokens": int, "latency_ms": int}
        """
        last_error = None
        for attempt in range(3):
            try:
                start = time.perf_counter()
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    stream=True,
                )

                prompt_tokens = 0
                completion_tokens = 0

                for chunk in stream:
                    # Token delta
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield {"token": delta.content}

                    # Usage info (appears in the final chunk for Groq)
                    if hasattr(chunk, "x_groq") and chunk.x_groq and hasattr(chunk.x_groq, "usage"):
                        usage = chunk.x_groq.usage
                        prompt_tokens = usage.prompt_tokens
                        completion_tokens = usage.completion_tokens

                latency_ms = int((time.perf_counter() - start) * 1000)

                yield {
                    "done": True,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                }
                return  # success, exit retry loop

            except RateLimitError as e:
                last_error = e
                wait = _backoff_wait(attempt, e)
                print(f"[GroqClient] Rate limited during stream (attempt {attempt+1}/3), waiting {wait:.1f}s")
                time.sleep(wait)

            except APIStatusError as e:
                if e.status_code in (503, 529):
                    last_error = e
                    wait = _backoff_wait(attempt, e)
                    print(f"[GroqClient] Service unavailable during stream (attempt {attempt+1}/3), waiting {wait:.1f}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error  # type: ignore


def _backoff_wait(attempt: int, error: Exception) -> float:
    """Exponential backoff: 1s, 2s, 4s. Respects retry-after header if present."""
    base = 2 ** attempt  # 1, 2, 4

    # Try to extract retry-after from response headers
    if hasattr(error, "response") and error.response is not None:
        retry_after = error.response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), base)
            except ValueError:
                pass

    return float(base)
