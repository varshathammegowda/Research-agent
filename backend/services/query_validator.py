"""Semantic validation for questions submitted to the research agent."""

import json
import logging
import os

logger = logging.getLogger(__name__)

VALIDATION_INSTRUCTIONS = """Classify whether a user's query asks for information meaningfully related to academic, scientific, technical, or research-oriented knowledge.

Allow explanatory and conceptual questions about technical/scientific topics (for example 'What are LLMs?', 'Explain transformers', or 'How does RAG work?'), comparisons of technical methods, and research questions. A query need not mention research or papers.

Reject unrelated everyday requests, biography/factual questions about public figures without a research framing, coding tasks asking you to write/debug code, recipes, jokes, weather, and time requests. Judge intent and meaning, not keywords.

Return ONLY a JSON object with exactly two fields: {\"valid\": boolean, \"reason\": string}."""

INVALID_QUERY_MESSAGE = (
    "This system is designed for academic, scientific, and research-oriented "
    "questions. Please enter a research-oriented question."
)


def validate_query(query: str, client) -> dict:
    """Return structured semantic classification using the shared Gemini client.

    On provider errors or malformed output, fail closed: the agent must not
    send an unvalidated query to OpenAlex. The caller can return a retry message.
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=(
                f"{VALIDATION_INSTRUCTIONS}\n\nUser query:\n{query}"
            ),
            config={"response_mime_type": "application/json"},
        )
        result = json.loads(response.text or "")
        if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
            raise ValueError("Gemini returned an invalid validation schema")
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Gemini returned no validation reason")
        return {"valid": result["valid"], "reason": reason.strip()}
    except Exception as exc:
        api_key = os.getenv("GEMINI_API_KEY") or ""
        safe_error = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        logger.error("Query validation failed: %s: %s", type(exc).__name__, safe_error)
        return {
            "valid": False,
            "reason": "Query validation is temporarily unavailable.",
            "error": True,
        }
