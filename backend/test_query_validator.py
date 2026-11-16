from types import SimpleNamespace
from unittest.mock import Mock

from services.query_validator import validate_query


def _client_with_result(payload):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=payload
    )
    return client


def test_valid_research_query():
    client = _client_with_result(
        '{"valid": true, "reason": "Research question about RAG."}'
    )
    assert validate_query("What are recent approaches to RAG?", client)["valid"]


def test_invalid_everyday_query():
    client = _client_with_result(
        '{"valid": false, "reason": "A cooking request."}'
    )
    assert not validate_query("How do I cook biryani?", client)["valid"]


def test_ambiguous_technical_concept_is_valid():
    client = _client_with_result(
        '{"valid": true, "reason": "Asks for technical explanation."}'
    )
    assert validate_query("Explain transformers.", client)["valid"]


def test_gemini_validation_error_returns_fail_closed_result():
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError("provider down")

    result = validate_query("How does RAG work?", client)

    assert result["valid"] is False
    assert result["error"] is True


def test_validation_api_failure_fails_closed_before_pipeline(monkeypatch):
    # agent_service requires GEMINI_API_KEY while importing; provide a dummy
    # value because the client itself is replaced for this orchestration test.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from services import agent_service

    monkeypatch.setattr(
        agent_service,
        "validate_query",
        lambda query, client: {"valid": False, "error": True},
    )
    search = Mock(side_effect=AssertionError("OpenAlex must not be called"))
    embed = Mock(side_effect=AssertionError("embeddings must not be created"))
    retrieve = Mock(side_effect=AssertionError("ChromaDB must not be queried"))
    monkeypatch.setattr(agent_service, "search_papers", search)
    monkeypatch.setattr(agent_service, "create_embeddings", embed)
    monkeypatch.setattr(agent_service, "search_chunks", retrieve)

    result = agent_service.run_research_agent("What is the weather today?")

    assert result["success"] is False
    assert "temporarily unavailable" in result["error"]
    search.assert_not_called()
    embed.assert_not_called()
    retrieve.assert_not_called()
