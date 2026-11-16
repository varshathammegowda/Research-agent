"""Validated actions emitted by the research decision model."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    action: Literal["search_papers", "get_paper_details", "retrieve_pdf", "extract_pdf_text", "index_document", "retrieve_relevant_content", "finish"]
    reason: str = Field(default="", max_length=300)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ResearchState(BaseModel):
    query: str
    objective: str = ""
    plan: list[str] = Field(default_factory=list)
    papers: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    actions: list[dict[str, str]] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    iteration: int = 0
    answer: str = ""
