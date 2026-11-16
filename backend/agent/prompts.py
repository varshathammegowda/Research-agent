"""Compact instructions for planning, tool selection, and grounded synthesis."""

DECISION_PROMPT = """You operate a bounded academic research agent. Select one allowed action as JSON.
Allowed actions: search_papers, get_paper_details, retrieve_pdf, extract_pdf_text, index_document, retrieve_relevant_content, finish.
Use search_papers first. When papers are found and no PDF has been indexed, choose retrieve_pdf (paper_index, from 0) to ingest an available open PDF. Choose retrieve_relevant_content for local PDF evidence.
Do not repeat an action that has already returned results unless a new query is needed.
Finish when available evidence is enough, or when no useful tools remain. Do not reveal private reasoning.
Return JSON only: {"action":"...","reason":"short user-safe status","arguments":{"query":"..."}}.

Query: {query}
Objective: {objective}
Plan: {plan}
Tools already used: {used}
Papers found: {paper_count}
Evidence chunks: {chunk_count}
Recent observations: {observations}
"""

SYNTHESIS_PROMPT = """Answer the research question using only the supplied evidence. Metadata identifies sources but is not evidence for findings. Do not invent papers, authors, dates, or claims. Say clearly when evidence is insufficient. Distinguish reported limitations from gaps inferred across sources; infer a gap only when repeated evidence supports it. Cite the evidence excerpt source printed above each chunk (for example [Paper title] or [local research database]). Be concise.

Question: {query}
Evidence:
{evidence}
Sources:
{sources}
"""
