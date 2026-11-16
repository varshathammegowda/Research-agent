from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.paper_search import search_papers
from services.agent_service import run_research_agent, client
from services.query_validator import validate_query, INVALID_QUERY_MESSAGE


app = FastAPI(
    title="AI Academic Research Agent",
    description="An AI-powered academic research assistant",
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "AI Academic Research Agent is running!"
    }


@app.get("/research")
def research(query: str):
    validation = validate_query(query, client)
    if not validation.get("valid"):
        if validation.get("error"):
            return {
                "query": query,
                "result": {
                    "success": False,
                    "error": (
                        "Query validation is temporarily unavailable. "
                        "Please try again shortly."
                    ),
                },
            }
        return {
            "query": query,
            "result": {
                "success": False,
                "answer": INVALID_QUERY_MESSAGE,
                "papers": [],
            },
        }
    result = search_papers(query)

    return {
        "query": query,
        "result": result
    }


@app.get("/research-agent")
def research_agent(query: str):
    result = run_research_agent(query)

    return result


@app.post("/api/research")
def research_post(payload: dict):
    query = payload.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "A research query is required."}
    return run_research_agent(query.strip())
