from contextlib import asynccontextmanager

from services.timing import configure_diagnostic_logging, log_process_rss

configure_diagnostic_logging()
log_process_rss("startup_begin")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.paper_search import search_papers
from services.agent_service import run_research_agent

log_process_rss("after_app_imports")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_process_rss("startup_complete")
    yield


app = FastAPI(
    title="AI Academic Research Agent",
    description="An AI-powered academic research assistant",
    version="1.0.0",
    lifespan=lifespan,
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://research-agent-seven-red.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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
