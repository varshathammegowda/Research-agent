from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.paper_search import search_papers
from services.agent_service import run_research_agent


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
    result = search_papers(query)

    return {
        "query": query,
        "result": result
    }


@app.get("/research-agent")
def research_agent(query: str):
    result = run_research_agent(query)

    return result