# 🔬 AI Research Agent

An AI-powered research assistant that automates the process of finding, retrieving, understanding, and summarizing research papers.

The project uses an **AI Agent architecture** to intelligently decide how to perform research tasks instead of following only a fixed RAG pipeline.

## 📌 Overview

The **AI Research Agent** is designed to help users perform academic and technical research efficiently.

A user can enter a research topic or question, and the agent:

1. Understands the user's research query.
2. Determines what information is required.
3. Searches for relevant academic papers.
4. Retrieves available paper content.
5. Extracts and processes relevant information.
6. Uses semantic retrieval to identify useful context.
7. Generates a structured research response using an LLM.

The final output remains focused on providing useful research information such as:

- Research overview
- Key findings
- Relevant papers
- Research gaps
- Important insights

---

## 🤖 AI Agent Architecture

Unlike a traditional fixed RAG pipeline, this project uses an **agent-based architecture**.

### High-Level Architecture

```text
                    ┌─────────────────────┐
                    │      User Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    AI Research      │
                    │       Agent         │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Query Analysis    │
                    │  & Task Planning    │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │   Search    │  │   Retrieve  │  │   Analyze   │
       │   Papers    │  │   Documents  │  │   Context   │
       └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Semantic Retrieval  │
                    │   / RAG Pipeline    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LLM / Gemini   │
                    │ Response Generation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Structured Research │
                    │      Response       │
                    └─────────────────────┘
```

---

## 🧠 How the AI Research Agent Works

### 1. User Query

The user enters a research question or topic through the React frontend.

Example:

```text
"What are the recent approaches for intrusion detection
in Internet of Vehicles?"
```

The query is sent to the backend.

---

### 2. Agentic Query Analysis

The AI agent first analyzes the query and determines what actions are required.

Instead of blindly executing a fixed sequence, the agent can decide which available research tools or retrieval steps are relevant.

For example:

```text
User Query
    ↓
Understand Intent
    ↓
Determine Research Requirements
    ↓
Search / Retrieve Relevant Sources
    ↓
Analyze Retrieved Information
    ↓
Generate Answer
```

This makes the system more flexible than a conventional RAG-only architecture.

---

### 3. Academic Paper Search

The agent can use academic sources such as **OpenAlex** to identify relevant research papers.

The search process retrieves information such as:

- Paper title
- Authors
- Publication information
- Abstract
- DOI
- Available PDF
- Other metadata

---

### 4. Document Retrieval

When a paper has an accessible PDF, the backend retrieves the document.

The PDF is processed using **PyMuPDF**.

```text
Research Paper PDF
        ↓
     PyMuPDF
        ↓
   Extracted Text
```

This allows the system to work with the actual content of research papers rather than relying only on metadata.

---

### 5. Text Processing

The extracted document is divided into smaller chunks.

Chunking allows the system to retrieve only the portions of a paper that are relevant to the user's query.

```text
PDF
 ↓
Text Extraction
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Embeddings
```

---

### 6. Semantic Retrieval

The text chunks are converted into vector embeddings using:

```text
SentenceTransformer
"all-MiniLM-L6-v2"
```

The embeddings are stored and searched using **ChromaDB**.

When the user asks a question, the system performs semantic similarity search to identify the most relevant chunks.

```text
User Query
     ↓
Query Embedding
     ↓
Vector Search
     ↓
Relevant Research Chunks
```

This is the RAG component inside the larger agent architecture.

---

## 🔄 Agent + RAG Architecture

The important distinction is:

### Traditional RAG

```text
Query
  ↓
Retrieve
  ↓
Generate
  ↓
Answer
```

### This Project

```text
Query
  ↓
AI Agent
  ↓
Understand Task
  ↓
Decide Required Actions
  ↓
Search / Retrieve / Analyze
  ↓
RAG Retrieval
  ↓
LLM Reasoning
  ↓
Generate Structured Answer
```

Therefore, **RAG is a component used by the AI agent rather than the entire architecture itself.**

---

## 🧩 Technology Stack

### Frontend

- React.js
- JavaScript
- HTML
- CSS
- Vite

### Backend

- Python
- FastAPI
- Uvicorn

### AI / LLM

- Google Gemini API
- Generative AI
- Agent-based workflow

### RAG / NLP

- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- Semantic Search
- Text Chunking

### Research Data

- OpenAlex API
- Academic Research Papers
- PDF Documents

### PDF Processing

- PyMuPDF

---

##  RAG Configuration

The Research Agent uses the following configuration for document chunking, embedding, and retrieval:

| Parameter | Value |
|---|---|
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Embedding Model | `all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Top-K Retrieved Chunks | 5 |

These parameters control how research documents are divided into chunks, converted into vector embeddings, and retrieved during the semantic search stage.

## ⚙️ Core Components

### AI Research Agent

Responsible for:

- Understanding research queries
- Planning research actions
- Selecting appropriate tools
- Coordinating retrieval and analysis
- Generating the final response

### OpenAlex

Used for discovering relevant academic literature.

### PyMuPDF

Used to extract text from research-paper PDFs.

### Sentence Transformers

Used to generate semantic vector embeddings.

### ChromaDB

Used for vector storage and similarity search.

### Gemini

Used for understanding retrieved context and generating the final research response.

### FastAPI

Provides the backend API layer connecting the frontend with the AI research pipeline.

### React

Provides the interactive research interface.

---

## 🚀 Features

- 🔎 Academic paper search
- 🤖 AI-agent-based research workflow
- 📄 PDF document processing
- 🧠 Semantic search
- 📚 RAG-based contextual retrieval
- ✨ LLM-powered answer generation
- 📊 Structured research results
- 🌐 Web-based interface
- ⚡ FastAPI backend
- 🔐 API-based architecture

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Research-agent
```

### 2. Backend Setup

Navigate to the backend:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Create a `.env` file inside the backend directory.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Add any other API keys required by the current implementation.

**Do not commit `.env` or API keys to GitHub.**

---

### 4. Start the Backend

```bash
uvicorn main:app --reload
```

The FastAPI server will start locally.

---

### 5. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The React development server will start.

---

## 🔗 API Flow

```text
React Frontend
      │
      ▼
FastAPI Backend
      │
      ▼
AI Research Agent
      │
      ├── OpenAlex Search
      │
      ├── PDF Retrieval
      │
      ├── PyMuPDF
      │
      ├── Embedding Model
      │
      ├── ChromaDB
      │
      └── Gemini
              │
              ▼
       Generated Response
              │
              ▼
       React Frontend
```

---

## 🔬 Example Research Workflow

Suppose the user enters:

```text
"Recent machine learning approaches for IoT intrusion detection"
```

The system performs approximately:

```text
1. Receive research query
          ↓
2. AI Agent understands the query
          ↓
3. Identify research requirements
          ↓
4. Search academic literature
          ↓
5. Retrieve relevant papers
          ↓
6. Download accessible PDFs
          ↓
7. Extract PDF text
          ↓
8. Split text into chunks
          ↓
9. Generate embeddings
          ↓
10. Perform semantic retrieval
          ↓
11. Send relevant context to Gemini
          ↓
12. Generate structured research response
```

---

## 🎯 Research Output

The system is designed to provide a concise research-oriented response containing information such as:

### Research Overview

A brief explanation of the research topic.

### Existing Approaches

Important methods and approaches identified from the retrieved literature.

### Key Findings

Major findings from relevant papers.

### Research Gap

Potential areas that are insufficiently explored or could be investigated further.

### Relevant References

Research papers used to support the generated response.

---

## 🧠 Why an AI Agent Instead of Only RAG?

A conventional RAG system generally follows a predefined workflow:

```text
Query → Retrieve → Generate
```

An AI agent introduces a decision-making layer:

```text
Query
  ↓
Agent
  ↓
Decide what to do
  ↓
Use appropriate tools
  ↓
Retrieve information
  ↓
Analyze
  ↓
Generate
```

This allows the system to be extended with additional tools and research capabilities in the future.

For example, additional tools could include:

- Web search
- Citation analysis
- Paper comparison
- Author analysis
- Research trend analysis
- Dataset discovery
- Reference verification

---






## 🚀 Future Enhancements

- Multi-agent research workflow
- Citation verification
- Automatic paper comparison
- Research trend visualization
- Author and citation network analysis
- Better source grounding
- Streaming responses
- Conversation memory
- Multi-document reasoning
- Research report generation

---

## 👩‍💻 Author

**Varsha T**

B.E. Electronics and Communication Engineering  
Dayananda Sagar College of Engineering, Bangalore

---

## 🌐 Project Links

**GitHub Repository:**  
https://github.com/varshathammegowda/Research-agent

---

## ⭐ Project Summary

**AI Research Agent** is an intelligent academic research assistant that combines **AI agents, semantic retrieval, RAG, academic search, PDF processing, and LLM-based generation** to automate the research process and provide structured, context-aware research insights.
