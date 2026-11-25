# 🔬 AI Academic Research Agent

An AI-powered academic research assistant that searches academic papers, retrieves relevant research evidence from PDFs, and uses Retrieval-Augmented Generation (RAG) with Gemini to generate research-based answers.

---

## 🚀 Project Overview

The AI Academic Research Agent combines academic search, document processing, semantic search, vector databases, and Generative AI into a single research workflow.

A user enters a research question such as:

> What machine learning techniques are used for IoT security?

The system then:

1. Searches academic papers using OpenAlex.
2. Extracts text from research PDFs.
3. Splits documents into smaller chunks.
4. Converts chunks into embeddings.
5. Stores embeddings in ChromaDB.
6. Performs semantic similarity search.
7. Retrieves relevant research evidence.
8. Sends the retrieved evidence to Gemini.
9. Generates a structured research answer.
10. Displays the answer and related academic papers in the React frontend.

---

## 🏗️ System Architecture

```text
                    User
                     │
                     ▼
              React Frontend
                     │
                     ▼
                Vite Proxy
                     │
                     ▼
              FastAPI Backend
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
      OpenAlex              ChromaDB
   Academic Search       Semantic Search
          │                     │
          │              Research Evidence
          │                     │
          └──────────┬──────────┘
                     ▼
                Gemini LLM
                     │
                     ▼
             Research Answer
                     │
                     ▼
              React Frontend


✨ Features
🔎 Academic paper search
📄 PDF text extraction
✂️ Text chunking with overlap
🧠 Sentence Transformer embeddings
🗄️ ChromaDB vector database
🔍 Semantic similarity search
📚 Retrieval-Augmented Generation (RAG)
🤖 Gemini-powered research synthesis
🌐 React frontend
⚡ FastAPI backend
🔗 Academic paper links
🔐 Environment-based API key management
🛠️ Tech Stack
Frontend
React
Vite
JavaScript
HTML
CSS
Backend
Python
FastAPI
Uvicorn
AI / ML
Sentence Transformers
Gemini
Retrieval-Augmented Generation (RAG)
Database
ChromaDB
Academic Search
OpenAlex
Document Processing
PyMuPDF
Development
Git
GitHub
VS Code
📂 Project Structure
academic-research-agent/
│
├── backend/
│   │
│   ├── data/
│   │   ├── papers/
│   │   │   └── sample.pdf
│   │   └── chroma/
│   │
│   ├── services/
│   │   ├── agent_service.py
│   │   ├── embedding_service.py
│   │   ├── paper_search.py
│   │   ├── pdf_parser.py
│   │   ├── text_chunker.py
│   │   └── vector_store.py
│   │
│   ├── main.py
│   ├── .env
│   └── requirements.txt
│
├── frontend/
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── vite.config.js
│   ├── package.json
│   └── package-lock.json
│
├── .gitignore
└── README.md
🧠 How the System Works
1. Academic Paper Search

The system uses OpenAlex to search for relevant academic papers.

Example:

User:
What machine learning techniques are used for IoT security?

                    ↓

OpenAlex

                    ↓

5 relevant academic papers

The returned information includes:

Paper title
Authors
Publication year
Abstract
Paper URL
2. PDF Processing

Research PDFs are processed using PyMuPDF.

PDF
 ↓
Text Extraction
 ↓
Raw Research Text

The extracted text can then be processed by the RAG pipeline.

3. Text Chunking

Large documents are divided into smaller chunks.

The project uses:

Chunk size = 1000 characters
Overlap = 200 characters

The overlap helps preserve context between neighboring chunks.

Document
│
├── Chunk 1
│   └── characters 0–999
│
├── Chunk 2
│   └── characters 800–1799
│
├── Chunk 3
│   └── characters 1600–2599
│
└── ...
4. Embeddings

Each text chunk is converted into a numerical vector using:

all-MiniLM-L6-v2

Each chunk is represented as a:

384-dimensional vector

The vector captures the semantic meaning of the text.

5. ChromaDB

The generated embeddings are stored in ChromaDB.

Research Chunk
      ↓
Embedding
      ↓
ChromaDB

When a user asks a question, the question is also converted into an embedding.

The system then searches ChromaDB for semantically similar research chunks.

6. Retrieval-Augmented Generation

The project uses RAG instead of sending a question directly to the LLM.

User Question
      ↓
Question Embedding
      ↓
ChromaDB
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

## 📂 Project Architecture

```text
Research-Agent/
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── ...
│
├── backend/
│   ├── main.py
│   ├── agent/
│   ├── rag/
│   ├── services/
│   ├── utils/
│   └── ...
│
├── requirements.txt
├── package.json
├── .env
└── README.md
```

> The exact folder structure may vary depending on the current implementation.

---

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

cd backend

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

python -m pip install fastapi uvicorn requests python-dotenv pymupdf chromadb sentence-transformers google-genai
🔑 Environment Variables

Create:

backend/.env

Add:

GEMINI_API_KEY=your_gemini_api_key
OPENALEX_API_KEY=your_openalex_api_key

Never commit .env to GitHub.

▶️ Run the Backend

From the backend directory:

uvicorn main:app --reload

The backend will run at:

http://127.0.0.1:8000
⚛️ Frontend Setup

Open another terminal.

Navigate to:

cd frontend

Install dependencies:

npm install

Start the development server:

npm run dev

The frontend will run at:

http://localhost:5173
🔬 Example Research Question

Enter:

What machine learning techniques are used for IoT security?

The agent retrieves relevant academic research and generates a structured answer covering topics such as:

Decision Trees
Random Forest
SVM
CNN
LSTM
GRU
Federated Learning
Anomaly Detection
Explainable AI
Edge-oriented ML techniques

The frontend also displays related academic papers with links.

🔐 Security

API keys are stored using environment variables.

The following files and directories should not be committed:

.env
venv/
node_modules/
data/chroma/

These are excluded through .gitignore.



📌 Future Improvements

📑 Upload PDFs directly from the frontend
📚 Search and compare multiple research papers
📝 Automatic literature review generation
🔗 Better citation management
📊 Research trend analysis
🧠 Multi-agent research workflow
💬 Conversational research assistant
📈 Paper relevance ranking
🗂️ Research history
🔐 User authentication

👩‍💻 Author
Varsha Thammegowda
Built as an AI/ML research project to explore:

Generative AI + RAG + Academic Search + Agentic AI


## Running with Docker

1. Copy `.env.example` to `.env` in the project root and set `GEMINI_API_KEY`. Set `OPENALEX_API_KEY` if you use one; it is optional. The root `.env` is read by Docker Compose and is not copied into either image.
2. Build and start the application:

```bash
docker compose up --build
```

3. Open the frontend at `http://localhost:8080`. The FastAPI backend is available at `http://localhost:8000` (API docs: `http://localhost:8000/docs`). The frontend proxies `/api/` requests to the backend at `http://backend:8000` over the private Compose network.

Useful commands:

```bash
# Run in background
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f
```

To rebuild after code changes, run `docker compose up --build` again. ChromaDB data is persisted in `backend/data/chroma` on the host, including the existing local database.
