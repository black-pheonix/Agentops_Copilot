# AgentOps Copilot

**AgentOps Copilot** is an engineering copilot built for UAV developers and ArduPilot ecosystem. Instead of relying solely on LLM memory, it combines semantic search, retrieval-augmented generation (RAG), and tool-calling workflows to answer technical questions directly from official ArduPilot documentation.

The system automatically ingests documentation, generates vector embeddings, retrieves the most relevant knowledge using semantic search, and can dynamically fetch full documentation pages when additional context is required. The result is a domain-specific AI assistant designed for UAV developers, robotics engineers, and ArduPilot users.

---

## Why This Project Exists

Large language models often hallucinate when answering highly technical questions about flight controllers, navigation systems, sensor fusion, and UAV configuration.

AgentOps Copilot addresses this by:

* Retrieving answers from official ArduPilot documentation
* Using semantic search instead of keyword matching
* Allowing the agent to call documentation tools dynamically
* Providing source-grounded technical responses
* Supporting both standard and streaming interactions

---

## Key Features

* Semantic search over ArduPilot documentation
* Retrieval-Augmented Generation (RAG)
* Tool-calling AI agent architecture
* Dynamic documentation retrieval
* FastAPI REST backend
* Real-time streaming responses via Server-Sent Events (SSE)
* Pinecone vector database integration
* Hugging Face embedding generation
* Production deployment on Render
* API-first design for future frontend integration

---

# Architecture

```text
                              ┌────────────────────┐
                              │     User Query     │
                              └─────────┬──────────┘
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │      FastAPI API       │
                           │  /ask   /ask/stream    │
                           └─────────┬──────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │      Agent Controller      │
                        │    Llama 3.1 Tool Agent    │
                        └─────────┬──────────────────┘
                                  │
                 ┌────────────────┴─────────────────┐
                 │                                  │
                 ▼                                  ▼

      ┌──────────────────────┐         ┌──────────────────────┐
      │     search_docs()    │         │     fetch_page()     │
      │   Semantic Retrieval │         │ Full Document Fetch  │
      └──────────┬───────────┘         └──────────┬───────────┘
                 │                               │
                 ▼                               ▼

      ┌──────────────────────┐      ┌─────────────────────────┐
      │ Pinecone Vector DB   │      │ Official ArduPilot Docs │
      │ Documentation Chunks │      │       Website           │
      └──────────┬───────────┘      └─────────────────────────┘
                 │
                 ▼

      ┌──────────────────────┐
      │ MiniLM Embeddings    │
      │ Retrieval Layer      │
      └──────────┬───────────┘
                 │
                 ▼

      ┌──────────────────────┐
      │   Grounded Answer    │
      └──────────────────────┘
```

---

## Technical Stack

### AI Layer

* Llama 3.1 8B Instant (Groq)
* all-MiniLM-L6-v2 Embeddings
* Hugging Face Inference API

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Retrieval

* Pinecone
* Recursive Character Chunking
* Semantic Vector Search

### Data Processing

* BeautifulSoup
* Requests
* LXML

### Deployment

* Render
* Python 3.12

---

# How It Works

### Step 1: Documentation Ingestion

The ingestion pipeline:

1. Crawls selected ArduPilot documentation pages
2. Cleans HTML content
3. Splits documents into semantic chunks
4. Generates embeddings using MiniLM
5. Stores vectors inside Pinecone


### Step 2: Question Processing

When a user submits a question:

1. Llama 3.1 receives the query
2. Agent selects the appropriate tool
3. Semantic retrieval searches documentation
4. Additional pages can be fetched if required
5. Grounded answer is generated

This reduces hallucinations compared to a standalone LLM.

---

# Running Locally

## 1. Clone Repository

```bash
git clone https://github.com/black-pheonix/Agentops_Copilot.git

cd Agentops_Copilot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
api_key=YOUR_GROQ_API_KEY

pinecone_key=YOUR_PINECONE_API_KEY

HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

---

## 5. Build Documentation Index

```bash
python ingestion/ingest.py
```

This:

* Scrapes documentation
* Chunks content
* Generates embeddings
* Uploads vectors to Pinecone

---

## 6. Start API Server

```bash
uvicorn api/main:app --reload
```

Server:

```text
http://localhost:8000
```

---

# API Reference

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "AgentOps Copilot"
}
```

---

## Ask a Question
Go to:
```text
http://localhost:8000/docs
```

```http
POST /ask
```

Request:

```json
{
  "text": "What is EKF?"
}
```

Response:

```json
{
  "question": "What is EKF?",
  "answer": "..."
}
```

---

## Streaming Endpoint

```http
GET /ask/stream?text=What%20is%20EKF
```

Returns real-time SSE events:

```json
{
  "event": "status",
  "message": "Searching documentation..."
}
```

```json
{
  "event": "answer",
  "answer": "..."
}
```

---

# Live API

Production deployment:

**[https://agentops-copilot.onrender.com/](https://agentops-copilot.onrender.com/)**

Health Check:

**[https://agentops-copilot.onrender.com/health](https://agentops-copilot.onrender.com/health)**

Interactive Swagger Docs:

**[https://agentops-copilot.onrender.com/docs](https://agentops-copilot.onrender.com/docs)**

---

# Engineering Highlights

* Built custom documentation ingestion pipeline
* Implemented semantic retrieval over technical UAV documentation
* Designed tool-calling workflow with dynamic documentation access
* Developed streaming API architecture using Server-Sent Events
* Integrated Groq inference, Hugging Face embeddings, and Pinecone retrieval
* Deployed production-ready backend on Render

---

## Roadmap
* MCP server for Claude Desktop integration
* Frontend dashboard
* Evaluation metrics (retrieval precision, answer faithfulness)
* Fine-tuned embedding model for UAV-specific terminology

---

## Example Query

```text
How does EKF3 use GPS and IMU data for state estimation?
```

Agent workflow:

```text
Question
   ↓
Semantic Search
   ↓
Retrieve Documentation
   ↓
Optional Page Fetch
   ↓
LLM Reasoning
   ↓
Grounded Response
```

