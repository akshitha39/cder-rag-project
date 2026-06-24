# 📚 CDER GraphRAG: Graph-Based RAG for Parallel Computing Curriculum

A research project comparing three AI retrieval architectures to minimize hallucinations in LLMs when applied to domain-specific educational content (CDER Parallel Computing Curriculum).

> 🏫 Montclair State University | Department of Data Science | June 2025 – Dec 2025

---

## 👥 Authors

- **Karthik Desina** — Computer Science, Montclair State University
- **Akshitha Prathani** — Data Science, Montclair State University  
- **Kruti Enugu** — Data Science, Montclair State University

---

## 🧠 Overview

Large Language Models (LLMs) like GPT-4o often generate fluent but factually incorrect answers — especially in specialized technical domains. This project builds and compares **three question-answering systems** on the CDER Parallel Computing Curriculum to find which architecture best reduces hallucination and improves grounding.

| System | Architecture |
|---|---|
| **System A** | LLM-Only (GPT-4o, no retrieval) |
| **System B** | Vector RAG (ChromaDB + semantic embeddings) |
| **System C** | Graph RAG (Neo4j knowledge graph) |

---

## 📊 Results

Evaluated on **25 instructor-designed questions** from the CDER curriculum:

| System | Correct (out of 25) | Factual Accuracy | Hallucination Rate | Avg Response Time |
|---|---|---|---|---|
| A — LLM Only | 10 / 25 | 40% | High | ~2.0 seconds |
| B — Vector RAG | 18 / 25 | 70% | Moderate | ~1.2 seconds |
| C — Graph RAG | **25 / 25** | **100%** | **Negligible** | **< 0.5 seconds** |

### Key Findings
- **Graph RAG** achieved 100% factual accuracy by leveraging explicit concept relationships in Neo4j
- **Vector RAG** improved accuracy by +30% over the LLM baseline through semantic text grounding
- **Graph RAG** was also the fastest system due to efficient graph traversal for context retrieval

---

## 🏗️ Architecture

```
User Query
    │
    ├──► System A: LLM Only
    │       └── GPT-4o ──────────────────────► Answer (40% accuracy)
    │
    ├──► System B: Vector RAG
    │       └── ChromaDB (semantic vectors)
    │               └── GPT-4o + Context ───► Answer (70% accuracy)
    │
    └──► System C: Graph RAG
            └── Neo4j Knowledge Graph
                    └── GPT-4o + Graph Paths ► Answer (100% accuracy)
```

### Graph RAG Node Relationships
```
CDER Chap 5 ──:DISCUSSES──► Mutex ──:REQUIRES──► Thread
                                └──:SOLVES──► Race Condition
```
Relationship types: `COVERS_TOPIC`, `REQUIRES_KNOWLEDGE_OF`, `IS_PART_OF_MODULE`, `REQUIRES`, `EXPLAINS`, `CAUSES`, `PREVENTS`

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| LLM | GPT-4o (OpenAI API) |
| Vector Store | ChromaDB |
| Knowledge Graph | Neo4j |
| Orchestration | LangChain |
| Embeddings | OpenAI Embeddings |
| PDF Parsing | PyPDF2 |
| PPT Parsing | python-pptx |
| NLP Extraction | spaCy, NLTK |
| Config | python-dotenv |

---

## 📁 Project Structure

```
cder-rag-project/
├── .env                        # API keys (never commit this!)
├── .gitignore
├── requirements.txt
├── cder_vector_db/             # ChromaDB persistent store
├── data/
│   └── cder_texts/             # CDER curriculum PDFs and PPTs
├── llm_only_demo.py            # System A: LLM baseline
├── graph_rag_demo.py           # System C: Graph RAG demo
├── eval_pipeline.py            # Evaluation pipeline (Week 8)
├── eval_pipeline_week9.py      # Evaluation pipeline (Week 9 final)
├── classify_dataset.json       # Question classification dataset
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/akshitha39/cder-rag-project.git
cd cder-rag-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your `.env` file
```
OPENAI_API_KEY=your_openai_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

### 4. Set up Neo4j
- Download [Neo4j Desktop](https://neo4j.com/download/)
- Create a new database and start it
- Update credentials in `.env`

### 5. Run the systems

```bash
# System A: LLM Only
python llm_only_demo.py

# System C: Graph RAG
python graph_rag_demo.py

# Run full evaluation
python eval_pipeline_week9.py
```

---

## 📋 Requirements

Create a `requirements.txt` with:
```
openai
langchain
langchain-community
chromadb
neo4j
spacy
nltk
PyPDF2
python-pptx
python-dotenv
```

---

## ⚠️ Known Issues & Fixes

| Issue | Fix Applied |
|---|---|
| LangChain v1+ `RetrievalQA` deprecated | Switched to `langchain-community` imports |
| Missing API key halting execution | Added `.env` file with `python-dotenv` |
| ChromaDB deprecation warnings | Used auto-persist behavior |
| Data duplication on re-embedding | Added length check via `vectordb.get()["ids"]` |

---

## 🔒 .gitignore

Make sure your `.gitignore` includes:
```
.env
__pycache__/
venv/
cder_vector_db/
*.pyc
```

---

## 📖 References

- [LangChain](https://github.com/langchain-ai/langchain)
- [Neo4j LangChain Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_graph/)
- [CDER Curriculum](https://tcpp.cs.gsu.edu/curriculum/?q=node/21245)
- [RAG Examples](https://github.com/reichenbch/RAG-examples)
