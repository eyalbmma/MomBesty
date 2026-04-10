# 🧠 Babys AI – RAG-Powered Parenting Assistant

A production-oriented AI backend that combines **LLM reasoning with Retrieval-Augmented Generation (RAG)** to deliver accurate, context-aware answers for parenting and baby-care scenarios.

---

## 🚀 Overview

This project implements a **FastAPI-based AI system** that:

* Accepts user questions via `/ask_final`
* Retrieves relevant knowledge from a structured dataset (`rag_clean`)
* Uses semantic similarity (embeddings + cosine similarity)
* Injects retrieved context into an LLM (OpenAI)
* Produces grounded, empathetic responses in Hebrew

---

## 🧩 Key Features

### 🔎 Retrieval-Augmented Generation (RAG)

* Local SQLite knowledge base (`rag_clean`)
* Precomputed embeddings (`text-embedding-3-small`)
* Real-time similarity search (cosine)
* Top-K context injection into LLM

### 🤖 LLM Integration

* OpenAI GPT-based response generation
* Context-aware answers
* Controlled tone (empathetic + structured)

### ⚡ Backend API

* Built with **FastAPI**
* Main endpoint: `/ask_final`
* Clean separation of concerns:

  * `api.py` → routing
  * `rag_retrieval.py` → retrieval logic
  * `chat_engine.py` → LLM orchestration

### 🧠 Intelligent Context Handling

* Dynamic context construction
* Prioritization of top-ranked results
* Token-preserving behavior (for codes, IDs)

---

## 🏗️ Architecture

```
User Question
     ↓
Embedding (query)
     ↓
Semantic Search (rag_clean)
     ↓
Top-K Results
     ↓
Context Builder
     ↓
LLM (GPT)
     ↓
Final Answer
```

---

## 📂 Project Structure

```
.
├── api.py                    # FastAPI entry point
├── chat_engine.py           # LLM logic and prompt building
├── rag_retrieval.py         # RAG retrieval + similarity
├── embed_db.py              # Embedding generation for DB
├── forum_api.py             # Forum-related logic
├── content_api.py           # Content endpoints
├── tracker_api.py           # Tracking features
├── create_*_tables.py       # DB schema scripts
├── insert_curated.py        # Data insertion utilities
├── README.md
├── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
CIRCLES_ADMIN_KEY=your_secure_admin_key
```

---

### 5. Run the server

```bash
uvicorn api:app --reload
```

Open Swagger:

```
http://127.0.0.1:8000/docs
```

---

## 🔬 RAG Pipeline

### Data Source

* Table: `rag_clean`
* Fields:

  * `question`
  * `answer`
  * `embedding`
  * `source`, `tags`, `is_active`

### Embedding Generation

```bash
python embed_db.py
```

---

## 🧪 Example Request

```json
POST /ask_final

{
  "question": "הבת שלי בת חודש וחצי ומראה סימני רעב",
  "user_id": "test-user-1",
  "conversation_id": "test-1"
}
```

---

## 📈 What Makes This Project Stand Out

✔ Real RAG (not prompt-only AI)
✔ End-to-end pipeline: retrieval → reasoning → response
✔ Production-aware design (logging, fallback, env separation)
✔ Clean architecture separation
✔ Hebrew NLP + real-world domain (parenting)
✔ Debugging + validation methodology (top-K verification)

---

## 🔐 Security & Best Practices

* No secrets committed to repository
* Uses environment variables for sensitive data
* Database and raw data excluded via `.gitignore`
* Designed for safe public sharing

---

## 🧠 Future Improvements

* Vector DB integration (FAISS / Pinecone)
* Caching layer for frequent queries
* Fine-tuned prompts per intent
* Multi-language support
* Real-time analytics / observability

---

## 👨‍💻 Author

**Eyal Berda**

Full Stack Developer | AI Systems Builder
Specializing in:

* .NET / React / Microservices
* AI integrations (LLM + RAG)
* Scalable backend architectures

---

## ⭐ Final Note

This project demonstrates not just usage of AI —
but **how to engineer AI systems correctly**:

* grounded
* explainable
* production-ready

