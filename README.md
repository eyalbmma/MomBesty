# Postpartum Support API

AI-powered backend for a postpartum support platform, combining conversational guidance, structured content, community interaction, and real-time tracking for new parents.

This project powers a Hebrew-language application designed to support mothers during the postpartum period through personalized responses, educational content, and social features.

---

## 🚀 Overview

This is a **FastAPI-based backend** built around a modular architecture, integrating:

- 🤖 AI chat (OpenAI-powered)
- 📚 Structured content (topics & articles)
- 💬 Community forum (posts, comments, empathy)
- 📊 Baby & mother tracking (sleep, feeding, pumping, etc.)
- 🌐 "Circles" directory (professionals, groups, events)
- 🔔 Push notifications (Expo)
- 🧠 Embedding pipeline (for future RAG capabilities)

The system is currently designed as a **single-service backend** using **SQLite**, with optional scripts for data enrichment and daily engagement workflows.

---

## ✨ Key Features

### 🤖 AI Chat
- Endpoint: `POST /ask_final`
- Intent detection & safety handling
- Conversation memory via `conversation_id`
- Uses OpenAI (`gpt-4o-mini`)
- Designed for emotional + informational postpartum support

> ⚠️ Note: RAG infrastructure exists but is not currently connected to the live chat flow.

---

### 📚 Content API
- Topics & articles management
- Search via SQL (`LIKE`)
- Structured educational content

---

### 💬 Forum
- Create posts & comments
- Empathy (likes)
- Reporting system
- Notifications & unread count
- Expo push integration

---

### 📊 Tracker
Track baby and postpartum activities:
- Feeding
- Sleep (start/stop sessions)
- Pumping
- Diapers
- Medicine

---

### 🌐 Circles
- Professionals directory
- Support groups
- Events
- Admin CRUD endpoints (secured via header key)

---

### 🔔 Daily Support Engine (Scripts)
- Generates daily support plans
- Sends push notifications
- Based on postpartum profiles & predefined messages

---

### 🧠 Embeddings & RAG (Planned)
- Embeddings stored in `rag_clean`
- Batch processing via `embed_db.py`
- Not yet connected to `/ask_final`

---

## 🧱 Architecture

Client (Mobile / Web)  
        ↓  
FastAPI (api.py)  
        ↓  
---------------------------------  
| Chat Engine (OpenAI)          |  
| Content API (SQLite)          |  
| Forum API + Push (Expo)       |  
| Tracker API                  |  
| Circles API (Admin + Public) |  
---------------------------------  
        ↓  
SQLite DB (/data/rag.db)

---

## 📁 Project Structure

api.py                 # Main FastAPI app  
chat_engine.py         # AI chat logic  
content_api.py         # Content routes  
forum_api.py           # Forum routes  
tracker_api.py         # Tracking logic  
circles_api.py         # Circles + admin  
push_utils.py          # Expo push logic  

embed_db.py            # Embedding pipeline  
daily_support_plan.py  # Daily plan logic  
daily_support_sender.py # Push execution  

create_*_tables.py     # DB setup scripts  

rag.db.sql             # ⚠️ DB export (review before publishing)

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|------------|
| OPENAI_API_KEY | Required for chat & embeddings |
| CIRCLES_ADMIN_KEY | Required for admin routes |

---

## 🛠️ Local Setup

git clone <repo>  
cd project  
python -m venv venv  

Windows:
venv\Scripts\activate  

Mac/Linux:
source venv/bin/activate  

pip install fastapi uvicorn pydantic openai requests  

(Optional)  
pip install beautifulsoup4 playwright  

---

## 🗄️ Database Setup

Default path in code:
/data/rag.db

You can:
- Create the directory manually  
- Or update DB_PATH in the code  

Run setup scripts:

python create_content_tables.py  
python create_forum_tables.py  
python create_tracker_tables.py  

---

## ▶️ Running the API

uvicorn api:app --reload --host 0.0.0.0 --port 8000  

Docs:
http://localhost:8000/docs  
http://localhost:8000/redoc  

---

## 🔌 API Example

Request:

POST /ask_final

{
  "question": "אני עייפה כל הזמן אחרי הלידה, זה נורמלי?",
  "conversation_id": "abc123"
}

Response:

{
  "answer": "...",
  "intent": "emotional_support",
  "used_gpt": true
}

---

## 🔐 Security Notes

⚠️ Important before production use:

- No authentication system (user_id is client-controlled)
- Admin routes use x-admin-key
- Debug endpoint exposes push tokens (/forum/debug/push-tokens)
- Remove or secure sensitive endpoints before deployment
- Do NOT commit real database files or user data

---

## ⚠️ Disclaimer

This system is NOT a medical device and does not replace professional medical advice.

---

## 🚧 Future Improvements

- Connect RAG to /ask_final  
- Add authentication (JWT / OAuth)  
- Move to PostgreSQL for scalability  
- Add background workers (Celery / Redis)  
- Add semantic search  
- Improve moderation system  

---

## 💡 Tech Stack

- Python  
- FastAPI  
- SQLite  
- OpenAI API  
- Expo Push  
- Requests  

---

## 📌 Summary

A production-style backend combining:

- AI chat  
- Community features  
- Content delivery  
- Real-time tracking  

Designed as a foundation for a scalable postpartum support product.