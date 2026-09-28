## Candidate
**Name:** Roshan Shrestha  

---

## Task Summary

This project implements a backend system with two core REST APIs:

1. **Document Ingestion API**
2. **Conversational RAG API**

The system supports document processing, semantic search, multi-turn conversation, and interview booking.

---

## System Architecture

The system follows a modular architecture:

- API Layer → FastAPI endpoints  
- Service Layer → Business logic (RAG, embeddings, booking)  
- Data Layer → SQLite + Qdrant  
- Memory Layer → Redis  

---

## 🔹 1. Document Ingestion API

### Endpoint
`POST /api/ingest`

### Features
- Upload `.txt` and `.pdf` files
- Extract text from files
- Apply **two token-based chunking strategies** (using the embedding model's tokenizer):
  - Fixed chunking: packs whole sentences up to `chunk_size` tokens, with `overlap` tokens carried between chunks
  - Paragraph chunking: detects paragraphs (including PDF text with single line breaks), merges small ones and splits long ones by sentence
  - No chunk exceeds the embedding model's limit (254 content tokens for `all-MiniLM-L6-v2`)
- Optional form fields `chunk_size` and `overlap` (defaults from `CHUNK_SIZE` / `CHUNK_OVERLAP`)
- Upload size limit (`MAX_UPLOAD_SIZE_MB`, default 10 MB)
- Generate embeddings using Sentence Transformers
- Store embeddings in **Qdrant vector database**
- Store document metadata in SQLite

---

## 🔹 2. Conversational RAG API

### Endpoint
`POST /api/chat`

### Features
- Custom RAG implementation (**no RetrievalQAChain used**)
- Rewrites follow-up questions into standalone search queries using the LLM and chat history
- Retrieve relevant chunks from Qdrant, dropping results below `MIN_SIMILARITY_SCORE`
- Optional request fields: `top_k`, `document_ids`, `chunking_strategy` (filters)
- Generate response using LLM (Ollama)
- Returns `sources` (filename, chunk index, score, snippet) with each answer

### Example Request

```json
{
  "session_id": "abc",
  "message": "What language is it written in?",
  "top_k": 3,
  "document_ids": [1, 2],
  "chunking_strategy": "paragraph"
}
```

---

## Multi-turn Chat Support

- Chat memory implemented using **Redis**
- Maintains conversation history per session
- History expires after `CHAT_HISTORY_TTL_SECONDS` and is capped at `CHAT_HISTORY_MAX_MESSAGES`
- Supports follow-up questions

---

## Interview Booking Feature

The system supports booking detection using LLM:

### Example Input

```bash
I want to book an interview. Name: Roshan, Email: roshan@example.com, Date: 2026-04-20, Time: 14:00
```


### Functionality
- Extracts structured data
- Stores booking in database
- Returns confirmation message

---

## Additional APIs

### GET `/api/documents`
- Returns metadata for all ingested documents (without the raw text)

### DELETE `/api/documents/{id}`
- Deletes a document from SQLite, its vectors from Qdrant and its uploaded file

### GET `/health`
- Reports the status of the database, Qdrant, Redis, Ollama and the embedding model

### GET `/api/bookings`
- Returns all interview bookings

---

## Technologies Used

- FastAPI
- Qdrant (Vector DB)
- Redis (Chat Memory)
- SQLite (Metadata Storage)
- Sentence Transformers (Embeddings)
- Ollama (LLM)

---

## How to Run

### Option A: Docker Compose (recommended)

```bash
docker compose up --build
```

This starts the API, Qdrant, Redis and Ollama, and pulls the LLM (`llama3` by default,
override with `LLM_MODEL`). The first start takes a while because it downloads the models.
Check that everything is up with `GET http://127.0.0.1:8000/health`.

### Option B: Run locally

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Configure (optional)
Copy `.env.example` to `.env` and adjust. All settings have sensible defaults.

#### 3. Start services

```bash
docker run -p 6333:6333 qdrant/qdrant
docker run -p 6379:6379 redis
ollama pull llama3
```

#### 4. Run Backend

```bash
uvicorn app.main:app --reload
```

### API Documentation

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## Constraints Followed

- No FAISS / Chroma used  
- No RetrievalQAChain used  
- Custom RAG pipeline implemented  
- Redis used for chat memory

## Screenshots

### Document Ingestion
![Swagger UI](assets/Document_Ingestion_1.png)

### Document Ingestion Response
![Chat](assets/Document_Ingestion_response.png)

### Document Database
![Booking](assets/Document_db.png)

### Conversational RAG
![Booking](assets/Conversational_RAG_first_question.png)

### Conversational RAG Response
![Booking](assets/Conversational_RAG_first_question_answer.png)

### Multi Turn Memory Test
![Booking](assets/Test_Multi_turn_memory.png)

### Multi Turn Memory Test Response
![Booking](assets/Test_Multi_turn_memory_test_response.png)

### Booking
![Booking](assets/Booking_1.png)

### Booking Response
![Booking](assets/Booking_1_response.png)


### Booking Database
![Booking](assets/booking_db.png)


